import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"artist", "song", "text"}
MAX_SONGS = 5000


def build_model(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place spotify_millsongdata.csv in the project folder or pass its "
            "path with --csv."
        )

    music = pd.read_csv(csv_path)
    music, song_vectors = prepare_model(music)
    similarity = cosine_similarity(song_vectors)

    with (PROJECT_DIR / "df.pkl").open("wb") as file:
        pickle.dump(music, file)
    with (PROJECT_DIR / "similarity.pkl").open("wb") as file:
        pickle.dump(similarity, file)
    build_recommendation_catalog(music, similarity)

    return len(music)


def build_recommendation_catalog(music, similarity, limit=5):
    songs = []
    for index, row in music.iterrows():
        ranked_indices = similarity[index].argsort()[::-1]
        recommendations = []
        for recommendation_index in ranked_indices:
            if recommendation_index == index:
                continue
            recommendation = music.iloc[recommendation_index]
            recommendations.append(
                {
                    "song": str(recommendation["song"]),
                    "artist": str(recommendation["artist"]),
                }
            )
            if len(recommendations) == limit:
                break
        songs.append(
            {
                "song": str(row["song"]),
                "artist": str(row["artist"]),
                "recommendations": recommendations,
            }
        )

    catalog_path = PROJECT_DIR / "recommendations.json"
    with catalog_path.open("w", encoding="utf-8") as file:
        json.dump({"songs": songs}, file, ensure_ascii=False, separators=(",", ":"))


def prepare_model(music):
    missing_columns = REQUIRED_COLUMNS.difference(music.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    music = music[["artist", "song", "text"]].dropna(subset=["artist", "song"])
    music["text"] = music["text"].fillna("").astype(str)
    if len(music) < 2:
        raise ValueError("The dataset must contain at least two songs.")

    music = music.sample(n=min(MAX_SONGS, len(music)), random_state=42)
    music = music.reset_index(drop=True)
    vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
    song_vectors = vectorizer.fit_transform(music["text"])
    return music, song_vectors


def main():
    parser = argparse.ArgumentParser(
        description="Build the song recommendation model from a lyrics CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_DIR / "spotify_millsongdata.csv",
        help="Path to a CSV with artist, song, and text columns.",
    )
    args = parser.parse_args()

    try:
        song_count = build_model(args.csv)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Created df.pkl and similarity.pkl using {song_count} songs "
        f"in {PROJECT_DIR}, plus recommendations.json."
    )


if __name__ == "__main__":
    main()
