import logging
import os
import pickle
from pathlib import Path

import requests
import streamlit as st
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials


PROJECT_DIR = Path(__file__).resolve().parent
FALLBACK_COVER_URL = "https://i.postimg.cc/0QNxYz4V/social.png"


@st.cache_resource
def load_models():
    dataframe_path = next(
        (path for path in (PROJECT_DIR / "df.pkl", PROJECT_DIR / "df") if path.is_file()),
        None,
    )
    similarity_path = next(
        (
            path
            for path in (PROJECT_DIR / "similarity.pkl", PROJECT_DIR / "similarity")
            if path.is_file()
        ),
        None,
    )
    if dataframe_path is None or similarity_path is None:
        raise FileNotFoundError(
            "The recommender model files are missing. Add "
            "'spotify_millsongdata.csv' to the project folder and run "
            "'python train_model.py'."
        )

    with dataframe_path.open("rb") as file:
        music = pickle.load(file)
    with similarity_path.open("rb") as file:
        similarity = pickle.load(file)

    if not {"song", "artist"}.issubset(music.columns):
        raise ValueError("df.pkl must contain 'song' and 'artist' columns.")
    if len(similarity.shape) != 2 or similarity.shape != (len(music), len(music)):
        raise ValueError("similarity.pkl dimensions do not match df.pkl.")
    return music, similarity


def get_spotify_client():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    credentials = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
    )
    return Spotify(client_credentials_manager=credentials)


@st.cache_data(ttl=86400)
def get_itunes_album_cover_url(song_name, artist_name):
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{song_name} {artist_name}",
                "entity": "song",
                "limit": 1,
            },
            timeout=8,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except (requests.RequestException, ValueError):
        return FALLBACK_COVER_URL, "unavailable"

    if results:
        artwork_url = results[0].get("artworkUrl100")
        if artwork_url:
            return artwork_url.replace("100x100bb", "600x600bb"), None
    return FALLBACK_COVER_URL, "not_found"


def get_song_album_cover_url(song_name, artist_name, spotify_client):
    if spotify_client is not None:
        try:
            results = spotify_client.search(
                q=f"track:{song_name} artist:{artist_name}",
                type="track",
                limit=1,
            )
            tracks = results.get("tracks", {}).get("items", [])
            if tracks and tracks[0].get("album", {}).get("images"):
                return tracks[0]["album"]["images"][0]["url"], None
        except SpotifyException as error:
            logging.warning("Spotify artwork lookup failed; trying iTunes: %s", error)
    return get_itunes_album_cover_url(song_name, artist_name)


def recommend(song, music, similarity, spotify_client):
    matches = music.index[music["song"] == song]
    if matches.empty:
        return []

    song_index = matches[0]
    ranked_songs = sorted(
        enumerate(similarity[song_index]),
        key=lambda item: item[1],
        reverse=True,
    )
    recommendations = []
    for index, _score in ranked_songs:
        if index == song_index:
            continue
        artist = music.iloc[index]["artist"]
        title = music.iloc[index]["song"]
        recommendations.append(
            {
                "song": title,
                "artist": artist,
                "cover_result": get_song_album_cover_url(
                    title, artist, spotify_client
                ),
            }
        )
        if len(recommendations) == 5:
            break
    return recommendations


st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide",
)
st.title("Spotify Song Recommender")

try:
    music, similarity = load_models()
except FileNotFoundError:
    st.error(
        "The recommendation model is not configured. Add the model files "
        "before starting the app."
    )
    st.stop()
except (OSError, pickle.UnpicklingError, ValueError) as error:
    st.error(f"Could not load the recommender model: {error}")
    st.stop()

spotify_client = get_spotify_client()

selected_song = st.selectbox(
    "Type or select a song",
    music["song"].dropna().astype(str).unique(),
)

if st.button("Show Recommendations"):
    recommendations = recommend(
        selected_song,
        music,
        similarity,
        spotify_client,
    )
    if not recommendations:
        st.warning("No recommendations were found for this song.")
    else:
        missing_artwork = sum(
            recommendation["cover_result"][1] == "not_found"
            for recommendation in recommendations
        )
        unavailable_artwork = sum(
            recommendation["cover_result"][1] == "unavailable"
            for recommendation in recommendations
        )
        if missing_artwork:
            st.info(f"Album artwork was not found for {missing_artwork} song(s).")
        if unavailable_artwork:
            st.warning(
                "Could not contact the album-art service. Check your internet "
                "connection and try again."
            )

        columns = st.columns(len(recommendations))
        for column, recommendation in zip(columns, recommendations):
            with column:
                st.image(
                    recommendation["cover_result"][0],
                    use_container_width=True,
                )
                st.write(recommendation["song"])
                st.caption(recommendation["artist"])
