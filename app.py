import logging
import html
import json
import os
from pathlib import Path

import requests
import streamlit as st
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials


PROJECT_DIR = Path(__file__).resolve().parent
FALLBACK_COVER_URL = "https://i.postimg.cc/0QNxYz4V/social.png"


@st.cache_data
def load_recommendations():
    catalog_path = PROJECT_DIR / "recommendations.json"
    with catalog_path.open(encoding="utf-8") as file:
        catalog = json.load(file)
    songs = catalog.get("songs")
    if not isinstance(songs, list) or not songs:
        raise ValueError("recommendations.json must contain a non-empty songs list.")
    if any(
        not isinstance(song, dict)
        or not isinstance(song.get("song"), str)
        or not isinstance(song.get("artist"), str)
        or not isinstance(song.get("recommendations"), list)
        for song in songs
    ):
        raise ValueError("recommendations.json contains an invalid song entry.")
    return songs


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


def recommend(song, catalog, spotify_client):
    selected = next((entry for entry in catalog if entry["song"] == song), None)
    if selected is None:
        return []

    recommendations = []
    for entry in selected["recommendations"]:
        title = entry["song"]
        artist = entry["artist"]
        cover_result = get_song_album_cover_url(title, artist, spotify_client)
        recommendations.append(
            {
                "song": title,
                "artist": artist,
                "cover_result": cover_result,
            }
        )
    return recommendations


st.set_page_config(
    page_title="Music Recommender System",
    page_icon="🎵",
    layout="centered",
)
st.markdown(
    """
    <style>
    .block-container {
        max-width: 680px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stButton"] > button {
        color: #ff4b4b;
        background: transparent;
        border: 2px solid #ff4b4b;
        border-radius: 0.35rem;
        font-weight: 600;
    }
    div[data-testid="stButton"] > button:hover {
        color: #ffffff;
        background: #ff4b4b;
        border-color: #ff4b4b;
    }
    .recommendation-title {
        overflow: hidden;
        margin-bottom: 0.65rem;
        font-family: monospace;
        font-size: 0.84rem;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    div[data-testid="stImage"] img {
        aspect-ratio: 1 / 1;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Music Recommender System")

try:
    song_catalog = load_recommendations()
except FileNotFoundError:
    st.error(
        "The recommendation catalog is missing. Run train_model.py and publish "
        "the generated recommendations.json file."
    )
    st.stop()
except (OSError, json.JSONDecodeError, ValueError) as error:
    st.error(f"Could not load the recommendation catalog: {error}")
    st.stop()

spotify_client = get_spotify_client()

selected_song = st.selectbox(
    "Type or select a song from the dropdown",
    list(dict.fromkeys(entry["song"] for entry in song_catalog)),
)

if st.button("Show Recommendation"):
    recommendations = recommend(selected_song, song_catalog, spotify_client)
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
                st.markdown(
                    '<div class="recommendation-title">'
                    f'{html.escape(recommendation["song"])}'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.image(
                    recommendation["cover_result"][0],
                    use_container_width=True,
                )
