# Spotify Music Recommender

## Run the web app on Windows

1. Install Python 3.13 or newer and make sure the Python launcher (`py`) is available.
2. Obtain a CSV dataset with `artist`, `song`, and `text` columns. Save it as
   `spotify_millsongdata.csv` in this folder. The dataset is not included.
3. Double-click `run_app.bat`, or run it from PowerShell:

   ```powershell
   .\run_app.bat
   ```

   The script creates a project-local `.venv`, installs dependencies, and starts
   Streamlit. Keep the terminal open and visit the `Local URL` it prints
   (normally <http://localhost:8501>).

4. If the model files do not exist yet, the app shows a setup message. In a
   second PowerShell window, from this folder, build them with:

   ```powershell
   .\.venv\Scripts\python.exe train_model.py
   ```

   To use a CSV stored elsewhere, pass its full path:

   ```powershell
   .\.venv\Scripts\python.exe train_model.py --csv "D:\Data\spotify_millsongdata.csv"
   ```

   The trainer uses up to 5,000 songs and creates `df.pkl` and `similarity.pkl`
   next to `app.py`. Restart Streamlit after training.

## Deploy on Streamlit Community Cloud

The lyrics dataset and generated model files are intentionally not committed to
GitHub. The app does not provide a dataset upload control. A deployment must
have `df.pkl` and `similarity.pkl` available next to `app.py` or it will show a
model-not-configured message.

To deploy, sign in at <https://share.streamlit.io>, select **Create app**, choose
`Soumya23698/Spotify_Recommendation_System`, branch `main`, and file path
`app.py`, then deploy. Configure the trained model files for the deployment
before expecting song recommendations.

## VS Code imports

Open this project folder in VS Code. Its workspace settings select the local
`.venv` interpreter. If Pylance still shows missing imports, press
`Ctrl+Shift+P`, select **Python: Select Interpreter**, and choose
`.venv\Scripts\python.exe`. If needed, install the dependencies in the selected
interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Album artwork

Album covers are fetched from the public iTunes Search API, so Spotify API
credentials are not required. If Spotify client credentials are configured in
`SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`, Spotify is tried first and
iTunes is used as a fallback. Covers are cached for a day; restart the app or
wait for the cache to expire if you need to refresh them.
