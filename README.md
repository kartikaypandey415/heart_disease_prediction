# Heart Health Check — Frontend

A Streamlit frontend for the provided KNN heart-disease model, scaler, and feature-column list.

## Files
- `app.py` — frontend and prediction logic
- `knn_heart_model.pkl` — trained KNN model
- `heart_scaler.pkl` — fitted StandardScaler
- `heart_columns.pkl` — expected feature order
- `requirements.txt` — Python dependencies

## Run locally
1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in the terminal.

## Important
This is an educational screening interface, not a medical diagnostic tool. The frontend assumes the saved model uses the feature columns included with this project and that the model expects scaled input.
