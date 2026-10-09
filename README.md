# House Price Prediction

A Streamlit app that estimates residential property prices in major cities in Pakistan.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The trained model is stored in `Random_forest_model.pkl` using Git LFS. The app also needs `scaler.pkl` and `feature_columns.pkl` beside `app.py`.

## Deployment

This app requires a persistent Streamlit server and is suited to Streamlit Community Cloud. Vercel's standard Python functions do not host Streamlit's server/websocket application; deploying this app there would require reworking it into a different application architecture.