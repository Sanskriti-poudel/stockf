from django.shortcuts import render

# Create your views here.
# mlapi/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import os
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

# global caches
model_cache = {}
x_scaler_cache = {}
y_scaler_cache = {}

# base folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(_file_)))
MODEL_DIR = os.path.join(BASE_DIR, "stockprediction", "allsavedmodels", "gru_fundamental")

@csrf_exempt
def predict_ltp(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            symbol = data.get("symbol")
            features = data.get("features")  # should be a dict

            if not symbol or not features:
                return JsonResponse({"error": "Missing symbol or features"}, status=400)

            # paths
            model_path = os.path.join(MODEL_DIR, f"{symbol}_GRU.keras")
            x_scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.save")
            y_scaler_path = os.path.join(MODEL_DIR, f"{symbol}_y_scaler.save")

            # check files
            if not os.path.exists(model_path) or not os.path.exists(x_scaler_path):
                return JsonResponse({"error": f"Model or scaler for {symbol} not found"}, status=404)

            # cache model
            if symbol not in model_cache:
                model_cache[symbol] = load_model(model_path)

            # cache scalers
            if symbol not in x_scaler_cache:
                x_scaler_cache[symbol] = joblib_load(x_scaler_path)
                if os.path.exists(y_scaler_path):
                    y_scaler_cache[symbol] = joblib_load(y_scaler_path)

            model = model_cache[symbol]
            x_scaler = x_scaler_cache[symbol]
            y_scaler = y_scaler_cache.get(symbol, None)

            # convert features dict to dataframe row
            import pandas as pd
            feature_order = [
                '% Change', 'High', 'Low', 'Open', 'Qty.', 'Turnover',
                'SECTOR', 'EPS', 'P/E_ratio', 'bookvalue', 'PBV',
                'SHARESOUTSTANDING', 'SENTIMENT_SCORE', 'RSI', 'MACD', 'MACD_signal'
            ]

            input_df = pd.DataFrame([features])
            input_scaled = x_scaler.transform(input_df[feature_order].values)
            input_scaled = np.expand_dims(input_scaled, axis=1)  # shape (1, 1, features)

            pred = model.predict(input_scaled, verbose=0)[0][0]

            # inverse transform if y_scaler exists
            if y_scaler:
                pred = y_scaler.inverse_transform([[pred]])[0][0]

            pred = max(pred, 0)

            return JsonResponse({
                "symbol": symbol,
                "predicted_ltp": round(float(pred), 2)
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"message": "Only POST allowed."})