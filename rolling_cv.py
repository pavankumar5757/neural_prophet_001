import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

from utils import yearly_train_test_split, plot_actual_vs_predicted, plot_forecasts


def rolling_origin_yearly_cv(data_path, forecast_col, plot_dir,
                             yearly_forecast_period=365, future_periods=[30, 180, 365]):
    data = pd.read_csv(data_path)
    data["ds"] = pd.to_datetime(data["ds"])
    years = data["ds"].dt.year.unique()
    mae_scores = []
    mse_scores = []
    preds_all, actuals_all = [], []

    for i in range(1, len(years)):
        train, test = yearly_train_test_split(data, years[:i], years[i:i+1])

        m = NeuralProphet()
        m.fit(train, freq="D")
        forecast = m.predict(test)

        y_pred = forecast[forecast_col].values
        y_true = test[forecast_col].values
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        mae_scores.append(mae)
        mse_scores.append(mse)
        preds_all.extend(list(y_pred))
        actuals_all.extend(list(y_true))

        plot_actual_vs_predicted(test["ds"], y_true, y_pred, os.path.join(plot_dir, f"actual_vs_pred_{years[i]}.png"))
        
    print("MAE scores:", mae_scores)
    print("MSE scores:", mse_scores)
    
    # Final fit for multi-horizon forecasting
    m = NeuralProphet()
    m.fit(data, freq="D")
    future = m.make_future_dataframe(data, periods=max(future_periods))
    forecast = m.predict(future)
    plot_forecasts(forecast, data, os.path.join(plot_dir, "long_horizon_forecasts.png"), periods=future_periods)

if __name__ == "__main__":
    # Example usage
    data_path = "data.csv"  # Should contain columns: ds (date), y (target)
    forecast_col = "yhat1"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    rolling_origin_yearly_cv(data_path, forecast_col, plot_dir)
