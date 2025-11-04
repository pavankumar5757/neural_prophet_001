import os
import pandas as pd
import matplotlib.pyplot as plt

def yearly_train_test_split(data, train_years, test_years):
    train = data[data['ds'].dt.year.isin(train_years)].reset_index(drop=True)
    test = data[data['ds'].dt.year.isin(test_years)].reset_index(drop=True)
    return train, test

def plot_actual_vs_predicted(ds, actual, predicted, out_path):
    plt.figure(figsize=(10,6))
    plt.plot(ds, actual, label="Actual", marker='o')
    plt.plot(ds, predicted, label="Predicted", linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_forecasts(forecast, data, out_path, periods=[30, 180, 365]):
    plt.figure(figsize=(14,7))
    plt.plot(data['ds'], data['y'], label='Historical', color='blue')
    for p in periods:
        plt.plot(forecast['ds'][-p:], forecast['yhat1'][-p:], label=f'Forecast next {p}d')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.title('Future Forecasts')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
