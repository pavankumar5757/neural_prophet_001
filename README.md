# neural_prophet_001

This project demonstrates rolling-origin yearly cross-validation and multi-horizon forecasting using NeuralProphet for time series electricity demand (or other univariate target). Year-by-year training and predictions are performed, and future forecasting is included for +30 days, 6 months, and 1 year extension beyond the training data.

## Overview
- **rolling_cv.py**: Main script for rolling yearly cross-validation and future forecasting.
- **utils.py**: Helper functions for yearly splits and plotting actual vs. predicted values and forecasts.
- **requirements.txt**: All Python dependencies.

## Usage
1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. **Prepare your data:**
    - Create a `data.csv` in the project root directory.
    - Required columns: `ds` (date: YYYY-MM-DD), `y` (target value)

3. **Run cross-validation and forecasting:**
    ```sh
    python rolling_cv.py
    ```
    - Actual-vs-predicted and future forecast plots are saved in `plots/`.

## Modular functions
- `rolling_cv.py` uses `utils.py` for data splitting and plotting for clarity and extensibility.
- Adapt `plot_actual_vs_predicted` or `plot_forecasts` as needed for your use-case.

## Outputs
- Year-wise plots: `plots/actual_vs_pred_YEAR.png` for each rolling test year
- Long-horizon prediction plot: `plots/long_horizon_forecasts.png`

## Notes
- Adjust the script for custom frequency, forecasting horizon, or NeuralProphet parameters as needed.
- Your data must be continuous with no missing days in column `ds` for daily forecast.

---
**Author:** pavankumar5757
