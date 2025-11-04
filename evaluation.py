"""Evaluation and metrics module for forecasting models.

Calculates comprehensive metrics:
- MAE, RMSE, MAPE, R², SMAPE
- Directional accuracy
- Forecast bias
- Cross-validation scores
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    """Comprehensive evaluation metrics calculator."""
    
    def __init__(self):
        """Initialize Evaluator."""
        self.metrics_history = []
    
    @staticmethod
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Error."""
        return mean_absolute_error(actual, predicted)
    
    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(actual, predicted))
    
    @staticmethod
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = actual != 0
        if mask.sum() == 0:
            return np.inf
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
        diff = np.abs(actual - predicted) / denominator
        diff[denominator == 0] = 0
        return 100.0 * np.mean(diff)
    
    @staticmethod
    def r2(actual: np.ndarray, predicted: np.ndarray) -> float:
        """R-Squared (Coefficient of Determination)."""
        return r2_score(actual, predicted)
    
    @staticmethod
    def median_ae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Median Absolute Error."""
        return np.median(np.abs(actual - predicted))
    
    @staticmethod
    def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate directional accuracy (% of correct direction changes)."""
        actual_direction = np.diff(actual) > 0
        predicted_direction = np.diff(predicted) > 0
        correct = (actual_direction == predicted_direction).sum()
        return (correct / len(actual_direction)) * 100
    
    @staticmethod
    def forecast_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate forecast bias (mean error)."""
        return np.mean(predicted - actual)
    
    @staticmethod
    def mre(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Mean Relative Error."""
        mask = actual != 0
        if mask.sum() == 0:
            return np.inf
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))
    
    def calculate_all_metrics(self, actual: np.ndarray, 
                             predicted: np.ndarray,
                             include_error_analysis: bool = True) -> Dict:
        """Calculate all available metrics.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            include_error_analysis: Whether to include directional accuracy, bias, etc.
            
        Returns:
            Dict: Dictionary of all calculated metrics.
        """
        metrics = {
            'mae': self.mae(actual, predicted),
            'rmse': self.rmse(actual, predicted),
            'mape': self.mape(actual, predicted),
            'smape': self.smape(actual, predicted),
            'r2': self.r2(actual, predicted),
            'median_ae': self.median_ae(actual, predicted),
            'mre': self.mre(actual, predicted),
        }
        
        if include_error_analysis:
            metrics['directional_accuracy'] = self.directional_accuracy(actual, predicted)
            metrics['forecast_bias'] = self.forecast_bias(actual, predicted)
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def compare_forecasts(self, actual: np.ndarray,
                         forecast1: np.ndarray,
                         forecast2: np.ndarray,
                         labels: Tuple[str, str] = ('Model 1', 'Model 2')) -> Dict:
        """Compare two forecasts.
        
        Args:
            actual: Actual values.
            forecast1: First forecast.
            forecast2: Second forecast.
            labels: Labels for the two forecasts.
            
        Returns:
            Dict: Comparison metrics.
        """
        metrics1 = self.calculate_all_metrics(actual, forecast1)
        metrics2 = self.calculate_all_metrics(actual, forecast2)
        
        comparison = {
            labels[0]: metrics1,
            labels[1]: metrics2,
            'winner': labels[0] if metrics1['mape'] < metrics2['mape'] else labels[1],
            'improvement': abs(metrics1['mape'] - metrics2['mape']),
        }
        
        return comparison
    
    def analyze_errors(self, actual: np.ndarray,
                      predicted: np.ndarray) -> Dict:
        """Analyze prediction errors.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            
        Returns:
            Dict: Error analysis statistics.
        """
        errors = actual - predicted
        
        analysis = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'abs_errors': np.abs(errors),
            'positive_errors': (errors > 0).sum(),
            'negative_errors': (errors < 0).sum(),
            'error_distribution': {
                'q25': np.percentile(errors, 25),
                'q50': np.percentile(errors, 50),  # Median
                'q75': np.percentile(errors, 75),
            }
        }
        
        return analysis
    
    def calculate_baseline_metrics(self, actual: np.ndarray,
                                  method: str = 'naive') -> Dict:
        """Calculate baseline forecast metrics.
        
        Args:
            actual: Actual values.
            method: Baseline method ('naive' or 'seasonal_naive').
            
        Returns:
            Dict: Baseline metrics.
        """
        if method == 'naive':
            # Naive forecast: previous value
            baseline_forecast = np.roll(actual, 1)[1:]
            actual_trimmed = actual[1:]
        elif method == 'seasonal_naive':
            # Seasonal naive (assuming 7-day seasonality)
            baseline_forecast = np.roll(actual, 7)[7:]
            actual_trimmed = actual[7:]
        else:
            raise ValueError(f"Unknown baseline method: {method}")
        
        return self.calculate_all_metrics(actual_trimmed, baseline_forecast)
    
    def get_improvement_over_baseline(self, actual: np.ndarray,
                                      predicted: np.ndarray,
                                      baseline_method: str = 'naive') -> Dict:
        """Calculate improvement over baseline forecast.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            baseline_method: Baseline method to compare against.
            
        Returns:
            Dict: Improvement metrics.
        """
        model_metrics = self.calculate_all_metrics(actual, predicted)
        baseline_metrics = self.calculate_baseline_metrics(actual, baseline_method)
        
        improvements = {}
        for key in ['mae', 'rmse', 'mape', 'smape']:
            if key in model_metrics and key in baseline_metrics:
                improvements[f'{key}_improvement'] = (
                    (baseline_metrics[key] - model_metrics[key]) / baseline_metrics[key] * 100
                )
        
        return improvements
    
    def print_metrics_report(self, metrics: Dict, title: str = "Metrics Report"):
        """Print a formatted metrics report.
        
        Args:
            metrics: Dictionary of metrics.
            title: Title for the report.
        """
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key.upper():.<40} {value:.4f}")
            else:
                print(f"  {key.upper():.<40} {value}")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    evaluator = Evaluator()
    
    # Example
    actual = np.array([100, 110, 105, 120, 115, 125, 130])
    predicted = np.array([102, 108, 107, 118, 117, 128, 128])
    
    metrics = evaluator.calculate_all_metrics(actual, predicted)
    evaluator.print_metrics_report(metrics, "Example Metrics Report")
