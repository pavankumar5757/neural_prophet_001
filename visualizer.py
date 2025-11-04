"""Visualization module for time series forecasting.

Generates comprehensive plots:
- Time series forecasts with confidence intervals
- Residual diagnostics
- Component decomposition
- Metrics comparison
- Cross-validation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class Visualizer:
    """Create comprehensive forecast visualizations."""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (14, 7), dpi: int = 300):
        """Initialize Visualizer.
        
        Args:
            style: Matplotlib style.
            figsize: Default figure size.
            dpi: Resolution for saving.
        """
        plt.style.use(style)
        self.figsize = figsize
        self.dpi = dpi
    
    def plot_forecast(self, history: pd.DataFrame,
                     forecast: pd.DataFrame,
                     title: str = "Time Series Forecast",
                     save_path: Optional[str] = None) -> None:
        """Plot historical data and forecast.
        
        Args:
            history: Historical data with 'ds' and 'y' columns.
            forecast: Forecast with 'ds' and 'yhat' columns.
            title: Plot title.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Plot historical
        plt.plot(history['ds'], history['y'], 'o-', label='Historical', linewidth=2, markersize=4)
        
        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat1'], 's--', label='Forecast', linewidth=2, markersize=4, color='red')
        
        # Add confidence interval if available
        if 'yhat1_lower' in forecast.columns and 'yhat1_upper' in forecast.columns:
            plt.fill_between(forecast['ds'],
                            forecast['yhat1_lower'],
                            forecast['yhat1_upper'],
                            alpha=0.2, color='red', label='95% Confidence Interval')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_actual_vs_predicted(self, actual: np.ndarray,
                                predicted: np.ndarray,
                                title: str = "Actual vs Predicted",
                                save_path: Optional[str] = None) -> None:
        """Plot actual vs predicted values.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            title: Plot title.
            save_path: Path to save the plot.
        """
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        plt.plot(actual, 'o-', label='Actual', linewidth=2, markersize=5)
        plt.plot(predicted, 's--', label='Predicted', linewidth=2, markersize=5)
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_residuals(self, actual: np.ndarray,
                      predicted: np.ndarray,
                      title: str = "Residuals Analysis",
                      save_path: Optional[str] = None) -> None:
        """Plot residual analysis.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            title: Plot title.
            save_path: Path to save the plot.
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        
        # Residuals over time
        axes[0, 0].plot(residuals, 'o-', color='steelblue', markersize=4)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Residual Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residual Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted scatter
        axes[1, 1].scatter(actual, predicted, alpha=0.6, s=50, color='steelblue')
        axes[1, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[1, 1].set_title('Actual vs Predicted Scatter', fontweight='bold')
        axes[1, 1].set_xlabel('Actual Value')
        axes[1, 1].set_ylabel('Predicted Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, float],
                               title: str = "Metrics Comparison",
                               save_path: Optional[str] = None) -> None:
        """Plot metrics comparison bar chart.
        
        Args:
            metrics_dict: Dictionary of metric names to values.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        metrics = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
        
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v + max(values)*0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_cv_results(self, cv_scores: List[float],
                       fold_labels: Optional[List[str]] = None,
                       title: str = "Cross-Validation Results",
                       save_path: Optional[str] = None) -> None:
        """Plot cross-validation results.
        
        Args:
            cv_scores: List of CV fold scores.
            fold_labels: Labels for each fold.
            title: Plot title.
            save_path: Path to save the plot.
        """
        if fold_labels is None:
            fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.dpi)
        
        colors = ['green' if score > np.mean(cv_scores) else 'orange' for score in cv_scores]
        ax.bar(fold_labels, cv_scores, color=colors, edgecolor='black', alpha=0.7)
        
        ax.axhline(y=np.mean(cv_scores), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_scores):.4f}')
        ax.axhline(y=np.mean(cv_scores) - np.std(cv_scores), color='r', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=np.mean(cv_scores) + np.std(cv_scores), color='r', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.set_ylabel('Score (MAPE)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()
    
    def plot_error_distribution(self, actual: np.ndarray,
                               predicted: np.ndarray,
                               title: str = "Error Distribution",
                               save_path: Optional[str] = None) -> None:
        """Plot error distribution analysis.
        
        Args:
            actual: Actual values.
            predicted: Predicted values.
            title: Plot title.
            save_path: Path to save the plot.
        """
        errors = actual - predicted
        pct_errors = (errors / actual) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)
        
        # Absolute errors
        axes[0].hist(np.abs(errors), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('Absolute Error Distribution', fontweight='bold')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(x=np.mean(np.abs(errors)), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(np.abs(errors)):.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Percentage errors
        axes[1].hist(pct_errors, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_title('Percentage Error Distribution', fontweight='bold')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].axvline(x=np.mean(pct_errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pct_errors):.2f}%')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    visualizer = Visualizer()
    print("Visualizer module loaded successfully!")
