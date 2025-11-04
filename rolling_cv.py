"""Rolling-origin cross-validation for time series forecasting.

Implements advanced rolling-origin validation strategy:
- Year-by-year expanding windows
- Multiple CV folds
- Comprehensive metrics per fold
- Temporal data integrity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from neuralprophet import NeuralProphet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import CrossValidationConfig, ModelConfig, DataConfig
from model_trainer import ModelTrainer
from evaluation import Evaluator


class RollingOriginCV:
    """Perform rolling-origin cross-validation."""
    
    def __init__(self, cv_config: CrossValidationConfig,
                 model_config: ModelConfig,
                 data_config: DataConfig):
        """Initialize RollingOriginCV.
        
        Args:
            cv_config: CrossValidationConfig object.
            model_config: ModelConfig object.
            data_config: DataConfig object.
        """
        self.cv_config = cv_config
        self.model_config = model_config
        self.data_config = data_config
        self.evaluator = Evaluator()
        self.fold_results = []
    
    def create_folds(self, data: pd.DataFrame,
                    date_col: str) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Create rolling-origin CV folds.
        
        Args:
            data: Full dataset.
            date_col: Date column name.
            
        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) folds.
        """
        folds = []
        years = sorted(data[date_col].dt.year.unique())
        
        print(f"\nCreating {len(years)-1} rolling-origin CV folds...")
        print(f"Years available: {years}")
        
        # Create expanding window splits
        for i in range(1, len(years)):
            train_years = years[:i]
            test_year = years[i]
            
            train = data[data[date_col].dt.year.isin(train_years)].reset_index(drop=True)
            test = data[data[date_col].dt.year == test_year].reset_index(drop=True)
            
            if len(train) > 0 and len(test) > 0:
                folds.append((train, test, train_years, test_year))
                print(f"  Fold {len(folds)}: Train years {train_years} ({len(train)} samples) -> Test year {test_year} ({len(test)} samples)")
        
        return folds
    
    def train_and_evaluate_fold(self, fold_idx: int,
                               train_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               date_col: str,
                               target_col: str) -> Dict:
        """Train model on fold and evaluate.
        
        Args:
            fold_idx: Fold index.
            train_data: Training data.
            test_data: Test data.
            date_col: Date column name.
            target_col: Target column name.
            
        Returns:
            Dict: Fold results and metrics.
        """
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        # Prepare data for NeuralProphet
        train_np = train_data[[date_col, target_col]].copy()
        train_np.columns = ['ds', 'y']
        
        test_np = test_data[[date_col, target_col]].copy()
        test_np.columns = ['ds', 'y']
        
        # Train model
        trainer = ModelTrainer(self.model_config, f"neuralprophet_fold_{fold_idx+1}")
        trainer.initialize_model()
        
        train_result = trainer.train(train_np, verbose=False)
        
        if train_result['status'] != 'success':
            print(f"Training failed for fold {fold_idx + 1}")
            return {
                'fold_idx': fold_idx,
                'status': 'failed',
                'error': train_result.get('error')
            }
        
        # Predict
        forecast = trainer.predict(test_np)
        
        # Extract predictions
        actual = test_np['y'].values
        predicted = forecast['yhat1'].values[:len(actual)]
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(actual, predicted)
        
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        
        return {
            'fold_idx': fold_idx,
            'status': 'success',
            'metrics': metrics,
            'actual': actual,
            'predicted': predicted,
            'trainer': trainer
        }
    
    def cross_validate(self, data: pd.DataFrame) -> Dict:
        """Execute complete rolling-origin cross-validation.
        
        Args:
            data: Full dataset with date and target columns.
            
        Returns:
            Dict: Cross-validation results and aggregate statistics.
        """
        print("\n" + "="*70)
        print("ROLLING-ORIGIN CROSS-VALIDATION")
        print("="*70)
        
        date_col = self.data_config.date_column
        target_col = self.data_config.target_column
        
        # Create folds
        folds = self.create_folds(data, date_col)
        
        if len(folds) == 0:
            print("Error: No valid folds created.")
            return {'status': 'failed', 'error': 'No valid folds'}
        
        # Train and evaluate each fold
        fold_results = []
        for fold_idx, (train, test, train_years, test_year) in enumerate(folds):
            result = self.train_and_evaluate_fold(
                fold_idx, train, test, date_col, target_col
            )
            fold_results.append(result)
        
        # Aggregate results
        aggregate_results = self.aggregate_cv_results(fold_results)
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"\nTotal Folds: {len(fold_results)}")
        print(f"Successful Folds: {sum(1 for r in fold_results if r['status'] == 'success')}")
        print(f"\nAggregate Metrics:")
        for metric, value in aggregate_results['mean_metrics'].items():
            if isinstance(value, float):
                std_val = aggregate_results['std_metrics'].get(metric, 0)
                print(f"  {metric.upper():.<30} {value:.4f} ± {std_val:.4f}")
        print("\n" + "="*70 + "\n")
        
        self.fold_results = fold_results
        
        return {
            'status': 'success',
            'fold_results': fold_results,
            'aggregate_results': aggregate_results
        }
    
    def aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results.
        
        Args:
            fold_results: List of fold results.
            
        Returns:
            Dict: Aggregated statistics.
        """
        successful_folds = [r for r in fold_results if r['status'] == 'success']
        
        if len(successful_folds) == 0:
            return {'status': 'failed', 'error': 'No successful folds'}
        
        # Extract all metrics
        all_metrics = {}
        for fold in successful_folds:
            for key, value in fold['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Calculate mean and std
        mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        std_metrics = {k: np.std(v) for k, v in all_metrics.items()}
        
        return {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'min_metrics': {k: np.min(v) for k, v in all_metrics.items()},
            'max_metrics': {k: np.max(v) for k, v in all_metrics.items()},
            'fold_count': len(successful_folds)
        }
    
    def get_best_fold(self) -> Optional[int]:
        """Get index of best performing fold.
        
        Returns:
            int: Index of fold with lowest MAPE.
        """
        successful = [r for r in self.fold_results if r['status'] == 'success']
        if not successful:
            return None
        
        best_idx = min(range(len(successful)),
                      key=lambda i: successful[i]['metrics'].get('mape', float('inf')))
        return best_idx
    
    def print_cv_report(self) -> None:
        """Print detailed cross-validation report."""
        print("\n" + "="*70)
        print("DETAILED CROSS-VALIDATION REPORT")
        print("="*70)
        
        for fold in self.fold_results:
            if fold['status'] == 'success':
                print(f"\nFold {fold['fold_idx'] + 1}:")
                print(f"  MAPE: {fold['metrics']['mape']:.2f}%")
                print(f"  MAE: {fold['metrics']['mae']:.4f}")
                print(f"  RMSE: {fold['metrics']['rmse']:.4f}")
                print(f"  R²: {fold['metrics']['r2']:.4f}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    from config import get_config
    
    config = get_config()
    cv = RollingOriginCV(config.cv, config.model, config.data)
    print("RollingOriginCV module loaded successfully!")
