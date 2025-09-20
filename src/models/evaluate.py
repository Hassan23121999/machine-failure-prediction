"""
Model evaluation and monitoring utilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.results = {}
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_false_negative: float = 10000,  # Cost of undetected failure
        cost_false_positive: float = 500,    # Cost of unnecessary maintenance
    ) -> Dict[str, float]:
        """Calculate business-oriented metrics"""
        
        # Confusion matrix elements
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        
        # Calculate costs
        total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
        cost_per_prediction = total_cost / len(y_true)
        
        # Savings calculation (prevented failures)
        savings = tp * cost_false_negative
        
        # Maintenance efficiency
        maintenance_efficiency = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return {
            'total_cost': total_cost,
            'cost_per_prediction': cost_per_prediction,
            'savings_from_prevention': savings,
            'maintenance_efficiency': maintenance_efficiency,
            'false_alarms': fp,
            'missed_failures': fn,
            'caught_failures': tp,
            'correct_no_maintenance': tn
        }
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """Find optimal probability threshold"""
        
        thresholds = np.arange(0.1, 0.9, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'business':
                business_metrics = self.calculate_business_metrics(y_true, y_pred)
                score = -business_metrics['cost_per_prediction']  # Negative for minimization
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_threshold = thresholds[best_idx]
        best_score = scores[best_idx]
        
        return best_threshold, best_score
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ):
        """Plot performance metrics across different thresholds"""
        
        thresholds = np.arange(0.05, 0.95, 0.05)
        
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'cost': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            business_metrics = self.calculate_business_metrics(y_true, y_pred)
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['cost'].append(business_metrics['cost_per_prediction'])
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Precision & Recall
        axes[0, 0].plot(thresholds, metrics['precision'], label='Precision', marker='o')
        axes[0, 0].plot(thresholds, metrics['recall'], label='Recall', marker='s')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Precision & Recall vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # F1 Score
        axes[0, 1].plot(thresholds, metrics['f1'], color='green', marker='o')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score vs Threshold')
        axes[0, 1].grid(alpha=0.3)
        
        # Business Cost
        axes[1, 0].plot(thresholds, metrics['cost'], color='red', marker='o')
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('Cost per Prediction ($)')
        axes[1, 0].set_title('Business Cost vs Threshold')
        axes[1, 0].grid(alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)
        
        axes[1, 1].plot(recall, precision, label=f'AP={average_precision:.3f}')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle('Threshold Analysis', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def generate_evaluation_report(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        # Calculate all metrics
        business_metrics = self.calculate_business_metrics(y_true, y_pred)
        optimal_threshold_f1, best_f1 = self.find_optimal_threshold(
            y_true, y_pred_proba, 'f1'
        )
        optimal_threshold_cost, best_cost = self.find_optimal_threshold(
            y_true, y_pred_proba, 'business'
        )
        
        report = f"""
        {'='*60}
        MODEL EVALUATION REPORT: {model_name}
        {'='*60}
        
        BUSINESS METRICS (at 0.5 threshold):
        ------------------------------------
        Total Cost: ${business_metrics['total_cost']:,.2f}
        Cost per Prediction: ${business_metrics['cost_per_prediction']:.2f}
        Savings from Prevention: ${business_metrics['savings_from_prevention']:,.2f}
        Maintenance Efficiency: {business_metrics['maintenance_efficiency']:.2%}
        
        False Alarms: {business_metrics['false_alarms']}
        Missed Failures: {business_metrics['missed_failures']}
        Caught Failures: {business_metrics['caught_failures']}
        
        OPTIMAL THRESHOLDS:
        -------------------
        Best F1 Score: {best_f1:.4f} at threshold {optimal_threshold_f1:.2f}
        Minimum Cost: ${-best_cost:.2f} at threshold {optimal_threshold_cost:.2f}
        
        RECOMMENDATIONS:
        ----------------
        1. For balanced performance: Use threshold {optimal_threshold_f1:.2f}
        2. For cost optimization: Use threshold {optimal_threshold_cost:.2f}
        3. Current threshold (0.5) may not be optimal
        
        {'='*60}
        """
        
        return report


if __name__ == "__main__":
    # Example usage
    print("Model Evaluator module loaded successfully")