"""
Baseline model training with MLflow tracking
"""
import os
import sys
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_processor import DataProcessor

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and track models with MLflow"""
    
    def __init__(self, config_path: str = "src/config/config.yaml"):
        """Initialize trainer with config"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.models = {}
        self.metrics = {}
        
    def evaluate_model(self, model, X_test, y_test, model_name: str):
        """Calculate comprehensive metrics"""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store for later use
        self.metrics[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics, cm
    
    def plot_confusion_matrix(self, cm, model_name: str):
        """Plot confusion matrix"""
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add text with rates
        tn, fp, fn, tp = cm.ravel()
        textstr = f'True Positive Rate (Recall): {tp/(tp+fn):.2%}\n'
        textstr += f'False Positive Rate: {fp/(fp+tn):.2%}\n'
        textstr += f'Precision: {tp/(tp+fp):.2%}'
        
        plt.text(0.02, 0.95, textstr, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        return fig
    
    def plot_feature_importance(self, model, feature_names, model_name: str):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
            plt.title(f'Top 15 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            
            return fig
        return None
    
    def plot_roc_curve(self, y_test, y_pred_proba, model_name: str):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        return fig
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression baseline"""
        
        with mlflow.start_run(run_name="logistic_regression_baseline"):
            # Log data info
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("train_failure_rate", float(y_train.mean()))
            
            # Train model
            model = LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=1000,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics, cm = self.evaluate_model(model, X_test, y_test, "LogisticRegression")
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log confusion matrix
            fig = self.plot_confusion_matrix(cm, "Logistic Regression")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()
            
            # Log ROC curve
            fig = self.plot_roc_curve(
                y_test, 
                self.metrics["LogisticRegression"]["y_pred_proba"],
                "Logistic Regression"
            )
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close()
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="predictive_maintenance_logistic"
            )
            
            self.models['logistic_regression'] = model
            
            print(f"Logistic Regression - ROC AUC: {metrics['roc_auc']:.4f}")
            
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, feature_names):
        """Train Random Forest model"""
        
        with mlflow.start_run(run_name="random_forest_baseline"):
            # Log parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config['model']['random_state'],
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics, cm = self.evaluate_model(model, X_test, y_test, "RandomForest")
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log confusion matrix
            fig = self.plot_confusion_matrix(cm, "Random Forest")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()
            
            # Log feature importance
            fig = self.plot_feature_importance(model, feature_names, "Random Forest")
            if fig:
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close()
            
            # Log ROC curve
            fig = self.plot_roc_curve(
                y_test,
                self.metrics["RandomForest"]["y_pred_proba"],
                "Random Forest"
            )
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close()
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="predictive_maintenance_rf"
            )
            
            self.models['random_forest'] = model
            
            print(f"Random Forest - ROC AUC: {metrics['roc_auc']:.4f}")
            
        return model
    
    def train_xgboost(self, X_train, y_train, X_test, y_test, feature_names):
        """Train XGBoost model"""
        
        # Clean feature names for XGBoost (remove special characters)
        import re
        clean_feature_names = [re.sub(r'[\[\]<>]', '', name).strip() for name in feature_names]
        
        # Create DataFrames with clean column names
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)
        
        # Rename columns to clean names
        X_train.columns = clean_feature_names
        X_test.columns = clean_feature_names
        
        with mlflow.start_run(run_name="xgboost_baseline"):
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
            
            # Log parameters
            params = {
                "model_type": "XGBoost",
                "n_estimators": self.config['model']['baseline']['n_estimators'],
                "max_depth": self.config['model']['baseline']['max_depth'],
                "learning_rate": self.config['model']['baseline']['learning_rate'],
                "scale_pos_weight": scale_pos_weight,
                "n_train_samples": len(X_train),
                "n_features": X_train.shape[1]
            }
            
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Train model
            model = XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                scale_pos_weight=scale_pos_weight,
                random_state=self.config['model']['random_state'],
                eval_metric='auc',
                early_stopping_rounds=10  # Moved here for newer XGBoost versions
            )
            
            # Train with evaluation set
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Evaluate
            metrics, cm = self.evaluate_model(model, X_test, y_test, "XGBoost")
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log confusion matrix
            fig = self.plot_confusion_matrix(cm, "XGBoost")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()
            
            # Log feature importance (use clean names for display)
            fig = self.plot_feature_importance(model, clean_feature_names, "XGBoost")
            if fig:
                mlflow.log_figure(fig, "feature_importance.png")
                plt.close()
            
            # Log ROC curve
            fig = self.plot_roc_curve(
                y_test,
                self.metrics["XGBoost"]["y_pred_proba"],
                "XGBoost"
            )
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close()
            
            # Log model
            mlflow.xgboost.log_model(
                model,
                "model",
                registered_model_name="predictive_maintenance_xgboost"
            )
            
            self.models['xgboost'] = model
            
            print(f"XGBoost - ROC AUC: {metrics['roc_auc']:.4f}")
            
        return model
    
    def compare_models(self):
        """Create comparison visualization of all models"""
        
        if not self.metrics:
            print("No models trained yet")
            return
        
        # Prepare comparison data
        comparison_data = []
        for model_name, model_metrics in self.metrics.items():
            for metric_name, metric_value in model_metrics['metrics'].items():
                comparison_data.append({
                    'Model': model_name,
                    'Metric': metric_name,
                    'Value': metric_value
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for idx, metric in enumerate(metrics_to_plot):
            metric_df = comparison_df[comparison_df['Metric'] == metric]
            axes[idx].bar(metric_df['Model'], metric_df['Value'])
            axes[idx].set_title(f'{metric.upper().replace("_", " ")}')
            axes[idx].set_ylim([0, 1])
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(metric_df['Value']):
                axes[idx].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        axes[-1].axis('off')
        
        plt.suptitle('Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        
        # Save comparison plot
        plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Print comparison table
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        pivot_df = comparison_df.pivot(index='Model', columns='Metric', values='Value')
        print(pivot_df.round(4))
        
        # Find best model
        best_model = pivot_df['roc_auc'].idxmax()
        best_score = pivot_df['roc_auc'].max()
        
        print("\n" + "="*60)
        print(f"BEST MODEL: {best_model} (ROC-AUC: {best_score:.4f})")
        print("="*60)
        
        return comparison_df
    
    def save_best_model(self, output_dir: str = "models/"):
        """Save the best performing model"""
        
        if not self.metrics:
            print("No models trained")
            return
        
        # Find best model based on ROC-AUC
        best_model_name = None
        best_score = 0
        
        for model_name, model_metrics in self.metrics.items():
            score = model_metrics['metrics']['roc_auc']
            if score > best_score:
                best_score = score
                best_model_name = model_name.lower().replace(" ", "_")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        best_model = self.models.get(best_model_name)
        if best_model:
            model_path = os.path.join(output_dir, f"best_model_{best_model_name}.pkl")
            joblib.dump(best_model, model_path)
            
            # Save metadata
            metadata = {
                'model_name': best_model_name,
                'roc_auc_score': best_score,
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics[best_model_name.replace("_", " ").title()]['metrics']
            }
            
            metadata_path = os.path.join(output_dir, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Best model saved to {model_path}")
            print(f"Metadata saved to {metadata_path}")
        
        return best_model_name


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("PREDICTIVE MAINTENANCE - MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Initialize processor and trainer
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Process data
    print("\n1. Processing data...")
    data = processor.process_pipeline(
        "data/raw/predictive_maintenance.csv",
        handle_imbalance=True
    )
    
    print("\n2. Training models...")
    
    # Train Logistic Regression
    print("\n   Training Logistic Regression...")
    trainer.train_logistic_regression(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )
    
    # Train Random Forest
    print("\n   Training Random Forest...")
    trainer.train_random_forest(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['feature_names']
    )
    
    # Train XGBoost
    print("\n   Training XGBoost...")
    trainer.train_xgboost(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data['feature_names']
    )
    
    # Compare models
    print("\n3. Comparing models...")
    trainer.compare_models()
    
    # Save best model
    print("\n4. Saving best model...")
    trainer.save_best_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"MLflow UI: Run 'mlflow ui' to view experiments at http://localhost:5000")
    print("="*60)


if __name__ == "__main__":
    main()