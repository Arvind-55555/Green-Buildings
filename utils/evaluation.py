"""
Evaluation and Interpretability Module
Includes metrics (RMSE, F1-score) and interpretability methods (SHAP, LIME)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Optional imports for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import lime
    import lime.lime_tabular
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime = None


class ModelEvaluator:
    """Comprehensive model evaluation for energy efficiency prediction"""
    
    def __init__(self):
        self.metrics = {}
        self.predictions = {}
        self.true_values = {}
    
    def calculate_regression_metrics(self, y_true, y_pred, prefix=''):
        """Calculate regression metrics (RMSE, MAE, R2)"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics = {
            f'{prefix}RMSE': rmse,
            f'{prefix}MAE': mae,
            f'{prefix}R2': r2,
            f'{prefix}MAPE': mape
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def calculate_classification_metrics(self, y_true, y_pred, prefix=''):
        """Calculate classification metrics (F1, Precision, Recall)"""
        # Convert to binary if needed
        if len(np.unique(y_true)) > 2:
            # Multi-class: use macro average
            f1 = f1_score(y_true, y_pred, average='macro')
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
        else:
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            f'{prefix}F1_Score': f1,
            f'{prefix}Precision': precision,
            f'{prefix}Recall': recall,
            f'{prefix}Accuracy': accuracy
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def plot_predictions(self, y_true, y_pred, title='Predictions vs Actual', save_path=None):
        """Plot predictions against actual values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Energy Efficiency')
        plt.ylabel('Predicted Energy Efficiency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Add metrics to plot
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.4f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, y_true, y_pred, save_path=None):
        """Plot residuals"""
        residuals = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance, feature_names, top_n=20, save_path=None):
        """Plot feature importance"""
        # Sort by importance
        indices = np.argsort(feature_importance)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, y_true, y_pred, task_type='regression', save_path=None):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        if task_type == 'regression':
            metrics = self.calculate_regression_metrics(y_true, y_pred)
            report.append("Regression Metrics:")
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.4f}")
        else:
            metrics = self.calculate_classification_metrics(y_true, y_pred)
            report.append("Classification Metrics:")
            for metric, value in metrics.items():
                report.append(f"  {metric}: {value:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


class SHAPExplainer:
    """SHAP-based interpretability for models"""
    
    def __init__(self, model, X_train, model_type='xgboost'):
        self.model = model
        self.X_train = X_train
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        self.X_explain_used = None  # Store the actual data used for explanation
        
    def explain(self, X_explain, max_evals=100):
        """Generate SHAP explanations"""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required. Install with: pip install shap")
        
        # Store the actual data used for explanation
        if self.model_type == 'xgboost':
            self.explainer = shap.TreeExplainer(self.model)
            # For TreeExplainer, use X_explain directly
            self.X_explain_used = X_explain
            self.shap_values = self.explainer.shap_values(X_explain)
        elif self.model_type == 'neural_network':
            # For neural networks, use KernelExplainer
            background_data = self.X_train.sample(min(100, len(self.X_train))) if hasattr(self.X_train, 'sample') else self.X_train[:min(100, len(self.X_train))]
            self.explainer = shap.KernelExplainer(
                self.model.predict, 
                background_data
            )
            # Sample X_explain if needed
            if hasattr(X_explain, 'sample'):
                X_explain_sample = X_explain.sample(min(max_evals, len(X_explain)))
            else:
                n_samples = min(max_evals, len(X_explain))
                indices = np.random.choice(len(X_explain), n_samples, replace=False)
                X_explain_sample = X_explain.iloc[indices] if hasattr(X_explain, 'iloc') else X_explain[indices]
            self.X_explain_used = X_explain_sample
            self.shap_values = self.explainer.shap_values(X_explain_sample)
        else:
            # Default: KernelExplainer
            background_data = self.X_train.sample(min(100, len(self.X_train))) if hasattr(self.X_train, 'sample') else self.X_train[:min(100, len(self.X_train))]
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                background_data
            )
            # Sample X_explain if needed
            if hasattr(X_explain, 'sample'):
                X_explain_sample = X_explain.sample(min(max_evals, len(X_explain)))
            else:
                n_samples = min(max_evals, len(X_explain))
                indices = np.random.choice(len(X_explain), n_samples, replace=False)
                X_explain_sample = X_explain.iloc[indices] if hasattr(X_explain, 'iloc') else X_explain[indices]
            self.X_explain_used = X_explain_sample
            self.shap_values = self.explainer.shap_values(X_explain_sample)
        
        return self.shap_values
    
    def plot_summary(self, feature_names=None, max_display=20, save_path=None):
        """Plot SHAP summary"""
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        # Use the actual data that was explained, not the full training set
        if self.X_explain_used is not None:
            X_to_plot = self.X_explain_used
        else:
            # Fallback to X_train if X_explain_used not set (for backward compatibility)
            X_to_plot = self.X_train
        
        # Convert to numpy array if needed
        if hasattr(X_to_plot, 'values'):
            X_to_plot = X_to_plot.values
        elif hasattr(X_to_plot, 'to_numpy'):
            X_to_plot = X_to_plot.to_numpy()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            X_to_plot,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_waterfall(self, instance_idx=0, feature_names=None, save_path=None):
        """Plot SHAP waterfall for a single instance"""
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        # Use the actual data that was explained
        if self.X_explain_used is not None:
            X_to_plot = self.X_explain_used
        else:
            X_to_plot = self.X_train
        
        # Get the instance data
        if hasattr(X_to_plot, 'iloc'):
            instance_data = X_to_plot.iloc[instance_idx]
        elif hasattr(X_to_plot, '__getitem__'):
            instance_data = X_to_plot[instance_idx]
        else:
            instance_data = X_to_plot
        
        # Get SHAP values for this instance
        if isinstance(self.shap_values, list):
            shap_vals = self.shap_values[instance_idx]
        else:
            shap_vals = self.shap_values[instance_idx] if len(self.shap_values.shape) > 1 else self.shap_values
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=self.explainer.expected_value,
                data=instance_data,
                feature_names=feature_names
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_names=None, save_path=None):
        """Plot SHAP feature importance"""
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            shap_importance = np.abs(self.shap_values).mean(0)
        else:
            shap_importance = np.abs(self.shap_values).mean(0)
        
        # Sort by importance
        indices = np.argsort(shap_importance)[::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), shap_importance[indices])
        if feature_names:
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('SHAP Feature Importance')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class LIMEExplainer:
    """LIME-based interpretability for models"""
    
    def __init__(self, model, X_train, feature_names=None, task_type='regression'):
        if not LIME_AVAILABLE:
            raise ImportError("LIME library is required. Install with: pip install lime")
        
        self.model = model
        self.X_train = X_train.values if hasattr(X_train, 'values') else X_train
        self.feature_names = feature_names
        self.task_type = task_type
        
        if task_type == 'regression':
            mode = 'regression'
        else:
            mode = 'classification'
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=feature_names,
            mode=mode,
            discretize_continuous=True
        )
    
    def explain_instance(self, instance, num_features=10):
        """Explain a single instance"""
        instance_array = instance.values if hasattr(instance, 'values') else instance
        
        explanation = self.explainer.explain_instance(
            instance_array,
            self.model.predict,
            num_features=num_features
        )
        
        return explanation
    
    def plot_explanation(self, explanation, save_path=None):
        """Plot LIME explanation"""
        fig = explanation.as_pyplot_figure()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def get_explanation_text(self, explanation):
        """Get explanation as text"""
        return explanation.as_list()


class TextLIMEExplainer:
    """LIME explainer for text data (BERT models)"""
    
    def __init__(self, model, tokenizer, class_names=None):
        if not LIME_AVAILABLE:
            raise ImportError("LIME library is required. Install with: pip install lime")
        
        self.model = model
        self.tokenizer = tokenizer
        self.explainer = LimeTextExplainer(class_names=class_names)
    
    def explain_instance(self, text, num_features=10):
        """Explain a text instance"""
        def predict_proba(texts):
            # Tokenize and predict
            inputs = self.tokenizer(
                texts, return_tensors='pt', padding=True, truncation=True, max_length=512
            )
            with torch.no_grad():
                outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
            # Convert to probabilities if needed
            return outputs.cpu().numpy()
        
        explanation = self.explainer.explain_instance(
            text, predict_proba, num_features=num_features
        )
        
        return explanation

