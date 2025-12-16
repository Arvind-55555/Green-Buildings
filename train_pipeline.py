"""
Main Training Pipeline for Green Building Energy Efficiency Prediction
Supports XGBoost, LSTM, and BERT models with comprehensive evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
utils_path = os.path.join(os.path.dirname(__file__), 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

try:
    from utils.data_preprocessing import DataPreprocessor
    from utils.models import XGBoostModel, LSTMModel, LSTMTrainer, create_sequences
    from utils.evaluation import ModelEvaluator, SHAPExplainer, LIMEExplainer
    from utils.data_generator import generate_synthetic_data, generate_temporal_data
except ImportError:
    # Fallback to direct imports if utils is in path
    from data_preprocessing import DataPreprocessor
    from models import XGBoostModel, LSTMModel, LSTMTrainer, create_sequences
    from evaluation import ModelEvaluator, SHAPExplainer, LIMEExplainer
    from data_generator import generate_synthetic_data, generate_temporal_data

# Import BERT components if available
try:
    from transformers import BertTokenizer
    try:
        from utils.models import BERTTextModel, BERTTrainer
    except ImportError:
        from models import BERTTextModel, BERTTrainer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: BERT dependencies not available. BERT model will be skipped.")


class EnergyEfficiencyPipeline:
    """Main pipeline for energy efficiency prediction"""
    
    def __init__(self, config_path='config.json'):
        """Initialize pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        self.evaluator = ModelEvaluator()
        self.models = {}
        self.results = {}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'model_type': 'xgboost',  # 'xgboost', 'lstm', 'bert', 'all'
            'preprocessing': {
                'imputation_strategy': 'knn',
                'normalization': 'standard',
                'encoding': 'label',
                'feature_selection': True,
                'n_features': 50
            },
            'xgboost': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lstm': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.2,
                'seq_length': 24,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            },
            'bert': {
                'model_name': 'bert-base-uncased',
                'epochs': 10,
                'batch_size': 16,
                'learning_rate': 2e-5
            },
            'evaluation': {
                'use_shap': True,
                'use_lime': True,
                'save_plots': True
            }
        }
    
    def load_data(self, data_path=None, generate_if_missing=True):
        """Load or generate data"""
        if data_path and os.path.exists(data_path):
            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
        elif generate_if_missing:
            print("Generating synthetic data...")
            df = generate_synthetic_data(n_samples=self.config.get('n_samples', 1000))
            # Save generated data
            df.to_csv('data/synthetic_data.csv', index=False)
            print("Synthetic data saved to data/synthetic_data.csv")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        return df
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60)
        
        xgb_model = XGBoostModel(self.config.get('xgboost', {}))
        xgb_model.fit(X_train, y_train, X_val, y_val)
        
        # Predictions
        y_train_pred = xgb_model.predict(X_train)
        y_val_pred = xgb_model.predict(X_val)
        
        # Evaluation
        train_metrics = self.evaluator.calculate_regression_metrics(y_train, y_train_pred, prefix='Train_')
        val_metrics = self.evaluator.calculate_regression_metrics(y_val, y_val_pred, prefix='Val_')
        
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save model
        import pickle
        with open('models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        
        # Plots
        if self.config.get('evaluation', {}).get('save_plots', True):
            self.evaluator.plot_predictions(y_val, y_val_pred, 
                                          title='XGBoost: Predictions vs Actual',
                                          save_path='results/xgboost_predictions.png')
            self.evaluator.plot_residuals(y_val, y_val_pred,
                                         save_path='results/xgboost_residuals.png')
        
        # Feature importance
        feature_importance = xgb_model.get_feature_importance()
        if feature_importance is not None:
            # Get feature names
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            elif hasattr(self.preprocessor, 'feature_names') and len(self.preprocessor.feature_names) > 0:
                feature_names = self.preprocessor.feature_names
            else:
                feature_names = [f'feature_{i}' for i in range(len(feature_importance))]
            
            self.evaluator.plot_feature_importance(
                feature_importance, feature_names,
                save_path='results/xgboost_feature_importance.png'
            )
        
        # SHAP explanations
        if self.config.get('evaluation', {}).get('use_shap', True):
            try:
                print("\nGenerating SHAP explanations...")
                # Ensure X_train and X_val are DataFrames
                if not isinstance(X_train, pd.DataFrame):
                    # Convert to DataFrame with feature names
                    if hasattr(self.preprocessor, 'feature_names') and len(self.preprocessor.feature_names) > 0:
                        feature_names = self.preprocessor.feature_names
                    else:
                        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                    X_val_df = pd.DataFrame(X_val, columns=feature_names)
                else:
                    X_train_df = X_train
                    X_val_df = X_val
                
                shap_explainer = SHAPExplainer(xgb_model.model, X_train_df, model_type='xgboost')
                # Sample validation data
                n_samples = min(100, len(X_val_df))
                X_val_sample = X_val_df.sample(n=n_samples, random_state=42) if len(X_val_df) > n_samples else X_val_df
                shap_explainer.explain(X_val_sample)
                shap_explainer.plot_summary(
                    feature_names=X_train_df.columns.tolist(),
                    save_path='results/xgboost_shap_summary.png'
                )
                shap_explainer.plot_feature_importance(
                    feature_names=X_train_df.columns.tolist(),
                    save_path='results/xgboost_shap_importance.png'
                )
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        # LIME explanations
        if self.config.get('evaluation', {}).get('use_lime', True):
            try:
                print("Generating LIME explanations...")
                # Ensure X_train and X_val are DataFrames
                if not isinstance(X_train, pd.DataFrame):
                    # Convert to DataFrame with feature names
                    if hasattr(self.preprocessor, 'feature_names') and len(self.preprocessor.feature_names) > 0:
                        feature_names = self.preprocessor.feature_names
                    else:
                        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
                    X_train_df = pd.DataFrame(X_train, columns=feature_names)
                    X_val_df = pd.DataFrame(X_val, columns=feature_names)
                else:
                    X_train_df = X_train
                    X_val_df = X_val
                
                lime_explainer = LIMEExplainer(
                    xgb_model.model, X_train_df, 
                    feature_names=X_train_df.columns.tolist(),
                    task_type='regression'
                )
                explanation = lime_explainer.explain_instance(X_val_df.iloc[0])
                lime_explainer.plot_explanation(
                    explanation, 
                    save_path='results/xgboost_lime_explanation.png'
                )
            except Exception as e:
                print(f"LIME explanation failed: {e}")
        
        self.models['xgboost'] = xgb_model
        return xgb_model, val_metrics
    
    def train_lstm(self, df, target_col='energy_efficiency'):
        """Train LSTM model on temporal data"""
        print("\n" + "="*60)
        print("Training LSTM Model")
        print("="*60)
        
        # Generate or use temporal data
        if 'timestamp' in df.columns:
            # Use existing temporal data
            temporal_df = df.sort_values('timestamp')
        else:
            # Generate temporal data
            temporal_df = generate_temporal_data(n_samples=2000)
        
        # Prepare sequences
        seq_length = self.config.get('lstm', {}).get('seq_length', 24)
        
        # Select features for LSTM
        feature_cols = ['temperature', 'humidity', 'solar_radiation', 'wind_speed']
        available_cols = [col for col in feature_cols if col in temporal_df.columns]
        
        if len(available_cols) < 2:
            print("Warning: Insufficient features for LSTM. Using all numeric columns.")
            available_cols = temporal_df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in available_cols:
                available_cols.remove(target_col)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(temporal_df[available_cols])
        target_scaled = temporal_df[target_col].values
        
        # Create sequences
        X_seq, y_seq = create_sequences(
            np.column_stack([features_scaled, target_scaled]),
            seq_length=seq_length
        )
        
        # Split sequences
        split_idx = int(0.8 * len(X_seq))
        X_train_seq = X_seq[:split_idx]
        y_train_seq = y_seq[:, -1]  # Last value is target
        X_val_seq = X_seq[split_idx:]
        y_val_seq = y_seq[split_idx:, -1]
        
        # Create model
        input_size = X_train_seq.shape[2]
        lstm_model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.get('lstm', {}).get('hidden_size', 128),
            num_layers=self.config.get('lstm', {}).get('num_layers', 2),
            dropout=self.config.get('lstm', {}).get('dropout', 0.2)
        )
        
        # Create data loaders
        try:
            from utils.models import TabularDataset
        except ImportError:
            from models import TabularDataset
        train_dataset = TabularDataset(X_train_seq.reshape(-1, seq_length, input_size), y_train_seq)
        val_dataset = TabularDataset(X_val_seq.reshape(-1, seq_length, input_size), y_val_seq)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.get('lstm', {}).get('batch_size', 32),
            shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.get('lstm', {}).get('batch_size', 32),
            shuffle=False
        )
        
        # Train
        trainer = LSTMTrainer(lstm_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        trainer.train(
            train_loader, val_loader,
            epochs=self.config.get('lstm', {}).get('epochs', 100),
            lr=self.config.get('lstm', {}).get('learning_rate', 0.001)
        )
        
        # Predictions
        y_train_pred = trainer.predict(train_loader)
        y_val_pred = trainer.predict(val_loader)
        
        # Evaluation
        train_metrics = self.evaluator.calculate_regression_metrics(
            y_train_seq, y_train_pred.flatten(), prefix='Train_'
        )
        val_metrics = self.evaluator.calculate_regression_metrics(
            y_val_seq, y_val_pred.flatten(), prefix='Val_'
        )
        
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plots
        if self.config.get('evaluation', {}).get('save_plots', True):
            self.evaluator.plot_predictions(
                y_val_seq, y_val_pred.flatten(),
                title='LSTM: Predictions vs Actual',
                save_path='results/lstm_predictions.png'
            )
        
        self.models['lstm'] = {'model': lstm_model, 'trainer': trainer}
        return lstm_model, val_metrics
    
    def train_bert(self, df):
        """Train BERT model on text data"""
        if not BERT_AVAILABLE:
            print("BERT dependencies not available. Skipping BERT training.")
            return None, {}
        
        print("\n" + "="*60)
        print("Training BERT Model")
        print("="*60)
        
        # Generate text data if not available
        try:
            from utils.data_generator import generate_text_data
        except ImportError:
            from data_generator import generate_text_data
        text_df = generate_text_data(n_samples=500)
        
        # Use text to predict energy efficiency (synthetic target)
        # In real scenario, this would be survey responses predicting efficiency
        text_df['energy_efficiency'] = (
            text_df['green_perception'] * 0.4 +
            text_df['environmental_awareness'] * 0.4 -
            text_df['perceived_risk'] * 0.2
        ) / 5.0
        
        # Split data
        train_texts, val_texts, train_targets, val_targets = train_test_split(
            text_df['text'], text_df['energy_efficiency'],
            test_size=0.2, random_state=42
        )
        
        # Tokenize
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize_texts(texts, max_length=512):
            return tokenizer(
                texts.tolist(),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
        
        train_encodings = tokenize_texts(train_texts)
        val_encodings = tokenize_texts(val_texts)
        
        # Create datasets
        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': torch.tensor(self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx], dtype=torch.float)
                }
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = TextDataset(train_encodings, train_targets)
        val_dataset = TextDataset(val_encodings, val_targets)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config.get('bert', {}).get('batch_size', 16), shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config.get('bert', {}).get('batch_size', 16), shuffle=False
        )
        
        # Create and train model
        bert_model = BERTTextModel(
            model_name=self.config.get('bert', {}).get('model_name', 'bert-base-uncased')
        )
        trainer = BERTTrainer(bert_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        trainer.tokenizer = tokenizer
        
        trainer.train(
            train_loader, val_loader,
            epochs=self.config.get('bert', {}).get('epochs', 10),
            lr=self.config.get('bert', {}).get('learning_rate', 2e-5)
        )
        
        # Predictions
        y_train_pred = trainer.predict(train_loader)
        y_val_pred = trainer.predict(val_loader)
        
        # Evaluation
        train_metrics = self.evaluator.calculate_regression_metrics(
            train_targets.values, y_train_pred.flatten(), prefix='Train_'
        )
        val_metrics = self.evaluator.calculate_regression_metrics(
            val_targets.values, y_val_pred.flatten(), prefix='Val_'
        )
        
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nValidation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        self.models['bert'] = {'model': bert_model, 'trainer': trainer}
        return bert_model, val_metrics
    
    def run(self, data_path=None):
        """Run complete pipeline"""
        print("="*60)
        print("Green Building Energy Efficiency Prediction Pipeline")
        print("="*60)
        
        # Load data
        df = self.load_data(data_path)
        print(f"\nData shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        model_type = self.config.get('model_type', 'xgboost')
        
        # Train XGBoost (for tabular data)
        if model_type in ['xgboost', 'all']:
            # Preprocess data
            X, y = self.preprocessor.preprocess_pipeline(
                df, target_col='energy_efficiency', is_training=True
            )
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train
            self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Train LSTM (for temporal data)
        if model_type in ['lstm', 'all']:
            try:
                self.train_lstm(df)
            except Exception as e:
                print(f"LSTM training failed: {e}")
        
        # Train BERT (for text data)
        if model_type in ['bert', 'all']:
            try:
                self.train_bert(df)
            except Exception as e:
                print(f"BERT training failed: {e}")
        
        # Generate final report
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print("\nResults saved to:")
        print("  - Models: models/")
        print("  - Plots: results/")
        print("  - Data: data/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train energy efficiency prediction models')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--data', type=str, default=None, help='Path to data file')
    parser.add_argument('--model', type=str, default='xgboost', 
                       choices=['xgboost', 'lstm', 'bert', 'all'],
                       help='Model type to train')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = EnergyEfficiencyPipeline(config_path=args.config)
    
    # Update model type if specified
    if args.model:
        pipeline.config['model_type'] = args.model
    
    # Run pipeline
    pipeline.run(data_path=args.data)

