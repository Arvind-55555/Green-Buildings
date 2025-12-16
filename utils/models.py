"""
Model Definitions for Green Building Energy Efficiency Prediction
Includes XGBoost, LSTM, and BERT-based models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Optional BERT imports
try:
    from transformers import BertModel, BertTokenizer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    BertModel = None
    BertTokenizer = None


class TabularDataset(Dataset):
    """Dataset class for tabular data"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values if isinstance(X, pd.DataFrame) else X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.Series) else y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class XGBoostModel:
    """XGBoost model for tabular data"""
    
    def __init__(self, params=None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        # Create model with params
        model_params = self.params.copy()
        
        # Handle early stopping for different XGBoost versions
        if X_val is not None and y_val is not None:
            # Check XGBoost version to determine API
            try:
                xgb_version = xgb.__version__
                major_version = int(xgb_version.split('.')[0])
            except:
                major_version = 1  # Default to old API if version check fails
            
            if major_version >= 2:
                # XGBoost 2.0+ API: early_stopping_rounds in constructor, eval_set in fit()
                if 'early_stopping_rounds' in model_params:
                    model_params.pop('early_stopping_rounds')
                model_params['early_stopping_rounds'] = 50
                self.model = xgb.XGBRegressor(**model_params)
                # eval_set is passed in fit() for XGBoost 2.0+
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                # XGBoost < 2.0 API: early_stopping_rounds in fit()
                self.model = xgb.XGBRegressor(**model_params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
        else:
            self.model = xgb.XGBRegressor(**model_params)
            self.model.fit(X_train, y_train)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if self.model is not None:
            return self.model.feature_importances_
        return None


class LSTMModel(nn.Module):
    """LSTM model for temporal/sequential data"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class LSTMTrainer:
    """Trainer for LSTM model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
    
    def train(self, train_loader, val_loader=None, epochs=100, lr=0.001, patience=10):
        """Train LSTM model"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5, verbose=False
            )
        except TypeError:
            # Some PyTorch versions don't support verbose parameter
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'models/best_lstm.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        # Load best model
        if val_loader is not None:
            self.model.load_state_dict(torch.load('models/best_lstm.pth'))
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def predict(self, test_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for X_batch in test_loader:
                if isinstance(X_batch, tuple):
                    X_batch = X_batch[0]
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)


class BERTTextModel(nn.Module):
    """BERT-based model for text data (survey responses, descriptions)"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=1, dropout=0.3):
        super(BERTTextModel, self).__init__()
        if not BERT_AVAILABLE:
            raise ImportError("transformers library is required for BERT models. Install with: pip install transformers")
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.fc(output)
        return output


class BERTTrainer:
    """Trainer for BERT model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.tokenizer = None
    
    def train(self, train_loader, val_loader=None, epochs=10, lr=2e-5, patience=3):
        """Train BERT model"""
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2, verbose=False
            )
        except TypeError:
            # Some PyTorch versions don't support verbose parameter
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2
            )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'models/best_bert.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        if val_loader is not None:
            self.model.load_state_dict(torch.load('models/best_bert.pth'))
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        return val_loss / len(val_loader)
    
    def predict(self, test_loader):
        """Make predictions"""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions, axis=0)


def create_sequences(data, seq_length=24):
    """Create sequences for LSTM from time series data"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

