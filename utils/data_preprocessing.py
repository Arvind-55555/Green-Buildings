"""
Data Preprocessing Module for Green Building Energy Efficiency Prediction
Handles normalization, missing data, and feature engineering for multiple data types
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Comprehensive data preprocessing for green building energy efficiency prediction"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_names = []
        
    def handle_missing_data(self, df, strategy='knn', k=5):
        """
        Handle missing data using various imputation strategies
        
        Args:
            df: DataFrame with missing values
            strategy: 'mean', 'median', 'mode', 'knn', 'forward_fill'
            k: Number of neighbors for KNN imputation
        """
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if strategy == 'knn':
            # KNN imputation for numeric columns
            if numeric_cols:
                knn_imputer = KNNImputer(n_neighbors=k)
                df_processed[numeric_cols] = knn_imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['knn_numeric'] = knn_imputer
                
            # Mode imputation for categorical
            if categorical_cols:
                mode_imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = mode_imputer.fit_transform(df_processed[categorical_cols])
                self.imputers['mode_categorical'] = mode_imputer
                
        elif strategy == 'mean':
            if numeric_cols:
                mean_imputer = SimpleImputer(strategy='mean')
                df_processed[numeric_cols] = mean_imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['mean'] = mean_imputer
                
        elif strategy == 'forward_fill':
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
            
        else:
            # Default: fill with median for numeric, mode for categorical
            if numeric_cols:
                median_imputer = SimpleImputer(strategy='median')
                df_processed[numeric_cols] = median_imputer.fit_transform(df_processed[numeric_cols])
                self.imputers['median'] = median_imputer
                
            if categorical_cols:
                mode_imputer = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = mode_imputer.fit_transform(df_processed[categorical_cols])
                self.imputers['mode'] = mode_imputer
        
        return df_processed
    
    def normalize_features(self, df, method='standard', feature_groups=None):
        """
        Normalize features using different scaling methods
        
        Args:
            df: DataFrame to normalize
            method: 'standard', 'minmax', 'robust'
            feature_groups: Dictionary mapping group names to column lists
        """
        df_normalized = df.copy()
        numeric_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        if numeric_cols:
            df_normalized[numeric_cols] = scaler.fit_transform(df_normalized[numeric_cols])
            self.scalers[method] = scaler
        
        return df_normalized
    
    def encode_categorical(self, df, encoding='label'):
        """
        Encode categorical variables
        
        Args:
            df: DataFrame with categorical columns
            encoding: 'label', 'onehot', 'target'
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if encoding == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                
        elif encoding == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
            
        return df_encoded
    
    def engineer_features(self, df):
        """
        Create engineered features for green building energy efficiency
        
        Args:
            df: Input DataFrame
        """
        df_engineered = df.copy()
        
        # IoT Sensor Features
        if all(col in df.columns for col in ['temperature', 'humidity']):
            df_engineered['comfort_index'] = df['temperature'] / (df['humidity'] + 1)
            df_engineered['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        if 'air_quality' in df.columns:
            df_engineered['air_quality_squared'] = df['air_quality'] ** 2
        
        # Building Design Features
        if 'orientation' in df.columns and 'solar_radiation' in df.columns:
            # Orientation impact on solar gain
            df_engineered['solar_gain_potential'] = df['solar_radiation'] * df['orientation']
        
        if all(col in df.columns for col in ['insulation', 'roof_type']):
            df_engineered['thermal_resistance'] = df['insulation'] * (df['roof_type'] + 1)
        
        # Climate Features
        if all(col in df.columns for col in ['wind_speed', 'temperature']):
            df_engineered['wind_chill_factor'] = df['wind_speed'] * (df['temperature'] - 10)
            df_engineered['cooling_potential'] = df['wind_speed'] / (df['temperature'] + 1)
        
        # Consumer Perception Features
        if all(col in df.columns for col in ['green_perception', 'environmental_awareness']):
            df_engineered['sustainability_score'] = (
                df['green_perception'] * df['environmental_awareness']
            )
        
        if 'perceived_risk' in df.columns:
            df_engineered['risk_inverse'] = 1 / (df['perceived_risk'] + 0.1)
        
        # Operational Features
        if 'metro_logistics' in df.columns and 'policy_compliance' in df.columns:
            df_engineered['operational_efficiency'] = (
                df['metro_logistics'] * df['policy_compliance']
            )
        
        # Time-based features (if timestamp exists)
        if 'timestamp' in df.columns:
            df_engineered['timestamp'] = pd.to_datetime(df_engineered['timestamp'])
            df_engineered['hour'] = df_engineered['timestamp'].dt.hour
            df_engineered['day_of_week'] = df_engineered['timestamp'].dt.dayofweek
            df_engineered['month'] = df_engineered['timestamp'].dt.month
            df_engineered['is_weekend'] = (df_engineered['day_of_week'] >= 5).astype(int)
            # Drop original timestamp column after extracting features
            df_engineered = df_engineered.drop(columns=['timestamp'])
        
        # Equipment status interactions
        if 'equipment_status' in df.columns:
            df_engineered['equipment_active'] = (df['equipment_status'] > 0).astype(int)
        
        return df_engineered
    
    def select_features(self, X, y, k=50, score_func=f_regression):
        """
        Select top k features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            score_func: Scoring function
        """
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Drop any datetime columns that might still exist
        datetime_cols = X.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            X = X.drop(columns=datetime_cols)
            print(f"Warning: Dropped {len(datetime_cols)} datetime column(s) before feature selection")
        
        # Ensure all columns are numeric (categoricals should already be encoded)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < X.shape[1]:
            non_numeric = set(X.columns) - set(numeric_cols)
            X = X[numeric_cols]
            if len(non_numeric) > 0:
                print(f"Warning: Dropped {len(non_numeric)} non-numeric column(s) before feature selection: {list(non_numeric)}")
        
        # Select features
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors['kbest'] = selector
        
        # Get selected feature names and return as DataFrame
        if hasattr(X, 'columns'):
            selected_mask = selector.get_support()
            self.feature_names = X.columns[selected_mask].tolist()
            # Return as DataFrame with proper column names
            return pd.DataFrame(X_selected, columns=self.feature_names, index=X.index if hasattr(X, 'index') else None)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_selected.shape[1])]
            # Return as DataFrame
            return pd.DataFrame(X_selected, columns=self.feature_names)
    
    def preprocess_pipeline(self, df, target_col=None, is_training=True):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            is_training: Whether this is training data
        """
        # Separate target if provided
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            X = df.drop(columns=[target_col])
        else:
            X = df.copy()
            y = None
        
        # Handle missing data
        X = self.handle_missing_data(X, strategy=self.config.get('imputation_strategy', 'knn'))
        
        # Encode categorical variables
        X = self.encode_categorical(X, encoding=self.config.get('encoding', 'label'))
        
        # Feature engineering
        X = self.engineer_features(X)
        
        # Normalize features
        X = self.normalize_features(X, method=self.config.get('normalization', 'standard'))
        
        # Ensure X is a DataFrame and drop any remaining datetime columns
        if isinstance(X, pd.DataFrame):
            datetime_cols = X.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                X = X.drop(columns=datetime_cols)
        
        # Feature selection (only for training)
        if is_training and y is not None and self.config.get('feature_selection', False):
            k = self.config.get('n_features', 50)
            X = self.select_features(X, y, k=k)
        
        if y is not None:
            return X, y
        return X
    
    def transform(self, df, target_col=None):
        """Transform new data using fitted preprocessors"""
        return self.preprocess_pipeline(df, target_col=target_col, is_training=False)

