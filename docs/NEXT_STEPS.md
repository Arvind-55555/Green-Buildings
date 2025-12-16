# Next Steps Guide

## ✅ Current Status

Your pipeline is working! The model achieved:
- **Validation R²: 0.9153** (91.5% variance explained)
- **Validation RMSE: 0.0500**
- Models and visualizations have been generated

## 📋 Immediate Next Steps

### 1. **Review Generated Results**

Check what was created:

```bash
# View generated files
ls -lh results/
ls -lh models/
ls -lh data/
```

**Expected outputs:**
- `results/xgboost_predictions.png` - Prediction vs actual plot
- `results/xgboost_residuals.png` - Residual analysis
- `results/xgboost_feature_importance.png` - Feature importance
- `results/xgboost_shap_summary.png` - SHAP summary plot
- `results/xgboost_shap_importance.png` - SHAP feature importance
- `results/xgboost_lime_explanation.png` - LIME explanation
- `models/xgboost_model.pkl` - Trained model

### 2. **Prepare Your Real Data**

If you have real data, format it as CSV with these columns:

**Required:**
- `energy_efficiency` (0-1 scale) or `energy_consumption` (kWh) - **Target variable**

**Optional (will be generated if missing):**
- **IoT Sensors**: `temperature`, `humidity`, `air_quality`, `equipment_status`
- **Building Design**: `orientation`, `material`, `roof_type`, `insulation`
- **Survey Data**: `green_perception`, `environmental_awareness`, `perceived_risk`
- **Climate**: `solar_radiation`, `wind_speed`, `temperature_profile`
- **Operational**: `metro_logistics`, `policy_compliance`
- **Temporal**: `timestamp` (for LSTM model)

### 3. **Train with Your Data**

```bash
# Train XGBoost with your data
python train_pipeline.py --data path/to/your/data.csv --model xgboost

# Train all models
python train_pipeline.py --data path/to/your/data.csv --model all
```

### 4. **Customize Configuration**

Edit `config.json` to optimize for your use case:

```json
{
  "model_type": "xgboost",  // or "lstm", "bert", "all"
  "preprocessing": {
    "imputation_strategy": "knn",  // "mean", "median", "forward_fill"
    "normalization": "standard",   // "minmax", "robust"
    "feature_selection": true,
    "n_features": 50
  },
  "xgboost": {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.01
  }
}
```

## 🔧 Advanced Usage

### 5. **Hyperparameter Tuning**

Create a tuning script:

```python
from train_pipeline import EnergyEfficiencyPipeline
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Load your data
pipeline = EnergyEfficiencyPipeline()
df = pipeline.load_data('data/your_data.csv')
X, y = pipeline.preprocessor.preprocess_pipeline(df, target_col='energy_efficiency')

# Define parameter grid
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 2000]
}

# Grid search
model = xgb.XGBRegressor()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### 6. **Model Comparison**

Compare different models:

```bash
# Train all models and compare
python train_pipeline.py --model all

# Check results in results/ directory
```

### 7. **Feature Engineering**

Add custom features in `utils/data_preprocessing.py`:

```python
def engineer_features(self, df):
    # Add your custom features here
    df_engineered = df.copy()
    
    # Example: Building age effect
    if 'construction_year' in df.columns:
        df_engineered['building_age'] = 2024 - df['construction_year']
        df_engineered['age_efficiency_interaction'] = (
            df_engineered['building_age'] * df_engineered['energy_efficiency']
        )
    
    return df_engineered
```

### 8. **Deploy Model for Predictions**

Create a prediction script:

```python
import pickle
import pandas as pd
from utils.data_preprocessing import DataPreprocessor

# Load trained model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load preprocessor
preprocessor = DataPreprocessor()
# Note: You'll need to save/load preprocessor state in production

# Load new data
new_data = pd.read_csv('data/new_buildings.csv')

# Preprocess
X_new = preprocessor.transform(new_data)

# Predict
predictions = model.predict(X_new)
print(f"Energy Efficiency Predictions: {predictions}")
```

## 📊 Interpreting Results

### Model Performance Metrics

- **RMSE (Root Mean Squared Error)**: Lower is better. Measures prediction error.
- **MAE (Mean Absolute Error)**: Average prediction error.
- **R² (R-squared)**: 0-1 scale. Higher is better. Proportion of variance explained.
  - 0.9+ = Excellent
  - 0.7-0.9 = Good
  - 0.5-0.7 = Moderate
  - <0.5 = Poor

### SHAP Values

- **Positive SHAP value**: Feature increases energy efficiency
- **Negative SHAP value**: Feature decreases energy efficiency
- **Magnitude**: How much the feature affects the prediction

### Feature Importance

Shows which features most influence energy efficiency predictions.

## 🚀 Production Deployment

### 9. **Create API Endpoint**

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    # Preprocess and predict
    prediction = model.predict(df)
    return jsonify({'energy_efficiency': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
```

### 10. **Model Monitoring**

Track model performance over time:

```python
# Monitor predictions vs actuals
import pandas as pd

def monitor_model(actuals, predictions, threshold=0.1):
    errors = abs(actuals - predictions)
    mae = errors.mean()
    rmse = (errors**2).mean()**0.5
    
    # Alert if performance degrades
    if mae > threshold:
        print(f"WARNING: Model performance degraded. MAE: {mae:.4f}")
    
    return {'mae': mae, 'rmse': rmse}
```

## 📈 Potential Improvements

### 11. **Data Collection**

- **More data**: Collect more building samples
- **Feature expansion**: Add more IoT sensors, building characteristics
- **Temporal data**: Collect time-series data for LSTM model
- **Text data**: Collect survey responses for BERT model

### 12. **Model Enhancements**

- **Ensemble methods**: Combine XGBoost, LSTM, and BERT predictions
- **Cross-validation**: Implement k-fold cross-validation
- **Feature selection**: Try different feature selection methods
- **Hyperparameter optimization**: Use Optuna or Hyperopt

### 13. **Interpretability**

- **SHAP interaction values**: Understand feature interactions
- **Partial dependence plots**: Visualize feature effects
- **Model cards**: Document model behavior and limitations

## 🎯 Recommended Workflow

1. ✅ **Test with synthetic data** (Done!)
2. **Prepare your real dataset**
3. **Train and evaluate models**
4. **Interpret results** (SHAP, LIME, feature importance)
5. **Tune hyperparameters** if needed
6. **Validate on holdout test set**
7. **Deploy for predictions**
8. **Monitor performance** over time

## 📚 Additional Resources

- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Documentation**: https://github.com/marcotcr/lime

## ❓ Troubleshooting

**Low R² score?**
- Check data quality (missing values, outliers)
- Try feature engineering
- Increase model complexity
- Collect more data

**Overfitting?**
- Reduce model complexity
- Add regularization
- Use more training data
- Enable early stopping

**Slow training?**
- Reduce dataset size for testing
- Use GPU for LSTM/BERT
- Reduce number of features
- Use smaller batch sizes

## 🎉 Success Criteria

Your model is production-ready when:
- ✅ R² > 0.8 on validation set
- ✅ RMSE is acceptable for your use case
- ✅ Feature importance makes domain sense
- ✅ SHAP values are interpretable
- ✅ Model generalizes to new data

Good luck with your green building energy efficiency predictions! 🌱

