# Visualization Guide

## Overview

This project includes a comprehensive visualization suite that creates presentation-ready charts for dataset exploration, model results, predictions, insights, and recommendations.

## Generated Visualizations

### 1. Dataset Exploration - Target Variable (`01_dataset_exploration_target.png`)
**Purpose**: Understand the distribution and characteristics of the energy efficiency target variable.

**Contents**:
- Histogram showing energy efficiency distribution
- Box plot for outlier detection
- Scatter plot: Energy consumption vs efficiency
- Statistical summary (mean, median, std dev, min, max, skewness)

**Insights**:
- Distribution shape and normality
- Presence of outliers
- Relationship between consumption and efficiency
- Key statistics for understanding the data

---

### 2. Feature Categories Analysis (`02_feature_categories_analysis.png`)
**Purpose**: Analyze how different feature categories relate to energy efficiency.

**Contents**:
- IoT Sensors: Temperature, Humidity, Air Quality vs Efficiency
- Building Design: Orientation, Insulation, Roof Type vs Efficiency

**Insights**:
- Which sensor readings correlate with efficiency
- Impact of building design choices
- Feature relationships and patterns

---

### 3. Correlation Heatmap (`03_correlation_heatmap.png`)
**Purpose**: Identify relationships between all features and the target variable.

**Contents**:
- Correlation matrix for all numeric features
- Color-coded heatmap (red = negative, blue = positive)
- Correlation values annotated

**Insights**:
- Strong positive/negative correlations
- Feature multicollinearity
- Most important features for prediction
- Feature interactions

---

### 4. Model Performance (`04_model_performance.png`)
**Purpose**: Evaluate model accuracy and prediction quality.

**Contents**:
- Predictions vs Actual scatter plot with perfect prediction line
- Residual plot (errors vs predictions)
- Residuals distribution histogram
- Absolute error distribution

**Metrics Displayed**:
- R² Score (variance explained)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

**Insights**:
- Model accuracy and fit quality
- Error patterns and bias
- Prediction reliability
- Areas needing improvement

---

### 5. Feature Importance Analysis (`05_feature_importance_analysis.png`)
**Purpose**: Identify which features most influence energy efficiency predictions.

**Contents**:
- Top 20 most important features (horizontal bar chart)
- Cumulative importance plot (shows how many features needed for X% importance)

**Insights**:
- Most critical features for energy efficiency
- Feature ranking by importance
- Optimal feature subset size
- Feature selection guidance

---

### 6. Insights and Recommendations (`06_insights_recommendations.png`)
**Purpose**: Provide actionable insights and strategic recommendations.

**Contents**:
- Energy efficiency by building material
- Energy efficiency by orientation
- Energy efficiency by temperature range
- Energy efficiency by insulation level
- Strategic recommendations panel

**Key Recommendations**:
1. **Building Material Selection**: Use concrete/composite materials
2. **Orientation Optimization**: Consider solar path and wind patterns
3. **Insulation Improvements**: Higher insulation = better efficiency
4. **Temperature Management**: Maintain optimal ranges
5. **Operational Efficiency**: Improve logistics and compliance
6. **Climate Considerations**: Leverage solar and wind
7. **Consumer Engagement**: Improve awareness and perception

---

### 7. Comprehensive Dashboard (`07_comprehensive_dashboard.png`)
**Purpose**: Single-page overview of all key metrics and insights.

**Contents**:
- Dataset overview (samples, features, statistics)
- Feature categories breakdown
- Energy efficiency distribution
- Top 5 correlated features
- Model performance summary
- Key insights panels

**Use Cases**:
- Executive presentations
- Quick project overview
- Stakeholder communication
- Project documentation

---

## How to Generate Visualizations

### Method 1: Run the Script Directly
```bash
python utils/create_visualizations.py
```

### Method 2: Import and Use
```python
from utils.create_visualizations import VisualizationSuite

# Create visualizations
viz = VisualizationSuite(data_path='data/uci_energy_efficiency.csv')
viz.run_all()

# Or run individual functions
viz.load_data()
viz.create_exploration_visualizations()
viz.create_model_performance_visualizations()
viz.create_feature_importance_analysis()
viz.create_insights_and_recommendations()
viz.create_summary_dashboard()
```

### Method 3: Integrate with Pipeline
```python
from train_pipeline import EnergyEfficiencyPipeline
from utils.create_visualizations import VisualizationSuite

# Train model
pipeline = EnergyEfficiencyPipeline()
pipeline.run(data_path='data/uci_energy_efficiency.csv')

# Create visualizations
viz = VisualizationSuite()
viz.run_all()
```

---

## Visualization Features

### Design Elements
- **Professional styling**: Seaborn darkgrid theme
- **Color schemes**: Consistent, accessible color palettes
- **High resolution**: 300 DPI for publication quality
- **Clear labels**: Descriptive titles and axis labels
- **Grid lines**: For easy value reading
- **Legends**: Where applicable

### Customization
Edit `utils/create_visualizations.py` to:
- Change color schemes
- Adjust figure sizes
- Modify chart types
- Add custom metrics
- Include additional insights

---

## Presentation Tips

### For Technical Audiences
- Use: Correlation heatmap, Feature importance, Model performance
- Focus on: Metrics, statistical significance, model accuracy

### For Business Audiences
- Use: Comprehensive dashboard, Insights & recommendations
- Focus on: Actionable insights, ROI, strategic recommendations

### For Stakeholders
- Use: Summary dashboard, Insights visualization
- Focus on: Key findings, recommendations, impact

---

## File Locations

All visualizations are saved in the `results/` directory:
```
results/
├── 01_dataset_exploration_target.png
├── 02_feature_categories_analysis.png
├── 03_correlation_heatmap.png
├── 04_model_performance.png
├── 05_feature_importance_analysis.png
├── 06_insights_recommendations.png
└── 07_comprehensive_dashboard.png
```

---

## Dependencies

Required packages (included in `requirements.txt`):
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Metrics calculation

---

## Troubleshooting

### Visualizations not generating?
1. Ensure data file exists: `data/uci_energy_efficiency.csv`
2. Check model file exists: `models/xgboost_model.pkl`
3. Verify dependencies: `pip install -r requirements.txt`

### Missing model performance plots?
- Train the model first: `python train_pipeline.py --data data/uci_energy_efficiency.csv`

### Import errors?
- Run from project root directory
- Ensure `utils/` directory is in Python path

---

## Best Practices

1. **Generate after training**: Create visualizations after model training for complete results
2. **Update regularly**: Regenerate when data or models change
3. **Customize for audience**: Select appropriate visualizations for your audience
4. **Include in reports**: Use visualizations in presentations and documentation
5. **Version control**: Track visualization versions with model versions

---

## Example Usage in Presentations

### Slide 1: Overview
- Use: Comprehensive Dashboard

### Slide 2: Data Understanding
- Use: Dataset Exploration, Feature Categories

### Slide 3: Model Performance
- Use: Model Performance, Feature Importance

### Slide 4: Insights
- Use: Insights and Recommendations

### Slide 5: Next Steps
- Use: Recommendations from insights visualization

---

## Additional Notes

- All visualizations are saved as PNG files (300 DPI)
- Files are named sequentially for easy organization
- Visualizations can be customized by editing the script
- Compatible with Jupyter notebooks (can be imported)
- Suitable for publication and presentation use

---

**Last Updated**: December 2024
**Version**: 1.0

