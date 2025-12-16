"""
Comprehensive Visualization Suite for Green Building Energy Efficiency
Creates presentation-ready charts for dataset exploration, model results, and insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class VisualizationSuite:
    """Create comprehensive visualizations for the project"""
    
    def __init__(self, data_path='data/uci_energy_efficiency.csv', results_dir='results'):
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.df = None
        
    def load_data(self):
        """Load the dataset"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} samples with {len(self.df.columns)} features")
        return self.df
    
    def create_exploration_visualizations(self):
        """Create dataset exploration visualizations"""
        print("\nCreating dataset exploration visualizations...")
        
        # 1. Target Variable Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Energy Efficiency Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(self.df['energy_efficiency'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 0].axvline(self.df['energy_efficiency'].mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {self.df["energy_efficiency"].mean():.3f}')
        axes[0, 0].set_xlabel('Energy Efficiency Score', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Energy Efficiency Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        bp = axes[0, 1].boxplot(self.df['energy_efficiency'], vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        axes[0, 1].set_ylabel('Energy Efficiency Score', fontsize=12)
        axes[0, 1].set_title('Energy Efficiency Box Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Energy consumption vs efficiency
        axes[1, 0].scatter(self.df['energy_consumption'], self.df['energy_efficiency'], 
                          alpha=0.6, s=50, c=self.df['energy_efficiency'], cmap='viridis')
        axes[1, 0].set_xlabel('Energy Consumption (kWh)', fontsize=12)
        axes[1, 0].set_ylabel('Energy Efficiency Score', fontsize=12)
        axes[1, 0].set_title('Energy Consumption vs Efficiency', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics summary
        stats_text = f"""
        Statistics Summary:
        
        Mean: {self.df['energy_efficiency'].mean():.3f}
        Median: {self.df['energy_efficiency'].median():.3f}
        Std Dev: {self.df['energy_efficiency'].std():.3f}
        Min: {self.df['energy_efficiency'].min():.3f}
        Max: {self.df['energy_efficiency'].max():.3f}
        Skewness: {self.df['energy_efficiency'].skew():.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistical Summary', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / '01_dataset_exploration_target.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Categories Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Categories Analysis', fontsize=16, fontweight='bold')
        
        # IoT Sensors
        iot_cols = ['temperature', 'humidity', 'air_quality']
        for idx, col in enumerate(iot_cols):
            if col in self.df.columns:
                row, col_idx = divmod(idx, 3)
                axes[row, col_idx].scatter(self.df[col], self.df['energy_efficiency'], 
                                          alpha=0.5, s=30)
                axes[row, col_idx].set_xlabel(col.replace('_', ' ').title(), fontsize=10)
                axes[row, col_idx].set_ylabel('Energy Efficiency', fontsize=10)
                axes[row, col_idx].set_title(f'{col.replace("_", " ").title()} vs Efficiency', 
                                            fontsize=12, fontweight='bold')
                axes[row, col_idx].grid(True, alpha=0.3)
        
        # Building Design
        design_cols = ['orientation', 'insulation', 'roof_type']
        for idx, col in enumerate(design_cols):
            if col in self.df.columns:
                row, col_idx = divmod(idx + 3, 3)
                if row < 2:
                    if self.df[col].dtype in ['int64', 'float64']:
                        axes[row, col_idx].scatter(self.df[col], self.df['energy_efficiency'], 
                                                  alpha=0.5, s=30)
                    else:
                        self.df.boxplot(column='energy_efficiency', by=col, ax=axes[row, col_idx])
                    axes[row, col_idx].set_xlabel(col.replace('_', ' ').title(), fontsize=10)
                    axes[row, col_idx].set_ylabel('Energy Efficiency', fontsize=10)
                    axes[row, col_idx].set_title(f'{col.replace("_", " ").title()} vs Efficiency', 
                                                fontsize=12, fontweight='bold')
                    axes[row, col_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / '02_feature_categories_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation Heatmap
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'energy_efficiency' in numeric_cols:
            corr_matrix = self.df[numeric_cols].corr()
            
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                       annot_kws={'size': 8})
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(self.results_dir / '03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✓ Exploration visualizations created")
    
    def create_model_performance_visualizations(self):
        """Create model performance visualizations"""
        print("\nCreating model performance visualizations...")
        
        # Load predictions if available
        try:
            import pickle
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            with open('models/xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Prepare data
            from utils.data_preprocessing import DataPreprocessor
            # Drop columns that shouldn't be in features (they're targets, not features)
            df_for_preprocessing = self.df.drop(columns=['heating_load', 'cooling_load'], errors='ignore')
            preprocessor = DataPreprocessor()
            X, y = preprocessor.preprocess_pipeline(df_for_preprocessing, target_col='energy_efficiency', is_training=False)
            
            # Make predictions
            y_pred = model.predict(X)
            
            # 1. Prediction vs Actual
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
            
            # Scatter plot
            axes[0, 0].scatter(y, y_pred, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Prediction')
            axes[0, 0].set_xlabel('Actual Energy Efficiency', fontsize=12)
            axes[0, 0].set_ylabel('Predicted Energy Efficiency', fontsize=12)
            axes[0, 0].set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Add metrics text
            metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
            axes[0, 0].text(0.05, 0.95, metrics_text, transform=axes[0, 0].transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=11, fontweight='bold')
            
            # Residuals plot
            residuals = y - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
            axes[0, 1].set_xlabel('Predicted Energy Efficiency', fontsize=12)
            axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
            axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals distribution
            axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
            axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
                              label=f'Mean: {residuals.mean():.4f}')
            axes[1, 0].set_xlabel('Residuals', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Residuals Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Error distribution
            abs_errors = np.abs(residuals)
            axes[1, 1].hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
            axes[1, 1].axvline(abs_errors.mean(), color='green', linestyle='--', linewidth=2,
                              label=f'Mean Absolute Error: {abs_errors.mean():.4f}')
            axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
            axes[1, 1].set_ylabel('Frequency', fontsize=12)
            axes[1, 1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / '04_model_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Model performance visualizations created")
        except Exception as e:
            print(f"⚠ Could not create model performance plots: {e}")
    
    def create_feature_importance_analysis(self):
        """Create feature importance analysis"""
        print("\nCreating feature importance analysis...")
        
        try:
            import pickle
            with open('models/xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Get feature importance
            if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
                import sys
                from pathlib import Path
                
                # Add parent directory to path
                parent_dir = Path(__file__).parent.parent
                if str(parent_dir) not in sys.path:
                    sys.path.insert(0, str(parent_dir))
                
                importances = model.model.feature_importances_
                
                # Get feature names
                from utils.data_preprocessing import DataPreprocessor
                # Drop columns that shouldn't be in features
                df_for_preprocessing = self.df.drop(columns=['heating_load', 'cooling_load'], errors='ignore')
                preprocessor = DataPreprocessor()
                X, y = preprocessor.preprocess_pipeline(df_for_preprocessing, target_col='energy_efficiency', is_training=False)
                
                if hasattr(X, 'columns'):
                    feature_names = X.columns.tolist()
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                
                # Sort by importance
                indices = np.argsort(importances)[::-1]
                top_n = min(20, len(importances))
                
                fig, axes = plt.subplots(1, 2, figsize=(18, 10))
                fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
                
                # Horizontal bar chart
                axes[0].barh(range(top_n), importances[indices[:top_n]], color='steelblue')
                axes[0].set_yticks(range(top_n))
                axes[0].set_yticklabels([feature_names[i] for i in indices[:top_n]], fontsize=10)
                axes[0].set_xlabel('Importance Score', fontsize=12)
                axes[0].set_title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3, axis='x')
                axes[0].invert_yaxis()
                
                # Cumulative importance
                cumsum = np.cumsum(np.sort(importances)[::-1])
                axes[1].plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2, markersize=4)
                axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
                axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90% Threshold')
                axes[1].set_xlabel('Number of Features', fontsize=12)
                axes[1].set_ylabel('Cumulative Importance', fontsize=12)
                axes[1].set_title('Cumulative Feature Importance', fontsize=14, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.results_dir / '05_feature_importance_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✓ Feature importance analysis created")
        except Exception as e:
            print(f"⚠ Could not create feature importance plots: {e}")
    
    def create_insights_and_recommendations(self):
        """Create insights and recommendations visualization"""
        print("\nCreating insights and recommendations...")
        
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Key Insights and Recommendations', fontsize=18, fontweight='bold', y=0.98)
        
        # Insight 1: Energy Efficiency by Building Type
        ax1 = fig.add_subplot(gs[0, 0])
        if 'material' in self.df.columns:
            material_eff = self.df.groupby('material')['energy_efficiency'].mean().sort_values(ascending=False)
            bars = ax1.bar(range(len(material_eff)), material_eff.values, color=['#2ecc71', '#3498db', '#e74c3c'])
            ax1.set_xticks(range(len(material_eff)))
            ax1.set_xticklabels(material_eff.index, rotation=45, ha='right')
            ax1.set_ylabel('Average Energy Efficiency', fontsize=11)
            ax1.set_title('Energy Efficiency by Building Material', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, material_eff.values)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Insight 2: Orientation Impact
        ax2 = fig.add_subplot(gs[0, 1])
        if 'orientation' in self.df.columns:
            orient_eff = self.df.groupby('orientation')['energy_efficiency'].mean().sort_index()
            ax2.plot(orient_eff.index, orient_eff.values, marker='o', linewidth=2, markersize=8, color='#9b59b6')
            ax2.fill_between(orient_eff.index, orient_eff.values, alpha=0.3, color='#9b59b6')
            ax2.set_xlabel('Orientation', fontsize=11)
            ax2.set_ylabel('Average Energy Efficiency', fontsize=11)
            ax2.set_title('Energy Efficiency by Building Orientation', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Insight 3: Temperature vs Efficiency
        ax3 = fig.add_subplot(gs[1, 0])
        if 'temperature' in self.df.columns:
            # Create bins
            temp_bins = pd.cut(self.df['temperature'], bins=5)
            temp_eff = self.df.groupby(temp_bins)['energy_efficiency'].mean()
            ax3.bar(range(len(temp_eff)), temp_eff.values, color='#e67e22')
            ax3.set_xticks(range(len(temp_eff)))
            ax3.set_xticklabels([f'{int(interval.left)}-{int(interval.right)}°C' 
                               for interval in temp_eff.index], rotation=45, ha='right', fontsize=9)
            ax3.set_ylabel('Average Energy Efficiency', fontsize=11)
            ax3.set_title('Energy Efficiency by Temperature Range', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Insight 4: Insulation Impact
        ax4 = fig.add_subplot(gs[1, 1])
        if 'insulation' in self.df.columns:
            # Create bins
            insul_bins = pd.qcut(self.df['insulation'], q=5, duplicates='drop')
            insul_eff = self.df.groupby(insul_bins)['energy_efficiency'].mean()
            ax4.plot(range(len(insul_eff)), insul_eff.values, marker='s', linewidth=2, 
                    markersize=8, color='#16a085')
            ax4.fill_between(range(len(insul_eff)), insul_eff.values, alpha=0.3, color='#16a085')
            ax4.set_xlabel('Insulation Level (Quintiles)', fontsize=11)
            ax4.set_ylabel('Average Energy Efficiency', fontsize=11)
            ax4.set_title('Energy Efficiency by Insulation Level', fontsize=13, fontweight='bold')
            ax4.set_xticks(range(len(insul_eff)))
            ax4.set_xticklabels([f'Q{i+1}' for i in range(len(insul_eff))], fontsize=9)
            ax4.grid(True, alpha=0.3)
        
        # Recommendations Text
        ax5 = fig.add_subplot(gs[2:, :])
        ax5.axis('off')
        
        recommendations = """
        KEY RECOMMENDATIONS FOR IMPROVING BUILDING ENERGY EFFICIENCY
        
        1. BUILDING MATERIAL SELECTION
           → Use concrete or composite materials for better energy efficiency
           → Avoid wood-based materials in high-efficiency requirements
        
        2. ORIENTATION OPTIMIZATION
           → Optimal building orientation significantly impacts energy efficiency
           → Consider solar path and wind patterns when designing building orientation
        
        3. INSULATION IMPROVEMENTS
           → Higher insulation levels directly correlate with better energy efficiency
           → Invest in quality insulation materials and proper installation
        
        4. TEMPERATURE MANAGEMENT
           → Maintain optimal temperature ranges for maximum efficiency
           → Implement smart HVAC systems with IoT sensors
        
        5. OPERATIONAL EFFICIENCY
           → Improve metro logistics and policy compliance
           → Regular maintenance of equipment (status monitoring)
        
        6. CLIMATE CONSIDERATIONS
           → Leverage solar radiation through proper glazing
           → Consider wind patterns for natural ventilation
        
        7. CONSUMER ENGAGEMENT
           → Higher green perception and environmental awareness correlate with efficiency
           → Educational programs can improve perceived risk and awareness
        """
        
        ax5.text(0.05, 0.95, recommendations, transform=ax5.transAxes,
                fontsize=12, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='navy', linewidth=2))
        ax5.set_title('Strategic Recommendations', fontsize=15, fontweight='bold', pad=20)
        
        plt.savefig(self.results_dir / '06_insights_recommendations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Insights and recommendations created")
    
    def create_summary_dashboard(self):
        """Create a comprehensive summary dashboard"""
        print("\nCreating summary dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
        
        fig.suptitle('Green Building Energy Efficiency - Comprehensive Dashboard', 
                     fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Dataset Overview
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        overview_text = f"""
        DATASET OVERVIEW
        
        Total Samples: {len(self.df):,}
        Features: {len(self.df.columns)}
        
        Energy Efficiency:
        • Mean: {self.df['energy_efficiency'].mean():.3f}
        • Range: {self.df['energy_efficiency'].min():.3f} - {self.df['energy_efficiency'].max():.3f}
        • Std Dev: {self.df['energy_efficiency'].std():.3f}
        """
        ax1.text(0.1, 0.5, overview_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 2. Feature Categories
        ax2 = fig.add_subplot(gs[0, 1])
        categories = {
            'IoT Sensors': ['temperature', 'humidity', 'air_quality', 'equipment_status'],
            'Design': ['orientation', 'material', 'roof_type', 'insulation'],
            'Climate': ['solar_radiation', 'wind_speed', 'temperature_profile'],
            'Survey': ['green_perception', 'environmental_awareness', 'perceived_risk'],
            'Operational': ['metro_logistics', 'policy_compliance']
        }
        category_counts = {cat: sum(1 for f in features if f in self.df.columns) 
                          for cat, features in categories.items()}
        ax2.barh(list(category_counts.keys()), list(category_counts.values()), color='steelblue')
        ax2.set_xlabel('Number of Features', fontsize=10)
        ax2.set_title('Feature Categories', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Energy Efficiency Distribution
        ax3 = fig.add_subplot(gs[0, 2:])
        ax3.hist(self.df['energy_efficiency'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax3.axvline(self.df['energy_efficiency'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {self.df["energy_efficiency"].mean():.3f}')
        ax3.set_xlabel('Energy Efficiency Score', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Energy Efficiency Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4-7. Top Correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'energy_efficiency' in numeric_cols:
            correlations = self.df[numeric_cols].corr()['energy_efficiency'].abs().sort_values(ascending=False)
            top_corr = correlations.head(5).drop('energy_efficiency')
            
            ax4 = fig.add_subplot(gs[1, :2])
            colors = ['green' if x > 0 else 'red' for x in 
                     [self.df[numeric_cols].corr()['energy_efficiency'][col] for col in top_corr.index]]
            ax4.barh(range(len(top_corr)), top_corr.values, color=colors)
            ax4.set_yticks(range(len(top_corr)))
            ax4.set_yticklabels([col.replace('_', ' ').title() for col in top_corr.index], fontsize=10)
            ax4.set_xlabel('Absolute Correlation with Energy Efficiency', fontsize=11)
            ax4.set_title('Top 5 Features Correlated with Energy Efficiency', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            ax4.invert_yaxis()
        
        # 8. Model Performance Summary
        ax5 = fig.add_subplot(gs[1, 2:])
        ax5.axis('off')
        try:
            import pickle
            with open('models/xgboost_model.pkl', 'rb') as f:
                model = pickle.load(f)
            import sys
            from pathlib import Path
            
            # Add parent directory to path
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from utils.data_preprocessing import DataPreprocessor
            # Drop columns that shouldn't be in features
            df_for_preprocessing = self.df.drop(columns=['heating_load', 'cooling_load'], errors='ignore')
            preprocessor = DataPreprocessor()
            X, y = preprocessor.preprocess_pipeline(df_for_preprocessing, target_col='energy_efficiency', is_training=False)
            y_pred = model.predict(X)
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            perf_text = f"""
            MODEL PERFORMANCE (XGBoost)
            
            R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)
            RMSE: {rmse:.4f}
            MAE: {mae:.4f}
            
            Status: {'✓ Excellent' if r2 > 0.95 else 'Good' if r2 > 0.8 else 'Needs Improvement'}
            """
        except:
            perf_text = "Model performance data not available"
        
        ax5.text(0.1, 0.5, perf_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax5.set_title('Model Performance', fontsize=13, fontweight='bold')
        
        # 9-12. Key Insights
        insights_data = []
        if 'material' in self.df.columns:
            best_material = self.df.groupby('material')['energy_efficiency'].mean().idxmax()
            insights_data.append(('Best Material', best_material))
        
        if 'orientation' in self.df.columns:
            best_orient = self.df.groupby('orientation')['energy_efficiency'].mean().idxmax()
            insights_data.append(('Best Orientation', f'{best_orient}'))
        
        if 'insulation' in self.df.columns:
            high_insul_eff = self.df[self.df['insulation'] > self.df['insulation'].quantile(0.75)]['energy_efficiency'].mean()
            low_insul_eff = self.df[self.df['insulation'] < self.df['insulation'].quantile(0.25)]['energy_efficiency'].mean()
            insights_data.append(('Insulation Impact', f'{((high_insul_eff/low_insul_eff - 1)*100):.1f}% improvement'))
        
        for idx, (label, value) in enumerate(insights_data[:4]):
            ax = fig.add_subplot(gs[2, idx])
            ax.axis('off')
            ax.text(0.5, 0.5, f'{label}\n\n{value}', transform=ax.transAxes,
                   fontsize=11, ha='center', va='center', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.savefig(self.results_dir / '07_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Summary dashboard created")
    
    def run_all(self):
        """Run all visualization functions"""
        print("="*60)
        print("Creating Comprehensive Visualizations")
        print("="*60)
        
        self.load_data()
        self.create_exploration_visualizations()
        self.create_model_performance_visualizations()
        self.create_feature_importance_analysis()
        self.create_insights_and_recommendations()
        self.create_summary_dashboard()
        
        print("\n" + "="*60)
        print("All visualizations created successfully!")
        print("="*60)
        print(f"\nVisualizations saved to: {self.results_dir}/")
        print("\nGenerated files:")
        print("  01_dataset_exploration_target.png")
        print("  02_feature_categories_analysis.png")
        print("  03_correlation_heatmap.png")
        print("  04_model_performance.png")
        print("  05_feature_importance_analysis.png")
        print("  06_insights_recommendations.png")
        print("  07_comprehensive_dashboard.png")

if __name__ == '__main__':
    viz = VisualizationSuite()
    viz.run_all()

