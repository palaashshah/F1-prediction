"""
F1 Model Performance Visualizer

This script trains the F1 prediction model and generates four key visualizations
to be included in the final report:
1.  A scatter plot comparing the model's predicted vs. actual race positions.
2.  A bar chart showing the importance of each feature in the model's predictions.
3.  A residuals plot to check for patterns in the prediction errors.
4.  A histogram showing the distribution of prediction errors.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def create_performance_graphs():
    """Trains the model and generates performance plots."""
    DATASET_FILE = 'f1_features_dataset.csv'
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset file '{DATASET_FILE}' not found.")
        return

    # --- Train the Model (same logic as before) ---
    X = df.drop(columns=['RacePosition', 'Year', 'EventName', 'FullName'])
    y = df['RacePosition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    errors = y_test - predictions

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Generate Prediction vs. Actual Scatter Plot ---
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.scatter(y_test, predictions, alpha=0.5, edgecolors='k', c='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Finishing Position', fontsize=12)
    ax1.set_ylabel('Predicted Finishing Position', fontsize=12)
    ax1.set_title('Model Performance: Actual vs. Predicted Positions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True)
    plt.savefig('actual_vs_predicted_plot.png', dpi=300, bbox_inches='tight')
    print("-> Graph 1: 'actual_vs_predicted_plot.png' saved successfully.")

    # --- 2. Generate Feature Importance Bar Chart ---
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue')
    ax2.invert_yaxis()
    ax2.set_xlabel('Importance Score', fontsize=12)
    ax2.set_title('Top 10 Most Important Features for Prediction', fontsize=14, fontweight='bold')
    plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
    print("-> Graph 2: 'feature_importance_plot.png' saved successfully.")

    # --- 3. NEW: Generate Residuals Plot ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(y_test, errors, alpha=0.5, edgecolors='k', c='green')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Actual Finishing Position', fontsize=12)
    ax3.set_ylabel('Prediction Error (Residuals)', fontsize=12)
    ax3.set_title('Residuals Plot', fontsize=14, fontweight='bold')
    ax3.grid(True)
    plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
    print("-> Graph 3: 'residuals_plot.png' saved successfully.")

    # --- 4. NEW: Generate Prediction Error Histogram ---
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(errors, bins=20, edgecolor='black', color='purple', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', lw=2)
    ax4.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.75)
    plt.savefig('prediction_error_histogram.png', dpi=300, bbox_inches='tight')
    print("-> Graph 4: 'prediction_error_histogram.png' saved successfully.")


if __name__ == "__main__":
    create_performance_graphs()

