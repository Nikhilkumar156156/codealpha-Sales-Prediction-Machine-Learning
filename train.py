import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    # ---------------------------------------------------------
    # 1. Setup Directories & Load Data
    # ---------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    print(f"Downloading dataset from: {url}...")
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error downloading data: {e}")
        # Create a mock dataset if the download fails
        print("Creating a fallback mock dataset...")
        np.random.seed(42)
        n_samples = 200
        tv = np.random.uniform(10, 300, n_samples)
        radio = np.random.uniform(5, 50, n_samples)
        newspaper = np.random.uniform(2, 80, n_samples)
        # Sales formula with some noise
        sales = 2.0 + 0.05 * tv + 0.18 * radio + 0.003 * newspaper + np.random.normal(0, 1.5, n_samples)
        df = pd.DataFrame({
            "TV": tv,
            "Radio": radio,
            "Newspaper": newspaper,
            "Sales": sales
        })
        
    # Normalize column names to standard case: 'TV', 'Radio', 'Newspaper', 'Sales'
    rename_dict = {}
    for col in df.columns:
        if col.lower() == "tv":
            rename_dict[col] = "TV"
        elif col.lower() == "radio":
            rename_dict[col] = "Radio"
        elif col.lower() == "newspaper":
            rename_dict[col] = "Newspaper"
        elif col.lower() == "sales":
            rename_dict[col] = "Sales"
    df = df.rename(columns=rename_dict)

    print("\n--- Initial Data Overview ---")
    print(df.head())
    print(f"Data Shape: {df.shape}")
    
    # ---------------------------------------------------------
    # 2. Data Cleaning & Preprocessing
    # ---------------------------------------------------------
    # Drop index column if it exists (Kaggle dataset sometimes has 'Unnamed: 0')
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    print("\n--- Checking for Missing Values & Duplicates ---")
    print(df.isnull().sum())
    print(f"Duplicate rows count: {df.duplicated().sum()}")
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    # ---------------------------------------------------------
    # 3. Exploratory Data Analysis (EDA) & Visualizations
    # ---------------------------------------------------------
    print("\nGenerating exploratory plots...")
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Correlation Heatmap (using premium custom HSL-like palette)
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".3f", linewidths=0.5)
    plt.title("Correlation Matrix of Advertising Spend & Sales", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("plots/correlation_matrix.png", dpi=150)
    plt.close()
    
    # Plot 2: Pairwise Scatter Plots with regression line
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["TV", "Radio", "Newspaper"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for i, col in enumerate(features):
        sns.regplot(data=df, x=col, y="Sales", ax=axes[i], color=colors[i], 
                    scatter_kws={"alpha":0.6, "edgecolor":"w"}, line_kws={"color":"red", "lw":2})
        axes[i].set_title(f"{col} Spend vs Sales", fontsize=12, fontweight="bold")
        axes[i].set_xlabel(f"{col} Advertising Spend ($)")
        axes[i].set_ylabel("Sales ($)")
    
    plt.suptitle("Impact of Marketing Platforms on Sales", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("plots/spend_vs_sales.png", dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # 4. Feature Selection & Train-Test Split
    # ---------------------------------------------------------
    X = df[features]
    y = df["Sales"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # 5. Model Training & Evaluation
    # ---------------------------------------------------------
    # Model A: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    y_pred_lr = lr.predict(X_test_scaled)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    
    # Model B: Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    y_pred_rf = rf.predict(X_test_scaled)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    
    print("\n--- Model Performance Comparison ---")
    print(f"Linear Regression:")
    print(f"  MAE:  {mae_lr:.4f}")
    print(f"  RMSE: {rmse_lr:.4f}")
    print(f"  R2:   {r2_lr:.4f}")
    print(f"Random Forest Regressor:")
    print(f"  MAE:  {mae_rf:.4f}")
    print(f"  RMSE: {rmse_rf:.4f}")
    print(f"  R2:   {r2_rf:.4f}")
    
    # Determine the best model
    if r2_rf > r2_lr:
        best_model = rf
        best_name = "Random Forest Regressor"
        best_r2 = r2_rf
        best_pred = y_pred_rf
    else:
        best_model = lr
        best_name = "Linear Regression"
        best_r2 = r2_lr
        best_pred = y_pred_lr
        
    print(f"\nBest Model selected: {best_name} with R^2 score of {best_r2:.4f}")
    
    # Save Model and Scaler
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Best model and scaler successfully saved to disk.")
    
    # ---------------------------------------------------------
    # 6. Actionable Business Insights & Interpretations
    # ---------------------------------------------------------
    print("\n--- Actionable Business Insights ---")
    
    # Analysis 1: Linear Regression Coefficients (scaled features)
    # Scaled coef shows the change in Sales per standard deviation change in feature spend.
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient (Effect Size)": lr.coef_
    }).sort_values(by="Coefficient (Effect Size)", ascending=False)
    
    print("\nLinear Regression Feature Coefficients (Effect of standard deviation increase):")
    print(coef_df.to_string(index=False))
    
    # Analysis 2: Random Forest Feature Importances
    importances_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    print("\nRandom Forest Feature Importances:")
    print(importances_df.to_string(index=False))
    
    # Generate business recommendations text
    top_channel_lr = coef_df.iloc[0]["Feature"]
    second_channel_lr = coef_df.iloc[1]["Feature"]
    worst_channel_lr = coef_df.iloc[2]["Feature"]
    
    print("\nStrategic Recommendations:")
    print(f"1. Double Down on {top_channel_lr}: It has the highest positive correlation and return on advertising spend.")
    print(f"2. Optimize or Maintain {second_channel_lr}: It offers a solid support channel.")
    print(f"3. Rethink {worst_channel_lr}: Spend here yields the lowest direct sales lift. Budgets might be better allocated elsewhere.")
    
    # Plot 3: Actual vs Predicted Sales for the best model
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, best_pred, color="#8884d8", alpha=0.8, edgecolor="w", s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted Sales ({best_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Actual Sales ($)", fontsize=12)
    plt.ylabel("Predicted Sales ($)", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/actual_vs_predicted.png", dpi=150)
    plt.close()
    
    # Plot 4: Feature Importance chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importances_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Random Forest Relative Feature Importances", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Advertising Platform", fontsize=12)
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()
    
    print("\nAll plots successfully generated and saved to the 'plots/' directory.")

if __name__ == "__main__":
    main()
