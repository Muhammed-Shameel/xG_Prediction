import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ---------------------------
# 1. Load and Clean Data
# ---------------------------
print("📌 Step 1: Loading and Cleaning Data...")
df = pd.read_csv('C:/AML/Shots.csv')

print("Initial dataset shape:", df.shape)
print(df.head())


# Drop unnecessary columns
drop_columns = ["id", "match_id", "player", "player_id", "date", "h_team", "a_team"]
df.drop(columns=drop_columns, inplace=True, errors="ignore")


# Remove rows with missing key features
df.dropna(subset=["player_assisted", "lastAction"], inplace=True)
print(f"✅ Dataset shape after cleaning: {df.shape}")


# ---------------------------
# 2. Feature Engineering
# ---------------------------
print("\n📌 Step 2: Engineering Features...")
df["X_real"] = df["X"] * 105
df["Y_real"] = df["Y"] * 68

goal_x, goal_y = 105, 34  # Define goal position

df["distance_from_goal"] = np.sqrt((df["X_real"] - goal_x) ** 2 + (df["Y_real"] - goal_y) ** 2)
df["shot_angle"] = np.degrees(np.arctan2(np.abs(df["Y_real"] - goal_y), (goal_x - df["X_real"])) + 1e-9)

df["goal_difference"] = df["h_goals"] - df["a_goals"]
df["goal_pressure"] = df["goal_difference"] * df["shot_angle"]
df["angle_distance_interaction"] = df["shot_angle"] * df["distance_from_goal"]

plt.figure(figsize=(8, 5))
sns.histplot(df["distance_from_goal"], bins=30, kde=True)
plt.title("Distribution of Shot Distances")
plt.xlabel("Distance from Goal (meters)")
plt.ylabel("Frequency")
plt.show()

# ---------------------------
# 3. Categorical Encoding
# ---------------------------
print("\n📌 Step 3: Encoding Categorical Features...")
df["h_a"] = LabelEncoder().fit_transform(df["h_a"])

top_assisters = df["player_assisted"].value_counts().nlargest(10).index
df["player_assisted"] = np.where(df["player_assisted"].isin(top_assisters), df["player_assisted"], "Other")

categorical_features = ["situation", "shotType", "lastAction", "player_assisted"]
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# ---------------------------
# 4. Target Variable Check
# ---------------------------
if "xG" not in df.columns:
    print("\n⚠️ Warning: Creating dummy xG values for demonstration")
    np.random.seed(42)
    df["xG"] = np.random.uniform(0, 0.5, size=len(df))  # Dummy values

# ---------------------------
# 5. Data Sampling
# ---------------------------
print("\n📌 Step 4: Sampling Data...")
sample_size = 15000
df = df.sample(n=min(sample_size, len(df)), random_state=42)
print(f"✅ Sampled dataset shape: {df.shape}")

# ---------------------------
# 6. Train-Test Split
# ---------------------------
print("\n📌 Step 5: Splitting Data...")
X = df.drop(columns=["xG", "result"] if "result" in df.columns else "xG")
y = df["xG"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 7. Feature Scaling
# ---------------------------
print("\n📌 Step 6: Scaling Numerical Features...")
numerical_features = ["X", "Y", "X_real", "Y_real", "distance_from_goal",
                      "shot_angle", "goal_difference", "angle_distance_interaction", "goal_pressure"]
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# ---------------------------
# 8. Model Training
# ---------------------------
print("\n📌 Step 7: Training Models...")
models = {
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42, verbose=-1)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results[name] = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred)
    }
    print(f"\n{name} Performance:")
    print(f"- MAE: {results[name]['MAE']:.4f}")
    print(f"- RMSE: {results[name]['RMSE']:.4f}")
    print(f"- R²: {results[name]['R²']:.4f}")

# ---------------------------
# 9. Model Comparison
# ---------------------------
print("\n📌 Step 8: Model Comparison...")
results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(10, 6))
results_df[["MAE", "RMSE"]].plot(kind="bar", rot=45)
plt.title("Model Error Comparison")
plt.ylabel("Error Value")
plt.tight_layout()
plt.show()

residuals = y_test - models["LightGBM"].predict(X_test)
sns.scatterplot(x=y_test, y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residual Plot for LightGBM")
plt.show()

# ---------------------------
# 10. SHAP Analysis (Fixed)
# ---------------------------

import time
print("\n📌 Step 9: Performing SHAP Analysis...")

# ✅ Start Timer
start_time = time.time()

best_model = models["LightGBM"]

# ✅ Reduce to 20 samples for faster SHAP computation
X_sample = X_train.sample(n=20, random_state=42)

# ✅ Use TreeExplainer with LightGBM fix
explainer = shap.TreeExplainer(best_model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_sample)

# ✅ Check if SHAP values are a list (LightGBM sometimes returns a list)
if isinstance(shap_values, list):
    shap_values = shap_values[0]  # Select the first element

# ✅ Force SHAP to render properly
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=True)
plt.title("Feature Importance using SHAP Values")
plt.tight_layout()
plt.show()

# ✅ End Timer
end_time = time.time()
print(f"✅ SHAP Analysis Completed in {end_time - start_time:.2f} seconds")


# ---------------------------
# 11. Diagnostic Plots
# ---------------------------
print("\n📌 Step 10: Generating Diagnostic Plots...")
best_pred = best_model.predict(X_test)
residuals = y_test - best_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=best_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Plot for LightGBM Predictions")
plt.xlabel("Predicted xG Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=best_pred, scatter_kws={'alpha': 0.3})
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("Actual vs Predicted xG Values")
plt.xlabel("Actual xG")
plt.ylabel("Predicted xG")
plt.tight_layout()
plt.show()


