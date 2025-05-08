
# E-commerce Sales Performance Analysis with Advanced ML and Visualizations

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Load Data
df = pd.read_csv("../data/ebay_mens_perfume.csv")

# 3. Data Cleaning and Feature Engineering
df.dropna(subset=["brand", "type", "sold", "available", "price"], inplace=True)
df.drop_duplicates(inplace=True)
df["sold_clipped"] = df["sold"].clip(upper=2000)
df["sales_rate"] = df["sold"] / df["available"].replace(0, np.nan)
df["sales_rate"] = df["sales_rate"].fillna(0)
df["price_bin"] = pd.cut(df["price"], bins=[0, 20, 40, 60, 80, 100, 150, 300],
                         labels=["0-20", "20-40", "40-60", "60-80", "80-100", "100-150", "150+"])
top_brands = df["brand"].value_counts().nlargest(10).index
df["brand_grouped"] = df["brand"].apply(lambda x: x if x in top_brands else "Other")

# 4. Visualizations

# Boxplot: Price by Type
plt.figure(figsize=(12, 6))
sns.boxplot(x="type", y="price", data=df)
plt.xticks(rotation=45)
plt.title("Price Distribution by Perfume Type")
plt.tight_layout()
plt.show()

# Heatmap: Brand vs Type
pivot_table = df.pivot_table(values="sold", index="brand_grouped", columns="type", aggfunc="mean")
plt.figure(figsize=(14, 6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average Items Sold by Brand and Type")
plt.tight_layout()
plt.show()

# Sales Rate Distribution
plt.figure(figsize=(10, 4))
sns.histplot(df["sales_rate"], bins=50, kde=True)
plt.title("Sales Rate Distribution (sold / available)")
plt.xlabel("Sales Rate")
plt.tight_layout()
plt.show()

# Price Elasticity
price_elasticity = df.groupby("price_bin")["sold"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=price_elasticity, x="price_bin", y="sold", marker="o")
plt.title("Average Items Sold vs Price Range")
plt.ylabel("Avg Sold")
plt.tight_layout()
plt.show()

# Bubble Plot: Price vs Sold
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="price", y="sold", size="available", hue="type", alpha=0.6, sizes=(20, 200))
plt.title("Price vs Sold with Availability Size")
plt.tight_layout()
plt.show()

# 5. Clustering
features = df[["price", "sold", "available"]].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(features_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="price", y="sold", hue="cluster", palette="Set2", alpha=0.8)
plt.title("KMeans Clusters of Products by Price and Sold")
plt.tight_layout()
plt.show()

# 6. Predictive Modeling

# Basic ML Dataset
ml_df = df[["brand_grouped", "type", "price", "available", "sold_clipped"]].copy()
ml_df = pd.get_dummies(ml_df, columns=["brand_grouped", "type"], drop_first=True)

X = ml_df.drop("sold_clipped", axis=1)
y = ml_df["sold_clipped"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, model_name):
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"  MAE : {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"  R²  : {r2_score(y_true, y_pred):.2f}")

evaluate(y_test, y_pred_lr, "Linear Regression")
evaluate(y_test, y_pred_rf, "Random Forest")

# Feature Importance
importances = rf.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.show()
