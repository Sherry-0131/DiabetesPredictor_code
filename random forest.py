import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib  # 引入 joblib 库用于保存和加载模型
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATASET
# ==========================================
print("=== Step 1: Loading Dataset ===")
file_path = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

# 自动提取第一列作为标签 y，其余列作为特征 X
y = df.iloc[:, 0].astype(int)
X = df.iloc[:, 1:]
feature_names = X.columns.tolist()

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
print(f"Total features detected: {len(feature_names)}\n")


# ==========================================
# 2. DATASET PARTITION (7:2:1 RATIO, WITH RANDOM SEED)
# ==========================================
print("=== Step 2: Stratified & Randomized Data Partitioning (7:2:1) ===")

# 第一步：完全打乱并切出 70% 作为训练集，留下 30% 供后续划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=3, stratify=y
)

# 第二步：将剩下的 30% 再次随机打乱，按 2:1 划分给验证集(20%)和测试集(10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
)

print(f"Train set shape      : {X_train.shape[0]} rows (70%)")
print(f"Validation set shape : {X_val.shape[0]} rows (20%)")
print(f"Test set shape       : {X_test.shape[0]} rows (10%)\n")


# ==========================================
# 3. RANDOM FOREST TRAINING
# ==========================================
print("=== Step 3: Training Balanced Random Forest Classifier ===")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Random Forest model training completed successfully.\n")


# ==========================================
# 4. EXCLUSIVELY CLASSIFICATION METRICS (前三个指标)
# ==========================================
print("=== Step 4: Generating Classification Core Metrics on Test Set ===")
y_pred = rf_model.predict(X_test)

print("\n--- Core Classification Performance Matrix ---")
print(classification_report(
    y_test, 
    y_pred, 
    target_names=['0: No Diabetes', '1: Prediabetes', '2: Diabetes'],
    digits=4
))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 5. FEATURE IMPORTANCE PERCENTAGE
# ==========================================
print("\n=== Step 5: Extracted Feature Importance Percentages ===")
importances = rf_model.feature_importances_
importance_percentages = importances * 100

feature_importance_df = pd.DataFrame({
    'Clinical_Feature': feature_names,
    'Importance_Weight (%)': importance_percentages
}).sort_values(by='Importance_Weight (%)', ascending=False)

print(feature_importance_df.to_string(index=False))


# ==========================================
# 💾 🔥 核心新增: Step 6: 保存模型到本地
# ==========================================
print("\n=== Step 6: Saving Trained Model ===")
# 定义模型文件名，建议带上算法名和 random_state 以防后续版本混淆
model_filename = "random_forest_diabetes_model.joblib"

# 使用 joblib.dump 序列化模型并保存到当前工作目录
joblib.dump(rf_model, model_filename)
print(f"Success: Model has been saved locally as '{model_filename}'")