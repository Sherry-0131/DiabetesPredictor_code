import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib  # 引入 joblib 用于模型本地保存
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD DATASET & PREPARE VARIABLES
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
# 2. DATASET PARTITION (7:2:1 RATIO)
# ==========================================
print("=== Step 2: Partitioning Dataset into Train/Validation/Test (7:2:1) ===")
# 第一步：先划分 70% 作为训练集，留下 30% 供后续划分
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=2, stratify=y
)

# 第二步：将剩下的 30% 按 2:1 比例划分给验证集(20%)和测试集(10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
)

print(f"Train set shape      : {X_train.shape[0]} rows (70%)")
print(f"Validation set shape : {X_val.shape[0]} rows (20%)")
print(f"Test set shape       : {X_test.shape[0]} rows (10%)\n")


# ==========================================
# 3. COMPUTE BALANCED CLASS WEIGHTS
# ==========================================
print("=== Step 3: Computing Balanced Class Weights to Fix Imbalance ===")
# 根据训练集中各类别的稀缺程度自动计算损失权重，解决前期(1)和糖尿病(2)被忽略的问题
sample_weights_train = compute_sample_weight(class_weight='balanced', y=y_train)


# ==========================================
# 4. XGBOOST MODEL TRAINING
# ==========================================
print("=== Step 4: Training Weighted XGBoost Classifier ===")
xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=150,
    learning_rate=0.1,
    max_depth=6,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# 在训练时传入样本权重
xgb_model.fit(
    X_train, y_train,
    sample_weight=sample_weights_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("Model training completed successfully.\n")


# ==========================================
# 5. MODEL EVALUATION & METRICS GENERATION (前三个核心指标)
# ==========================================
print("=== Step 5: Generating Classification Core Metrics on Test Set ===")
y_pred = xgb_model.predict(X_test)

print("\n--- Core Classification Performance Matrix ---")
# 输出精确率 (Precision)、召回率 (Recall) 和 F1-Score
print(classification_report(
    y_test, 
    y_pred, 
    target_names=['0: No Diabetes', '1: Prediabetes', '2: Diabetes'],
    digits=4  # 保留四位小数，满足学术严谨性
))

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 6. FEATURE IMPORTANCE PERCENTAGE
# ==========================================
print("\n=== Step 6: Extracting Feature Importance Percentages ===")
importances = xgb_model.feature_importances_
importance_percentages = importances * 100

# 转换为易于阅读的 DataFrame 格式并降序排列
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance_Percentage (%)': importance_percentages
}).sort_values(by='Importance_Percentage (%)', ascending=False)

print(feature_importance_df.to_string(index=False))


# ==========================================
# 💾 🔥 核心新增: Step 7: 保存模型到本地
# ==========================================
print("\n=== Step 7: Saving Trained XGBoost Model ===")
model_filename = "xgboost_diabetes_model.joblib"

# 使用 joblib 将模型持久化到本地文件
joblib.dump(xgb_model, model_filename)
print(f"Success: Model has been saved locally as '{model_filename}'")