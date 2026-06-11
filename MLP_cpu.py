import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保深度学习的权重初始化和随机打乱完全可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. LOAD DATASET & PREPARE VARIABLES
# ==========================================
print("=== Step 1: Loading Dataset ===")
file_path = "diabetes_012_health_indicators_BRFSS2015.csv"
df = pd.read_csv(file_path)

y = df.iloc[:, 0].astype(int).values
X = df.iloc[:, 1:].values

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n")


# ==========================================
# 2. DATASET PARTITION (7:2:1 RATIO, WITH RANDOM SEED)
# ==========================================
print("=== Step 2: Stratified & Randomized Data Partitioning (7:2:1) ===")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp
)

# 💡 特征标准化 (Neural Networks 对特征尺度极其敏感)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ==========================================
# 3. COMPUTE CLASS WEIGHTS FOR LOSS FUNCTION
# ==========================================
print("=== Step 3: Computing Class Weights for Loss Function ===")
class_counts = np.bincount(y_train)
total_samples = len(y_train)
class_weights = total_samples / (len(class_counts) * class_counts)
class_weights_tensor = torch.FloatTensor(class_weights)
print(f"Calculated Class Weights: {class_weights}\n")


# ==========================================
# 4. PREPARE PYTORCH DATALOADER
# ==========================================
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 5. DEFINE RESIDUAL MLP NETWORK ARCHITECTURE
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DiabetesDeepPredictor(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super(DiabetesDeepPredictor, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.res_block = ResidualBlock(128)
        
        self.output_layer = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        out = self.input_layer(x)
        out = self.res_block(out)
        out = self.output_layer(out)
        return out

model = DiabetesDeepPredictor(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


# ==========================================
# 6. MODEL TRAINING WITH VALIDATION LOOP
# ==========================================
print("=== Step 4: Training Residual MLP Deep Learning Model ===")
EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

print("Deep Learning model training completed successfully.\n")


# ==========================================
# 7. UNIFIED CLASSIFICATION PERFORMANCE AUDIT
# ==========================================
print("=== Step 5: Generating Classification Core Metrics on Test Set ===")
model.eval()
all_preds = []

with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())

y_pred = np.array(all_preds)

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
# 8. SAVE TRAINED MODEL
# ==========================================
print("\n=== Step 6: Saving Trained PyTorch Model ===")
model_filename = "pytorch_diabetes_mlp_model.pth"
torch.save(model.state_dict(), model_filename)
print(f"Success: Deep Learning model weights saved locally as '{model_filename}'")