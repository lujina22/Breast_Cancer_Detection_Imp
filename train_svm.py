import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_dataset import build_dataset

# ---------- LOAD DATA ----------
X, y1, y2 = build_dataset("train_dataset.csv")
print(f"Total samples: {len(y1)}, Feature dim: {X.shape[1]}")

# ---------- TRAIN/TEST SPLIT 70/30 ----------
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, train_size=0.7, stratify=y1, random_state=42
)

# ---------- SCALE ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---------- STAGE-1: Normal vs Abnormal (reuse previous mapping) ----------
svm_stage1 = SVC(kernel='rbf', C=1, gamma='scale')
svm_stage1.fit(X_train_scaled, y1_train)
y1_pred = svm_stage1.predict(X_test_scaled)
acc_stage1 = accuracy_score(y1_test, y1_pred)
print(f"Stage-1 Accuracy (Normal vs Abnormal): {acc_stage1*100:.2f}%")

# ---------- STAGE-2: Mass vs MC ----------
train_mask = y2_train != -1
test_mask  = y2_test != -1

X2_train = X_train_scaled[train_mask]
X2_test  = X_test_scaled[test_mask]
y2_train2 = y2_train[train_mask]
y2_test2  = y2_test[test_mask]

svm_stage2 = SVC(kernel='rbf', C=1, gamma='scale')
svm_stage2.fit(X2_train, y2_train2)
y2_pred = svm_stage2.predict(X2_test)

# ---------- MC AND MASS ACCURACY ----------
mc_mask = y2_test2 == 1
mass_mask = y2_test2 == 0

acc_mc = accuracy_score(y2_test2[mc_mask], y2_pred[mc_mask])
acc_mass = accuracy_score(y2_test2[mass_mask], y2_pred[mass_mask])

print(f"MC Accuracy:   {acc_mc*100:.2f}%")
print(f"Mass Accuracy: {acc_mass*100:.2f}%")

# ---------- FINAL AVERAGE ----------
final_acc = (acc_stage1 + acc_mc + acc_mass)/3
print(f"\nFinal Average Accuracy: {final_acc*100:.2f}%")
