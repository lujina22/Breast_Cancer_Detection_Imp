import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_dataset import build_dataset

# ---------- LOAD DATA ----------
X, y1, y2, y3 = build_dataset("train_dataset.csv")
print(f"Samples: {len(y1)}, Feature dim: {X.shape[1]}")

# ---------- SPLIT 70/30 ----------
X_tr, X_te, y1_tr, y1_te, y2_tr, y2_te, y3_tr, y3_te = train_test_split(
    X, y1, y2, y3,
    train_size=0.7,
    stratify=y1,
    random_state=42
)

# ---------- SCALE ----------
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)
X_te = scaler.transform(X_te)

# ===============================
# STAGE-1: Normal vs Abnormal
# ===============================
svm1 = SVC(kernel="rbf", C=1, gamma="scale")
svm1.fit(X_tr, y1_tr)
y1_pred = svm1.predict(X_te)
acc_stage1 = accuracy_score(y1_te, y1_pred)

print(f"Stage-1 Accuracy (Normal vs Abnormal): {acc_stage1*100:.2f}%")

# ===============================
# STAGE-2: Mass vs MC
# ===============================
mask_tr2 = y2_tr != -1
mask_te2 = y2_te != -1

svm2 = SVC(kernel="rbf", C=1, gamma="scale")
svm2.fit(X_tr[mask_tr2], y2_tr[mask_tr2])

y2_pred = svm2.predict(X_te[mask_te2])
y2_true = y2_te[mask_te2]

acc_mc = accuracy_score(y2_true[y2_true == 1], y2_pred[y2_true == 1])
acc_mass = accuracy_score(y2_true[y2_true == 0], y2_pred[y2_true == 0])

print(f"MC Accuracy:   {acc_mc*100:.2f}%")
print(f"Mass Accuracy:{acc_mass*100:.2f}%")

# ===============================
# STAGE-3: Benign vs Malignant
# ===============================
mask_tr3 = y3_tr != -1
mask_te3 = y3_te != -1

svm3 = SVC(kernel="rbf", C=1, gamma="scale")
svm3.fit(X_tr[mask_tr3], y3_tr[mask_tr3])

y3_pred = svm3.predict(X_te[mask_te3])
acc_stage3 = accuracy_score(y3_te[mask_te3], y3_pred)

print(f"Stage-3 Accuracy (Benign vs Malignant): {acc_stage3*100:.2f}%")

# ---------- FINAL ----------
final_acc = (acc_stage1 + acc_mc + acc_mass) / 3
print(f"\nFINAL AVERAGE ACCURACY: {final_acc*100:.2f}%")
