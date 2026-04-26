import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# =========================
# DATA PREPARATION (Same as MLP)
# =========================
print("Loading and preparing data...")

df = pd.read_csv('heart_disease_uci.csv', header=None)

df.drop(0, axis=1, inplace=True)
df.dropna(subset=[5, 12, 13], inplace=True)

df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
y = (df.iloc[:, -1] > 0).astype(int)

X = df.iloc[:, :-1].copy()

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# =========================
# 1. LINEAR SVM (Paper)
# =========================
print("\nTraining Linear SVM...")

svm_linear = SVC(kernel='linear', C=0.3, probability=True)
svm_linear.fit(X_train, y_train)

y_pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)

# =========================
# 2. POLYNOMIAL SVM (Paper - BEST)
# =========================
print("Training Polynomial SVM...")

svm_poly = SVC(kernel='poly', degree=3, C=0.7, probability=True)
svm_poly.fit(X_train, y_train)

y_pred_poly = svm_poly.predict(X_test)
acc_poly = accuracy_score(y_test, y_pred_poly)

# =========================
# 3. RBF SVM (Improved Model)
# =========================
print("Training RBF SVM...")

svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

# =========================
# RESULTS
# =========================
print("\n===== SVM RESULTS =====")
print(f"Linear SVM Accuracy     : {acc_linear:.4f}")
print(f"Polynomial SVM Accuracy : {acc_poly:.4f}")
print(f"RBF SVM Accuracy        : {acc_rbf:.4f}")

print("\n===== BEST MODEL (Polynomial SVM) REPORT =====")
print(classification_report(y_test, y_pred_poly, target_names=['No Disease', 'Disease']))

# =========================
# CONFUSION MATRICES
# =========================
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_linear), annot=True, fmt='d', cmap='Blues')
plt.title("Linear SVM")

plt.subplot(1, 3, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_poly), annot=True, fmt='d', cmap='Blues')
plt.title("Polynomial SVM")

plt.subplot(1, 3, 3)
sns.heatmap(confusion_matrix(y_test, y_pred_rbf), annot=True, fmt='d', cmap='Blues')
plt.title("RBF SVM")

plt.tight_layout()
plt.show()

# =========================
# ACCURACY COMPARISON PLOT
# =========================
models = ['Linear', 'Polynomial', 'RBF']
accuracies = [acc_linear, acc_poly, acc_rbf]

plt.figure(figsize=(6, 4))
plt.bar(models, accuracies)
plt.title("SVM Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# =========================
# SAVE BEST MODEL (Polynomial)
# =========================
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_poly, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\nFiles saved:")
print("- svm_model.pkl (Polynomial SVM)")
print("- scaler.pkl")
print("- label_encoders.pkl")