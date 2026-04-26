import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("heart_disease_uci.csv")

# Drop ID column if exists
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# -----------------------------
# 2. SPLIT FEATURES & TARGET
# -----------------------------
X = df.drop(columns=['num'])
y = df['num']

# Convert target safely
y = pd.to_numeric(y, errors='coerce')

# Remove invalid rows
valid_idx = y.notna()
X = X[valid_idx]
y = y[valid_idx]

# Binary classification
y = (y > 0).astype(int)

# -----------------------------
# 3. ENCODE CATEGORICAL DATA
# -----------------------------
label_encoders = {}
category_maps = {}

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str).str.strip()

        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

        label_encoders[col] = le
        category_maps[col] = le.classes_.tolist()

# -----------------------------
# 4. HANDLE NUMERIC DATA
# -----------------------------
X = X.apply(pd.to_numeric, errors='coerce')

# Fill missing values with median
X.fillna(X.median(), inplace=True)

# -----------------------------
# 5. FEATURE SCALING
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 6. SAVE FEATURE COLUMNS
# -----------------------------
feature_columns = X.columns.tolist()

# -----------------------------
# 7. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 8. BUILD MODEL
# -----------------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.4),

    Dense(16, activation='relu'),
    Dropout(0.3),

    Dense(8, activation='relu'),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 9. EARLY STOPPING
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# -----------------------------
# 10. TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# 11. EVALUATE MODEL
# -----------------------------
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# -----------------------------
# 12. SAVE EVERYTHING (CRITICAL)
# -----------------------------
model.save("mlp_final_model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# 🔥 NEW (IMPORTANT)
with open("feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

with open("category_maps.pkl", "wb") as f:
    pickle.dump(category_maps, f)

print("✅ Model, scaler, encoders, feature columns, and mappings saved!")