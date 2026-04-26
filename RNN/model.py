import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# =========================
# DATA PREPARATION
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

# 🔥 IMPORTANT: Reshape for RNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# =========================
# 1. OVERFITTING MODEL
# =========================
print("Training overfitting RNN...")

model_overfit = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = model_overfit.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# =========================
# 2. EARLY STOPPING MODEL
# =========================
print("Training RNN with early stopping...")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_early = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_early.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_early = model_early.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# 3. DROPOUT MODEL
# =========================
print("Training RNN with dropout...")

model_dropout = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_dropout = model_dropout.fit(
    X_train, y_train,
    epochs=80,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =========================
# 4. BEST MODEL
# =========================
print("Training final RNN model...")

model_best = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_best.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop_best = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history_best = model_best.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop_best],
    verbose=1
)

# =========================
# EVALUATION
# =========================
print("\nEvaluating RNN model...")

test_loss, test_acc = model_best.evaluate(X_test, y_test, verbose=0)
y_pred = (model_best.predict(X_test) > 0.5).astype(int).flatten()

print(f"Final Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

# =========================
# VISUALIZATION (SAME STYLE)
# =========================
fig = plt.figure(figsize=(20, 15))

plt.subplot(3, 3, 1)
plt.plot(history_overfit.history['loss'], label='Overfit Train')
plt.plot(history_overfit.history['val_loss'], label='Overfit Val')
plt.plot(history_best.history['loss'], label='Best Train')
plt.plot(history_best.history['val_loss'], label='Best Val')
plt.title('1. Overfitting vs Regularized (RNN)')
plt.legend()

plt.subplot(3, 3, 2)
plt.plot(history_early.history['loss'], label='Train')
plt.plot(history_early.history['val_loss'], label='Val')
plt.title('2. Early Stopping Effect')
plt.legend()

plt.subplot(3, 3, 3)
plt.plot(history_dropout.history['val_accuracy'], label='Dropout')
plt.plot(history_overfit.history['val_accuracy'], label='No Dropout')
plt.title('3. Dropout Effect')
plt.legend()

plt.subplot(3, 3, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('4. Confusion Matrix')

plt.subplot(3, 3, 5)
plt.plot(history_best.history['accuracy'], label='Train')
plt.plot(history_best.history['val_accuracy'], label='Val')
plt.title('5. Accuracy Curve')
plt.legend()

plt.subplot(3, 3, 6)
plt.plot(history_best.history['loss'], label='Train')
plt.plot(history_best.history['val_loss'], label='Val')
plt.title('6. Loss Curve')
plt.legend()

plt.tight_layout()
plt.savefig('rnn_analysis.png', dpi=300)
plt.show()

# =========================
# SAVE MODEL
# =========================
model_best.save('rnn_model.h5')

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Files saved:")
print("- rnn_model.h5")
print("- rnn_analysis.png")