import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv('heart_disease_uci.csv', header=None)
df.drop(0, axis=1, inplace=True)
df.dropna(subset=[5, 12, 13], inplace=True)

df.iloc[:, -1] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
y = (df.iloc[:, -1] > 0).astype(int)
X = df.iloc[:, :-1].copy()

# Encode categorical
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# ==============================
# 2. NORMALIZATION COMPARISON
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_raw = X.values.reshape(X.shape[0], X.shape[1], 1)
X_norm = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Train-test split
X_train_n, X_test_n, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y)

X_train_r, X_test_r, _, _ = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)

print("Iterations per epoch:", len(X_train_n)//32)

# ==============================
# 3. OVERFITTING MODEL
# ==============================
model_overfit = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_overfit.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_overfit = model_overfit.fit(
    X_train_n, y_train, epochs=100, batch_size=16,
    validation_split=0.2, verbose=0)

# ==============================
# 4. DROPOUT MODEL
# ==============================
model_dropout = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_dropout = model_dropout.fit(
    X_train_n, y_train, epochs=100, batch_size=32,
    validation_split=0.2, verbose=0)

# ==============================
# 5. EARLY STOPPING MODEL
# ==============================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model_early = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_early.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_early = model_early.fit(
    X_train_n, y_train, epochs=100, batch_size=32,
    validation_split=0.2, callbacks=[early_stop], verbose=0)

# ==============================
# 6. NORMALIZATION EFFECT
# ==============================
model_norm = Sequential([
    Conv1D(32, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_norm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_norm = model_norm.fit(X_train_n, y_train, epochs=50, verbose=0)
history_raw = model_norm.fit(X_train_r, y_train, epochs=50, verbose=0)

# ==============================
# 7. LEARNING RATE TEST
# ==============================
learning_rates = [0.001, 0.01, 0.1]
lr_results = {}

for lr in learning_rates:
    model = Sequential([
        Conv1D(32, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_n, y_train, epochs=30, verbose=0)
    lr_results[lr] = history.history['accuracy'][-1]

# ==============================
# 8. BATCH SIZE TEST
# ==============================
batch_sizes = [8, 32, 64]
batch_results = {}

for bs in batch_sizes:
    model = Sequential([
        Conv1D(32, 2, activation='relu', input_shape=(X_train_n.shape[1], 1)),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_n, y_train, epochs=30, batch_size=bs, verbose=0)
    batch_results[bs] = history.history['accuracy'][-1]

# ==============================
# 9. FINAL MODEL (BEST)
# ==============================
model_final = model_dropout
test_loss, test_acc = model_final.evaluate(X_test_n, y_test, verbose=0)

y_pred = (model_final.predict(X_test_n) > 0.5).astype(int)

print("\nFinal Test Accuracy:", test_acc)
print(classification_report(y_test, y_pred))

# ==============================
# 10. ALL GRAPHS (9 PLOTS)
# ==============================
plt.figure(figsize=(20,15))

# 1 Overfitting
plt.subplot(3,3,1)
plt.plot(history_overfit.history['accuracy'], label='Train')
plt.plot(history_overfit.history['val_accuracy'], label='Validation')
plt.title("Overfitting")
plt.legend()

# 2 Early Stopping
plt.subplot(3,3,2)
plt.plot(history_early.history['loss'], label='Loss')
plt.title("Early Stopping")
plt.legend()

# 3 Dropout
plt.subplot(3,3,3)
plt.plot(history_dropout.history['val_accuracy'], label='Dropout')
plt.plot(history_overfit.history['val_accuracy'], label='No Dropout')
plt.title("Dropout Effect")
plt.legend()

# 4 Normalization
plt.subplot(3,3,4)
plt.plot(history_norm.history['accuracy'], label='Normalized')
plt.plot(history_raw.history['accuracy'], label='Raw')
plt.title("Normalization Effect")
plt.legend()

# 5 Learning Rate
plt.subplot(3,3,5)
plt.bar(list(lr_results.keys()), list(lr_results.values()))
plt.title("Learning Rate")

# 6 Batch Size
plt.subplot(3,3,6)
plt.bar(list(batch_results.keys()), list(batch_results.values()))
plt.title("Batch Size")

# 7 Accuracy Curve
plt.subplot(3,3,7)
plt.plot(history_dropout.history['accuracy'], label='Train')
plt.plot(history_dropout.history['val_accuracy'], label='Val')
plt.title("Epoch vs Accuracy")
plt.legend()

# 8 Loss Curve
plt.subplot(3,3,8)
plt.plot(history_dropout.history['loss'], label='Train')
plt.plot(history_dropout.history['val_loss'], label='Val')
plt.title("Epoch vs Loss")
plt.legend()

# 9 Confusion Matrix
plt.subplot(3,3,9)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()