import matplotlib.pyplot as plt
import pandas as pd

mlp_acc = 0.7742
cnn_acc = 0.7096
rnn_acc = 0.6935
svm_acc = 0.8065   # use best (poly or rbf)

models = ['MLP', 'CNN', 'RNN', 'SVM']
accuracies = [mlp_acc, cnn_acc, rnn_acc, svm_acc]

# =========================
# BAR GRAPH
# =========================
plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)
plt.title("Comparison of ML Models for Heart Disease Prediction")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xlabel("Models")

# Add values on bars
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()

# =========================
# TABLE
# =========================
df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})

print("\nFinal Comparison Table:")
print(df)