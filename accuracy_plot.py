import matplotlib.pyplot as plt
import os

# Absolute project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "visuals", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")

# Example accuracy values (you can change later)
models = ["Naive Bayes", "Logistic Regression", "SVM"]
accuracies = [86, 91, 88]

plt.figure()
plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")

plt.savefig(OUTPUT_FILE)
plt.close()

print("ACCURACY GRAPH SAVED AT:")
print(OUTPUT_FILE)
