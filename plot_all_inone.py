import matplotlib.pyplot as plt

model_names = [
    "GoogLeNet + SVM",
    "GoogLeNet + KNN",
    "GoogLeNet (CNN)",
    "AlexNet + SVM",
    "AlexNet + KNN"
]

accuracies = [
    0.8241,   # these are the validation accuracy obtained from training of the models
    0.7752,
    0.88281,
    0.8352,
    0.81647
]

plt.figure(figsize=(10,6))

bars = plt.bar(model_names, accuracies)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

plt.ylim(0,1)
plt.ylabel("Validation Accuracy")
plt.title("Comparison of Different Models")

plt.xticks(rotation=25)
plt.grid(axis='y')

plt.show()