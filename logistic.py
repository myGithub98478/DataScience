import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Input features (height, weight)
X = np.array([
    [185, 72],
    [170, 54],
    [168, 60],
    [179, 68],
    [183, 72],
    [188, 77]
])

# Step 2: Labels (0 = short/light, 1 = tall/heavy) — arbitrary for this example
y = np.array([1, 0, 0, 1, 1, 1])

# Step 3: Create logistic regression model and train it
model = LogisticRegression()
model.fit(X, y)

# Step 4: Predict on the training data
predictions = model.predict(X)
print("Predictions:", predictions)

# Step 5: Accuracy
acc = accuracy_score(y, predictions)
print("Accuracy:", acc)

# Step 6: Visualize the decision boundary
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Logistic Regression Decision Boundary")
plt.grid(True)
plt.show()
