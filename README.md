# K-Nearest Neighbors from Scratch

Implementation of the K-Nearest Neighbors (KNN) algorithm using NumPy.
Based on Stanford CS231N | Spring 2025 | Lecture 2: Image Classification with Linear Classifiers

## Features

- L1 (Manhattan) distance
- Configurable k
- Majority voting
- No external ML libraries

## Usage

```python
knn = NearestNeighbor(k=3)
knn.train(X_train, y_train)
predictions = knn.predict(X_test)