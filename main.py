from knearest import NearestNeighbor
import pickle
import numpy as np
import matplotlib.pyplot as plt

image_batches = []
label_batches = []

for batch_num in range(1, 6):
    path = f"images/cifar-10-batches-py/data_batch_{batch_num}"

    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        image_batches.append(batch[b"data"])
        label_batches.append(batch[b"labels"])

training_images = np.vstack(image_batches).astype(np.int32)
training_labels = np.concatenate(label_batches).astype(np.int32)


knn = NearestNeighbor()
knn.train(training_images, training_labels)

with open("images/cifar-10-batches-py/test_batch", "rb") as f:
    batch = pickle.load(f, encoding="bytes")
    test_images = np.array(batch[b"data"]).astype(np.int32)


label_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

actual_labels = np.array(batch[b"labels"])
predictions = knn.predict(test_images[:10].astype(np.int32))

for i, image in enumerate(test_images[:10]):
    image = image.reshape(3, 32, 32).transpose(1, 2, 0)

    plt.imshow(image)
    plt.title(
        f"Predicted: {label_names[predictions[i]]}\n"
        f"Actual: {label_names[actual_labels[i]]}"
    )
    plt.axis("off")
    plt.show()