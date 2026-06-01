import numpy as np

class NearestNeighbor:
    def __init__(self, k=3):
        self.k = k


    def train(self, training_images, training_labels):
        # training_images is a numpy Array
        self.training_images = training_images
        # labels, in this case, are represented as numbers to make computation and comparisons faster
        self.training_labels = training_labels

    def predict(self, test_images):
        # this is getting the number of items to test
        num_test = test_images.shape[0]

        # set the output type to be the same as the input type 
        prediction_array = np.zeros(num_test, dtype=self.training_labels.dtype)
        # at this point, we have an array of length = num_test. This is because, for each item in that prediction array, we will output a label

    
        # for each test row we wish to predict, we will calculate the differences between the image provided and the get the k minimum indexes
        for i in range(num_test):
            distances = np.sum(
                np.abs(self.training_images - test_images[i, :]),
                axis=1
            )

            nearest_indices = np.argpartition(distances, self.k)[:self.k]
            nearest_labels = self.training_labels[nearest_indices]

            prediction_array[i] = np.bincount(nearest_labels).argmax()

        return prediction_array


    
