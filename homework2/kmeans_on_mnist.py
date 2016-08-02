import amitgroup.io.mnist
import matplotlib.pyplot as plt
import matplotlib
from numpy import empty, unique, take, reshape, array
from kmeans import Kmeans

def main():
    '''Train tha model and evaluate performance'''

    '''Load the MNIST training data. Also flatten the images from 28X28 arrays to a single vector'''
    images, labels = amitgroup.io.mnist.load_mnist('training', path='./', asbytes=True)
    images = [image.ravel() for image in images]

    '''Find unique labels and which are the first images that correspnd to them'''
    indices = unique(labels, return_index=True)[1]

    '''Create the clustering engine. Use the unique images found above as centers'''
    clustering = Kmeans()
    clustering.train(data=images, centers=take(images, indices, axis=0), max_iterations=100)

    '''Load the testing data set and flatten the images'''
    test_images, test_labels = amitgroup.io.load_mnist('testing', path='./', asbytes=True)
    test_images = [image.ravel() for image in test_images]

    '''Assign the test data to clusters and evaluate the performance'''
    predictions = [clustering.cluster(image) for image in test_images]
    success = (predictions == test_labels)
    correct, counts = unique(success, return_counts=True)

    print('{} of the testing data set where put in the wrong cluster'.format(counts[0]))

    plot_images_separately([reshape(center, (28,28)) for center in clustering.centers])

def plot_images_separately(images):
    "Plot the MNIST images."
    fig = plt.figure()
    for j in range(1, 11):
        ax = fig.add_subplot(1, 10, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(array([]))
        plt.yticks(array([]))
    plt.savefig('centers.png')

if __name__ == "__main__":
    main()