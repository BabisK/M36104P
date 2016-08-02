# Design
We opted to go away with randomly selecting the initial centers from within the algorithm. The algorithm expects in its
input a list with centers to begin. We chose to start the algorithm with one image of each digit, that is 10 clusters.

Needed libraries are in requirements.txt, to be loaded with pip

Execute the kmeans_on_mnist.py file to run the algorithm on the MNIST data

python ./kmeans_on_mnist.py

#Results

The results of the clustering as executed here, are really bad. About 50% of test data are put in the wrong cluster.
This is to be expected with k-means on this dataset. The distances are not far enough between digits and many digits
overlap (eg 1 with 7, 9 with 8). During the execution we observer that some clusters have much more data assigned to
them than others.

We can se the centers visualized in the file centers.png after the execution

We observe that some centers have moved far enough from their starting positions to represent different digits (8 has
moved in place of 3,). Other centers have mixed to become amalgams of digits (7 and 9, 8 and 5)