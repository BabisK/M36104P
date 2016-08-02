from numpy.linalg import norm
from numpy import mean, empty, asarray, unique
from scipy.spatial.distance import euclidean

class Kmeans:
    centers = None

    def train(self, data, centers, max_iterations):
        '''Trains the k-means clustering engine
        The number of clusters to create is the same as the number of initial centers that are passed to the algorithm
        '''

        '''Initialize data'''
        self.centers = centers
        iterations = 0
        oldcenters = empty(self.centers.shape)

        '''As long as the centers change and the maximum iterations is not reached'''
        while not ((oldcenters == self.centers).all() or iterations >= max_iterations):
            #Store previous centers
            oldcenters = self.centers
            iterations += 1

            print('Iteration {}'.format(iterations))

            print('Computing labels for data')
            '''For each row in the data get the closest cluster center'''
            labels = [self._get_label(row) for row in data]
            unique_labels, counts = unique(labels, return_counts=True)
            print('Data assignments: {}: {}'.format(unique_labels, counts))

            print('Adjusting centers of clusters')
            '''For each label get the data that are assigned to it and caclulate the mean of this cluster'''
            self.centers = asarray([mean([x for index2, x in enumerate(data) if labels[index2] == index], axis=0) for index, center in enumerate(self.centers)])

        print('Training complete!')

    def cluster(self, data):
        '''Predicts the cluster where this row belongs to'''
        return self._get_label(data)

    def _get_label(self, data):
        '''Returns '''
        distances = [euclidean(data, center) for center in self.centers]
        return distances.index(min(distances))
