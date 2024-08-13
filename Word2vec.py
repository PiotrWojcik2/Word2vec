from .Vocabulary import Vocabulary
from .Utilities import sigmoid

import numpy as np

class Word2vec:
    """CBOW implementation of Word2vec algorithm with hierarchical softmax optimization"""
    
    def __init__(self, vocabulary: Vocabulary, dim: int):
        """
        :param vocabulary: object of class Vocabulary used in training
        :param dim: Dimensionality of words reduction  
        """

        self.vocabulary = vocabulary
        self.input_size = len(vocabulary.ordered_dictionary)
        self.dim = dim
        self.weights = []
        self.weights.append(np.random.normal(loc = 0, scale = 0.05, size = (self.dim, self.input_size)).astype(np.float32))
        self.weights.append(np.random.normal(loc = 0, scale = 0.05, size = (len(self.vocabulary.huffman_coding.nodes), self.dim)).astype(np.float32))
        self.bias = np.random.normal(loc = 0, scale = 0.05, size = (len(self.vocabulary.huffman_coding.nodes), 1)).astype(np.float32)


    def train(self, learning_rate = 0.01, epochs = 10):
        """
        Train word2vec model iterating over each word in vocabulary for a fixed number of epochs

        :param learning_rate: coefficient to scale gradients in SGD
        :param epochs: number of iterations over the etire dataset
        """

        for epoch in range(epochs):
            loss = 0

            for i in self.vocabulary.train_recipe:
                y_node = self.vocabulary.train_recipe[i]['Y']
                x_ids = self.vocabulary.train_recipe[i]['X']

                y_ids = self.vocabulary.huffman_coding.get_nodes_ids_to_node(y_node)

                forward_steps = []
                forward_steps.append(y_node.code)
                forward_steps.append(np.ones(len(x_ids)))
                forward_steps.append((self.weights[0][:,x_ids]).dot(forward_steps[-1]))
                forward_steps.append(sigmoid(forward_steps[0]*((self.weights[1][y_ids,:]).dot(forward_steps[-1]) + self.bias[y_ids].T)))
                forward_steps.append(np.prod(forward_steps[-1]))

                loss -= np.log(forward_steps[-1])

                dL = (forward_steps[0]*(forward_steps[-2] - 1))
                dLdb = dL.T
                dLdW1 = dL.T.dot(forward_steps[-3].reshape(forward_steps[-3].shape[0], 1).T)
                dLda2 = dL.dot(self.weights[1][y_ids,:])
                dLdW0 = dLda2.T.dot(forward_steps[-4].reshape(forward_steps[-4].shape[0], 1).T)

                self.weights[0][:,x_ids] -= (learning_rate)*dLdW0
                self.weights[1][y_ids,:] -= (learning_rate)*dLdW1
                self.bias[y_ids,:] -= (learning_rate)*dLdb

            print('Epoch: ', epoch+1, 'loss: ', loss)
                    
        return None

    def find_closest(self, word, n):
        """Find closest n elements to a specified word using cosine similarity"""

        word_vec = self.weights[0].T[self.vocabulary.one_hot_encoding[word]]
        distances = np.inner(word_vec, self.weights[0].T)/(np.linalg.norm(word_vec) * np.linalg.norm(self.weights[0].T, axis = 1))
        idx = np.argsort(distances)[::-1]
        words = [self.vocabulary.ordered_dictionary[i] for i in idx[:n]]
        ret = dict(zip(words, np.sort(distances)[::-1][:n]))

        return ret