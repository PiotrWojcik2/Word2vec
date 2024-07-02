from Vocabulary import Vocabulary
import Utilities

import numpy as np
import time
import os

from sys import platform

in_notebook = Utilities.in_notebook()
if in_notebook:
    from IPython.display import clear_output

class Word2vec:
    """Naive implementation of Word2vec algorithm"""
    
    def __init__(self, vocabulary: Vocabulary, optimizer, dim: int):
        """
        :param vocabulary: object of class Vocabulary used in training
        :param optimizer: object of optimizer type class used to minimizing objective function
        :param dim: Dimensionality of words reduction  
        """

        self.vocabulary = vocabulary
        self.optimizer = optimizer
        self.input_size = len(vocabulary.ordered_dictionary)
        self.dim = dim
        self.weights = []
        self.weights.append(np.random.normal(loc = 0, scale = 0.05, size = (self.dim, self.input_size)).astype(np.float32))
        self.weights.append(np.random.normal(loc = 0, scale = 0.05, size = (self.input_size, self.dim)).astype(np.float32))

    def forward_pass(self, x):
        forward_steps = []
        forward_steps.append(x.T)
        forward_steps.append(self.weights[0].dot(forward_steps[-1]))
        forward_steps.append(Utilities.softmax(self.weights[1].dot(forward_steps[-1]), axis = 0))
        return forward_steps

    def backward_pass(self, y_true, forward_steps):
        weights_gradients = []
        dLda3 = (forward_steps[-1] - y_true.T).T.reshape(y_true.shape[0], forward_steps[-1].shape[0], 1)

        weights_gradients.append(np.tensordot(forward_steps[0].T, np.tensordot(dLda3, self.weights[1].T, axes = (1,1)).swapaxes(1,2), axes = (0,0)).T[0]) 
        weights_gradients.append(np.tensordot(forward_steps[1].T, dLda3, axes = (0,0)).T[0]) 
        
        return weights_gradients

    def stochastic_gradient(self, X, Y):

        forward_steps = self.forward_pass(X)
        weights_gradients = self.backward_pass(Y, forward_steps)
        return weights_gradients

    def train(self, epochs = 10, batch_size = 200, verbose = 100):
        """
        Train word2vec model iterating over training set using mini batches for a fixed number of epochs

        :param epochs: number of iterations over the etire dataset
        :param batch_size: size of mini batches used to calculate gradient
        :param verbose: every how many batches log progress
        """
        n = self.vocabulary.n
        batches = n//batch_size
        indices = list(range(n))

        X_batch = np.zeros((batch_size, self.input_size), dtype=bool)
        Y_batch = np.zeros_like(X_batch)
        
        for epoch in range(epochs):
                
            start_time = time.time()
            np.random.shuffle(indices)
            
            for batch in range(batches):
                idx = indices[(batch*batch_size):((batch+1)*batch_size)]
                
                for i, id in enumerate(idx):
                    Y_batch[i, self.vocabulary.train_recipe[id]['Y']] = 1
                    X_batch[i, self.vocabulary.train_recipe[id]['X']] = 1
            
                weights_gradients = self.stochastic_gradient(X_batch, Y_batch)

                grads_updates = self.optimizer.evaluate(weights_gradients, batch_size)
                
                self.weights[0] -= grads_updates[0]
                self.weights[1] -= grads_updates[1]

                if batch % verbose == 0:
                    bce = np.around(Utilities.binaryCrossEntropy(Y_batch, self.predict(X_batch)), decimals = 3)
                    if in_notebook:
                        clear_output(wait=True)
                    elif platform == 'win32':
                        os.system('cls')
                    elif platform == "linux" or platform == "linux2" or platform == 'darwin':
                        os.system('clear')

                    print('Epoch:',epoch+1,'/',epochs)
                    print('Batch:',batch+1,'/',batches)
                    print('binaryCrossEntropy:', bce)
                    print('time elapsed:', time.time()  - start_time)
                    
                X_batch.fill(0)
                Y_batch.fill(0)
                    
        return None

    def predict(self, X):
        forward_step = X.T
        forward_step = self.weights[0].dot(forward_step)
        forward_step = Utilities.softmax(self.weights[1].dot(forward_step), axis = 0)
        return forward_step.T

    def find_closest(self, word, n):
        """Find closest n elements to a specified word using cosine similarity"""

        word_vec = self.weights[0].T[self.vocabulary.one_hot_encoding[word]]
        distances = np.inner(word_vec, self.weights[0].T)/(np.linalg.norm(word_vec) * np.linalg.norm(self.weights[0].T, axis = 1))
        idx = np.argsort(distances)[::-1]
        words = [self.vocabulary.ordered_dictionary[i] for i in idx[:n]]
        ret = dict(zip(words, np.sort(distances)[::-1][:n]))
        return ret