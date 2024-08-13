from .Huffman_coding import Tree, Node

from collections import Counter
import numpy as np

class Vocabulary:
    """Generates vocabulary and training dataset for CBOW word2vec model"""

    def __init__(self, sentences: list['str'], window: int, min_word: int, subsampling_value = 10**(-3)):
        """
        :param sentences: list of sentences used to generate vocabulary
        :param window: window size used in CBOW
        :param min_word: skip words of less than specified frequency
        :param subsampling_value: punishment for too frequent words
        """
        self.sentences = sentences.copy()
        self.window = window
        self.min_word = min_word
        self.subsampling_value = subsampling_value
        self.dictionary = set()
        self.ordered_dictionary = []
        self.one_hot_encoding = {}
        self.skipped_words = {}
        self.n = 0
        self.train_recipe = {}

        self.prep_data()
        self.huffman_coding = self.create_huffman_coding()
        self.fill_train_recipe()
        return None

    def prep_data(self):
        sentences_too_short = []
        freq_dictionary = []
        
        for sentence in self.sentences:
            words = sentence.lower().split()
                
            if len(words) > 2:
                self.dictionary.update(words)
                freq_dictionary += words
            else:
                sentences_too_short.append(sentence)

        self.sentences = [x for x in self.sentences if x not in sentences_too_short]
        

        freq_counter = Counter(freq_dictionary)
        corups_size = freq_counter.total()
        self.words_prob = dict(zip(list(freq_counter), [max(0, (1 + ((freq_counter[x]/corups_size)/self.subsampling_value)**(1/2))*(self.subsampling_value/(freq_counter[x]/corups_size))**2) for x in freq_counter]))

        self.skipped_words = set([x for x in freq_counter if (freq_counter[x] < self.min_word)])
        self.ordered_dictionary = [x for x in self.dictionary if x not in self.skipped_words]

        new_sentences = []

        for sentence in self.sentences:
            words = sentence.lower().split()
            sent = ''
            for word in words:
                if word not in self.skipped_words:
                    if self.words_prob[word] > np.random.uniform():
                        sent = sent + ' ' + word
                    else:
                        freq_counter[word] -= 1

            if len(sent.split()) > 2:
                new_sentences.append(sent)

        self.sentences = new_sentences

        self.freq_counter = dict([(x, freq_counter[x]) for x in freq_counter if x not in self.skipped_words])

        for i, word in enumerate(self.ordered_dictionary):
            self.one_hot_encoding[word] = i

        return None
    
    
    def create_huffman_coding(self):
        trees_list = [Tree(Node(value = self.freq_counter[k], tag = k, parent = None)) for k in self.freq_counter]

        while len(trees_list) > 1:
            sorted_trees = sorted(trees_list, key = lambda x: x.value)
            glued_tree = sorted_trees[0].glue_trees(sorted_trees[1])
            trees_list = [x for x in trees_list if x not in [sorted_trees[0], sorted_trees[1]]]
            trees_list.append(glued_tree)

        trees_list[0].update_nodes_id()
        return trees_list[0]


    def fill_train_recipe(self):
        s = 0
        for sentence in self.sentences:
            words = sentence.lower().split()
            for i, word in enumerate(words):
                if word not in self.skipped_words:
                    self.train_recipe[s] = {}
                    self.train_recipe[s]['Y'] = self.huffman_coding.get_node_by_tag(word)
                    self.train_recipe[s]['X'] = []

                    l = len(words)
                    for x in range(max(0, i-self.window), i):
                        self.train_recipe[s]['X'].append(self.one_hot_encoding[words[x]])
                    for x in range(i+1, min(i+self.window+1, l)):
                        self.train_recipe[s]['X'].append(self.one_hot_encoding[words[x]])
                    s += 1
        self.n = s
        return None
