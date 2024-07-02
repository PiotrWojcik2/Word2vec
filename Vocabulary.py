from collections import Counter

class Vocabulary:
    """Generates vocabulary and training dataset for CBOW word2vec model"""

    def __init__(self, sentences: list['str'], window: int, min_word: int):
        """
        :param sentences: list of sentences used to generate vocabulary
        :param window: window size used in CBOW
        :param min_word: skip words of less than specified frequency
        """
        self.sentences = sentences.copy()
        self.window = window
        self.min_word = min_word
        self.dictionary = set()
        self.ordered_dictionary = []
        self.one_hot_encoding = {}
        self.skipped_words = {}
        self.n = 0
        self.train_recipe = {}

        self.prep_data()
        self.fill_train_recipe()
        return None

    def prep_data(self):
        sentences_too_short = []
        freq_dictionary = []
        train_approx_len = 0
        
        for sentence in self.sentences:
            words = sentence.split()
                
            if len(words) > 2*self.window+1:
                train_approx_len += len(words)-(2*self.window)
                self.dictionary.update(words)
                freq_dictionary += words
            else:
                sentences_too_short.append(sentence)

        self.sentences = [x for x in self.sentences if x not in sentences_too_short]
        

        freq_counter = Counter(freq_dictionary)
        self.skipped_words = set([x for x in freq_counter if (freq_counter[x] < self.min_word or len(x) < 3)])
        self.ordered_dictionary = list(self.dictionary)

        for i, word in enumerate(self.ordered_dictionary):
            self.one_hot_encoding[word] = i

        return None

    def fill_train_recipe(self):
        s = 0
        for sentence in self.sentences:
            words = sentence.split()
            for i, word in enumerate(words[self.window:-self.window]):
                if word not in self.skipped_words:
                    self.train_recipe[s] = {}
                    self.train_recipe[s]['Y'] = self.one_hot_encoding[word]
                    self.train_recipe[s]['X'] = []
                    for p in range(self.window):
                        self.train_recipe[s]['X'].append(self.one_hot_encoding[words[i-(p+1)]])
                    s += 1
        self.n = s
        return None
