from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import nltk
nltk.download('punkt')
from flask import current_app
import pickle, re
import collections


class SpamClassifier:

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)


    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        # tk_texts= [[word1 for word1 in tk.split()] for tk in text]
                    
        # cl_text=[[txt.lower() for txt in tk_list if txt.isalpha() and len(txt)>=3]
        #             for tk_list in tk_texts]
        # lw_tk_text=list(zip(cl_text,target))
        # return lw_tk_text
        data = []
        for i in range(len(text)):
            tokens = [word for sent in nltk.sent_tokenize(text[i]) for word in nltk.word_tokenize(sent)]
            tokens = [word.lower() for word in tokens if len(word) > 3]
            data.append((tokens, target[i]))
        return data

    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels

        Return Type is a set
        """
        features = []
        for sent in corpus:
            for j in sent[0]:
                features.append(j)
        u_data = set(features)
        return u_data

    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string

        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        d = {word: word in document for word in self.unique_data}
        return d
        # self.extract_dict = {word: (word in document) for word in self.word_features}
        # return self.extract_dict

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        train_data = []
        self.tokenized_data = self.extract_tokens(text, labels)
        self.unique_data = self.get_features(self.tokenized_data)
        counter = 0
        for i in range(len(text)):
            print('number of completed emails ', counter)
            self.data_extraction = self.extract_features(text[i])
            train_data.append((self.data_extraction, labels[i]))
            counter += 1
        print('Training NB classifier')
        self.clf = nltk.NaiveBayesClassifier.train(train_data)
        return self.clf, self.unique_data

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
        test_sentence = {i: (i in nltk.word_tokenize(text)) for i in self.word_features}
        return self.classifier.classify(test_sentence)


if __name__ == '__main__':

    print('Done')