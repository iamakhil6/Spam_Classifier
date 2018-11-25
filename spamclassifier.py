from sklearn.metrics import classification_report
from nltk import NaiveBayesClassifier
import pandas as pd
import pickle
import nltk
# nltk.download('punkt')
from sklearn.model_selection import train_test_split



class SpamClassifier:
    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text,
        label)
        parameters:
        text: array of texts
        target: array of target labels
        NOTE: consider only those words which have all alphabets and atleast 3
        characters.
        """
        # self.text
        # print('Text values',text)
        data = []
        for i in range(len(text)):
            tokens = [word for sent in nltk.sent_tokenize(text[i]) for word in nltk.word_tokenize(sent)]
            tokens = [word.lower() for word in tokens if len(word) > 3]
            data.append((tokens, target[i]))
        # print('Inside Extract_ tokens', data, len(data))
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
        unique_data = set(features)
        return unique_data

    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string
        Return type : A dictionary with keys being the train data set word features.
        The values correspond to True or False
        """
        d = {}

        for word in self.unique_data:
            if (word in document):
                d[word] = True
            else:
                d[word] = False
        print('Values in dict ', d)

        return d

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        train_data = []
        self.tokenized_data = self.extract_tokens(text, labels)
        self.unique_data = self.get_features(self.tokenized_data)
        counter = 0
        for i in range(len(self.tokenized_data)):
            print('counter ', counter)
            self.data_extraction = self.extract_features(self.tokenized_data[i][0])
            train_data.append((self.data_extraction, (self.tokenized_data[i][1])))
            counter += 1
        print('Training the classifier')
        self.clf = nltk.NaiveBayesClassifier.train(train_data)
        return self.clf, self.unique_data

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        Allowed Text can be a simple string i.e one input email, a list of emails, or a
        dictionary of emails identified by their labels.
        """
        # 1. Simple string
        # with open(model_name, 'rb') as model:
        #     classifier = pickle.load(model)
        # with open(model_word_features_name, 'rb') as features:
        #     unique_data = pickle.load(features)
        # test_sentence = {i: (i in nltk.word_tokenize(test_X[0])) for i in unique_data}
        # prediction = classifier.classify(test_sentence)

        # 2. List of emails
        with open(model_name, 'rb') as model:
            classifier = pickle.load(model)
        with open(model_word_features_name, 'rb') as features:
            unique_data = pickle.load(features)
        prediction = []
        for email in text:
            test_sentence = {i: (i in nltk.word_tokenize(email)) for i in unique_data}
            prediction.append(classifier.classify(test_sentence))
        return prediction


if __name__ == '__main__':

     data = pd.read_csv('emails.csv')
     train_X, test_X, train_Y, test_Y = train_test_split(data["text"].values, data["spam"].values, test_size = 0.25,
                                                         random_state = 50, shuffle=True, stratify=data["spam"].values)
     classifier = SpamClassifier()
     # classifier_model, model_word_features = classifier.train(train_X, train_Y)
     model_name = 'spam_classifier_model.pk'
     model_word_features_name = 'spam_classifier_model_word_features.pk'
     # print('Pickling model')
     # with open(model_name, 'wb') as model_fp:
     #    pickle.dump(classifier_model, model_fp)
     # print('pickling words fetures')
     # with open(model_word_features_name, 'wb') as model_fp:
     #    pickle.dump(model_word_features, model_fp)
     # print('unique data ', len(classifier.unique_data))
     predicted = classifier.predict(test_X[:10])
     print(predicted, test_Y[:10])
     count = 0
     for i in range(len(predicted)):
         if (predicted[i]== test_Y[i]):
             count+=1
     print('Accuracy ', count/len(predicted)*100)

