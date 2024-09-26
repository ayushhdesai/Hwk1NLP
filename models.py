# models.py

import random
from sentiment_classifier import evaluate
from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter
import re

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # A bit for data cleaning, by lowercasing and removing non alphabetic words
        # Took help from https://stackoverflow.com/questions/33127900/can-the-a-za-z-python-regex-pattern-be-made-to-match-and-replace-non-ascii-uni
        x = [re.sub(r'[^a-zA-Z]', '', w.lower()) for w in sentence if w.isalpha()]
        # Counter dictionary intialisedto count the frequency of the word 
        f = Counter()
        # Loop through words for indexing and counts
        for w in x:
            wi = self.indexer.add_and_get_index(f"Unigram={w}", add_to_indexer)
            if wi != -1: # if doesn't exist then it would be -1, so we want to move forward with the one existing
                f[wi] += 1

        return f

    def get_indexer(self):
        return self.indexer


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, w, f: UnigramFeatureExtractor):
        self.w = w
        self.f = f

    def top_features(self, n=10):
        top_positive_indices = np.argsort(self.w)[-n:]
        top_negative_indices = np.argsort(self.w)[:n]
        top_positive_features = [(self.f.indexer.get_object(i), self.w[i]) for i in top_positive_indices]
        top_negative_features = [(self.f.indexer.get_object(i), self.w[i]) for i in top_negative_indices]
        
        return top_positive_features, top_negative_features

    # extract and predict
    def predict(self, sentence: List[str]) -> int:
        fs = self.f.get_features(sentence, add_to_indexer=False)
        s = sum(self.w[i] * v for i, v in fs.items())
        return 1 if s > 0 else 0


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, w, f: UnigramFeatureExtractor):
        self.w = w
        self.f = f

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # extract, apply sigmoid and predict
    def predict(self, sentence: List[str]) -> int:
        fs = self.f.get_features(sentence, add_to_indexer=False)
        s = sum(self.w[i] * v for i, v in fs.items())
        p = self.sigmoid(s)
        return 1 if p >= 0.5 else 0 # because of the curve, check at 0.5
    
class LogisticRegressionClassifierStep(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, w, f: UnigramFeatureExtractor):
        self.w = w
        self.f = f

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # extract, apply sigmoid and predict
    def predict(self, sentence: List[str]) -> int:
        fs = self.f.get_features(sentence, add_to_indexer=False)
        s = sum(self.w[i] * v for i, v in fs.items())
        p = self.sigmoid(s)
        return 1 if p >= 0.5 else 0 # because of the curve, check at 0.5


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    n_epochs = 12
    i = feat_extractor.get_indexer()
    weights = None
    lr = 0.67

    # Looping through each epoch, randomizing training examples, adjusting the learning rate, extracting features, 
    # initializing/resizing weights if required, making predictions, and updating weights when predictions are incorrect
    for e in range(n_epochs):
        random.shuffle(train_exs) # randomising at every epoch
        lr = 1.0 / (e + 1) 
        
        for x in train_exs:
            f = feat_extractor.get_features(x.words, add_to_indexer=True)
            
            if weights is None:
                weights = np.zeros(len(i))
            
            if len(weights) < len(i):
                nw = np.zeros(len(i))
                nw[:len(weights)] = weights 
                weights = nw
            
            s = sum(weights[index] * value for index, value in f.items())
            pred = 1 if s > 0 else 0
            
            if pred != x.label:
                for index, value in f.items():
                    weights[index] += lr * value * (1 if x.label == 1 else -1)
    
    return PerceptronClassifier(weights, feat_extractor)

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    n_epochs = 10
    i = feat_extractor.get_indexer()
    lr = 0.1
    weights = None

    # Looping through each epoch, randomizing training examples, extracting features, initializing/resizing weights if required,
    # calculating prediction probabilities using sigmoid, and updating weights using gradient descent
    for e in range(n_epochs):
        random.shuffle(train_exs)
        
        for x in train_exs:
            f = feat_extractor.get_features(x.words, add_to_indexer=True)

            if weights is None:
                weights = np.zeros(len(i))

            if len(weights) < len(i):
                nw = np.zeros(len(i))
                nw[:len(weights)] = weights 
                weights = nw

            s = sum(weights[index] * value for index, value in f.items())
            
            pred_prob = 1 / (1 + np.exp(-s))

            err = x.label - pred_prob
            for index, value in f.items():
                weights[index] += lr * err * value 
    
    return LogisticRegressionClassifier(weights, feat_extractor)

def train_logistic_regression_step(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, dev_exs: List[SentimentExample], step_size: float) -> LogisticRegressionClassifierStep:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    n_epochs = 12
    i = feat_extractor.get_indexer()
    #lr = 0.1
    weights = None

    l_likelihood = []
    d_accuracies = []

    # Looping through each epoch, randomizing training examples, extracting features, initializing/resizing weights if required,
    # calculating prediction probabilities using sigmoid, and updating weights using gradient descent
    for e in range(n_epochs):
        random.shuffle(train_exs)
        tl_likelihood = 0.0
        
        for x in train_exs:
            f = feat_extractor.get_features(x.words, add_to_indexer=True)

            if weights is None:
                weights = np.zeros(len(i))

            if len(weights) < len(i):
                nw = np.zeros(len(i))
                nw[:len(weights)] = weights 
                weights = nw

            s = sum(weights[index] * value for index, value in f.items())
            
            pred_prob = 1 / (1 + np.exp(-s))

            if x.label == 1:
                tl_likelihood += np.log(pred_prob)
            else:
                tl_likelihood += np.log(1 - pred_prob)

            err = x.label - pred_prob
            for index, value in f.items():
                weights[index] += step_size * err * value 

        l_likelihood.append(tl_likelihood)
        d_accuracy, _ = evaluate(LogisticRegressionClassifier(weights, feat_extractor), dev_exs)
        d_accuracies.append(d_accuracy)
        
    return LogisticRegressionClassifierStep(weights, feat_extractor), l_likelihood, d_accuracies


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    elif args.model == "LRS":
        model, _, _ = train_logistic_regression_step(train_exs, feat_extractor, dev_exs, 0.1)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model