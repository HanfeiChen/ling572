#!/usr/bin/env python3

from collections import Counter, defaultdict
from io import UnsupportedOperation
import sys
from typing import Any, Callable, Dict, IO, List, Set, Tuple
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np


def cosine_similarity(a, b) -> float:
    return 1. - distance.cosine(a, b)


def negative_euclidean_distance(a, b) -> float:
    return - distance.euclidean(a, b)


def load_data(file_path: str) -> Tuple:
    x, y = [], []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            if len(line.strip()) > 0:
                parts = line.split()
                label, feature_strings = parts[0], parts[1:]
                features = Counter()
                for feat_str in feature_strings:
                    feat, val = feat_str.split(':')
                    features[feat] = int(val)
                x.append(features)
                y.append(label)
    return x, y


def accuracy(y: List[Any], pred: List[Any]) -> float:
    correct = 0
    for yi, pi in zip(y, pred):
        if yi == pi:
            correct += 1
    return correct / len(y)


def dump_confusion_matrix(confusion_matrix: Dict[Tuple[str, str], int],
                          labels: Set[str],
                          file: IO = sys.stdout) -> None:
    print(' ' * 13, end='', file=file)
    for pred_label in sorted(labels):
        print('', pred_label, end='', file=file)
    print(file=file)
    for truth_label in sorted(labels):
        print(truth_label, end='', file=file)
        for pred_label in sorted(labels):
            print('', confusion_matrix[truth_label, pred_label], end='', file=file)
        print(file=file)


def dump_sys_output(prob_list: List[Dict[str, float]], file: IO = sys.stdout) -> None:
    for idx, probs in enumerate(prob_list):
        print(f'array:{idx}', end='', file=file)
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            print('', label, prob, sep=' ', end='', file=file)
        print(file=file)


class KNN:
    def __init__(self, k_val: int = 5, similarity_func: Callable = cosine_similarity):
        self.k_val = k_val
        self.similarity_func = similarity_func

    def train(self, x: List[Dict[str, float]], y: List[str]) -> None:
        self.feat_set = set()
        for xi in x:
            for feat in xi.keys():
                self.feat_set.add(feat)
        self.num_feat = len(self.feat_set)
        self.feat_to_index = defaultdict(lambda: -1)
        for idx, feat in enumerate(self.feat_set):
            self.feat_to_index[feat] = idx
        self.x = []
        for xi in x:
            self.x.append(self._process_features(xi))
        self.label_set = set(y)
        self.y = y

    def _process_features(self, xi: Dict[str, float]) -> List[float]:
        normalized_xi = [0.] * self.num_feat
        for feat, value in xi.items():
            feat_idx = self.feat_to_index[feat]
            if feat_idx >= 0:
                normalized_xi[feat_idx] = value
        return np.array(normalized_xi)

    def predict_probabilities(self, x: List[Dict[str, int]]) -> List[Dict[str, float]]:
        return [self.predict_probability(xi) for xi in tqdm(x)]

    def predict_probability(self, xi: Dict[str, int]) -> Dict[str, float]:
        xi = self._process_features(xi)
        distances = [-self.similarity_func(xj, xi) for xj in self.x]
        knn_indices = np.argsort(distances)[:self.k_val]
        vote_counter = Counter()
        for knn_idx in knn_indices:
            vote_counter[self.y[knn_idx]] += 1
        probs = {label: 0. for label in self.label_set}
        for label, votes in vote_counter.items():
            probs[label] = votes / self.k_val
        return probs

    def predict(self, x: List[Dict[str, int]], pred_probs: List[Dict[str, float]] = None) -> List[str]:
        preds = []
        for probs in pred_probs if pred_probs is not None else self.predict_probabilities(x):
            preds.append(max(self.label_set, key=probs.get))
        return preds

    def dump(self, file: IO = sys.stdout) -> None:
        raise UnsupportedOperation


if __name__ == '__main__':

    TRAIN_DATA, \
        TEST_DATA, \
        K_VAL, \
        SIMILARITY_FUNC, \
        SYS_OUTPUT = sys.argv[1:7]

    K_VAL = int(K_VAL)
    SIMILARITY_FUNC = int(SIMILARITY_FUNC)
    if SIMILARITY_FUNC == 1:
        similarity_func = negative_euclidean_distance
    elif SIMILARITY_FUNC == 2:
        similarity_func = cosine_similarity
    else:
        raise UserWarning("Unsupported similarity function")

    x_train, y_train = load_data(TRAIN_DATA)
    x_test, y_test = load_data(TEST_DATA)

    knn = KNN(K_VAL, similarity_func)
    knn.train(x_train, y_train)

    prob_train = knn.predict_probabilities(x_train)
    prob_test = knn.predict_probabilities(x_test)
    with open(SYS_OUTPUT, 'w') as sys_output_file:
        print('%%%%% training data:', file=sys_output_file)
        dump_sys_output(prob_train, file=sys_output_file)
        print(file=sys_output_file)
        print(file=sys_output_file)
        print('%%%%% test data:', file=sys_output_file)
        dump_sys_output(prob_test, file=sys_output_file)

    pred_train = knn.predict(x_train, pred_probs=prob_train)
    pred_test = knn.predict(x_test, pred_probs=prob_test)
    confusion_matrix_train = Counter()
    confusion_matrix_test = Counter()
    for yi, pi in zip(y_train, pred_train):
        confusion_matrix_train[yi, pi] += 1
    for yi, pi in zip(y_test, pred_test):
        confusion_matrix_test[yi, pi] += 1

    print('Confusion matrix for the training data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_train, labels=knn.label_set)
    print()
    print(f' Training accuracy={accuracy(y_train, pred_train)}')
    print()
    print()
    print('Confusion matrix for the test data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_test, labels=knn.label_set)
    print()
    print(f' Test accuracy={accuracy(y_test, pred_test)}')
