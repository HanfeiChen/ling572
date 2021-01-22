#!/usr/bin/env python3

from collections import Counter, defaultdict
import math
import sys
from typing import Any, Dict, IO, List, Set, Tuple


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


class MultinomialNaiveBayes:
    def __init__(self, prior_delta: float = 1., cond_prob_delta: float = 1.):
        self.prior_delta: float = prior_delta
        self.cond_prob_delta: float = cond_prob_delta

    def train(self, x: List[Dict[str, int]], y: List[str]) -> None:
        class_set = set()
        feat_set = set()
        self.class_counter = Counter()
        self.feat_class_counter = Counter()
        self.z_counter = Counter()
        for xi, yi in zip(x, y):
            class_set.add(yi)
            self.class_counter[yi] += 1
            for feat_name, count in xi.items():
                feat_set.add(feat_name)
                self.feat_class_counter[(feat_name, yi)] += count
                self.z_counter[yi] += count

        self.class_set = class_set
        self.feat_set = feat_set

    def cond_prob(self, feat_name: str, class_name: str) -> float:
        return (self.feat_class_counter[(feat_name, class_name)] + self.cond_prob_delta) \
                / (self.z_counter[class_name] + self.cond_prob_delta * len(self.feat_set))

    def prior_prob(self, class_name: str) -> float:
        return (self.class_counter[class_name] + self.prior_delta) \
                / sum(self.class_counter.values()) + self.prior_delta * len(self.class_set)

    def predict_probabilities(self, x: List[Dict[str, int]]) -> List[Dict[str, float]]:
        self.log_z = defaultdict(float)
        for c in self.class_set:
            for f in self.feat_set:
                self.log_z[c] += math.log10(1. - self.cond_prob(f, c))
        # print('logZ', self.log_z)
        return [self.predict_probability(xi) for xi in x]

    def predict_probability(self, xi: Dict[str, int]) -> Dict[str, float]:
        log_probs = dict()  # log P(xi | c)
        for c in self.class_set:
            log_prob = self.log_z[c]
            for f, count in xi.items():
                log_prob += count * math.log10(self.cond_prob(f, c))
            log_probs[c] = log_prob

        # print('log P(xi | c)', log_probs)
        log_probs = {c: v + math.log10(self.prior_prob(c)) for c, v in log_probs.items()}
        # print('log P(xi, c)', log_probs)
        log_probs_fix = {c: v - max(log_probs.values()) for c, v in log_probs.items()}
        # print('log P(xi, c)', log_probs)
        probs_fix = {c: math.pow(10, v) for c, v in log_probs_fix.items()}
        probs = {c: v / sum(probs_fix.values()) for c, v in probs_fix.items()}
        return probs

    def predict(self, x: List[Dict[str, int]]) -> List[str]:
        preds = []
        for probs in self.predict_probabilities(x):
            preds.append(max(self.class_set, key=probs.get))
        return preds

    def dump(self, file: IO = sys.stdout) -> None:
        print(f'%%%%% prior prob P(c) %%%%%', file=file)
        for c in sorted(self.class_set):
            prob = self.prior_prob(c)
            if prob > 0.:
                print(c, prob, math.log10(prob), file=file)
        print(f'%%%%% conditional prob P(f|c) %%%%%', file=file)
        for c in sorted(self.class_set):
            print(f'%%%%% conditional prob P(f|c) c={c} %%%%%', file=file)
            for f in sorted(self.feat_set):
                prob = self.cond_prob(f, c)
                if prob > 0.:
                    print(f, c, prob, math.log10(prob), file=file)


if __name__ == '__main__':

    TRAIN_DATA, \
        TEST_DATA, \
        CLASS_PRIOR_DELTA, \
        COND_PROB_DELTA, \
        MODEL_FILE, \
        SYS_OUTPUT = sys.argv[1:7]

    CLASS_PRIOR_DELTA = float(CLASS_PRIOR_DELTA)
    COND_PROB_DELTA = float(COND_PROB_DELTA)

    x_train, y_train = load_data(TRAIN_DATA)
    x_test, y_test = load_data(TEST_DATA)

    nb = MultinomialNaiveBayes(prior_delta=CLASS_PRIOR_DELTA, cond_prob_delta=COND_PROB_DELTA)
    nb.train(x_train, y_train)
    with open(MODEL_FILE, 'w') as model_file:
        nb.dump(model_file)

    prob_train = nb.predict_probabilities(x_train)
    prob_test = nb.predict_probabilities(x_test)
    with open(SYS_OUTPUT, 'w') as sys_output_file:
        print('%%%%% training data:', file=sys_output_file)
        dump_sys_output(prob_train, file=sys_output_file)
        print(file=sys_output_file)
        print(file=sys_output_file)
        print('%%%%% test data:', file=sys_output_file)
        dump_sys_output(prob_test, file=sys_output_file)

    pred_train = nb.predict(x_train)
    pred_test = nb.predict(x_test)
    confusion_matrix_train = Counter()
    confusion_matrix_test = Counter()
    for yi, pi in zip(y_train, pred_train):
        confusion_matrix_train[yi, pi] += 1
    for yi, pi in zip(y_test, pred_test):
        confusion_matrix_test[yi, pi] += 1

    print('Confusion matrix for the training data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_train, labels=nb.class_set)
    print()
    print(f' Training accuracy={accuracy(y_train, pred_train)}')
    print()
    print()
    print('Confusion matrix for the test data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_test, labels=nb.class_set)
    print()
    print(f' Test accuracy={accuracy(y_test, pred_test)}')
