#!/usr/bin/env python

import math
import sys
import time
from typing import Any, Counter, IO, Iterable, List, Set, Tuple, Dict


def entropy(label_counts: Iterable[int]) -> float:
    total_count = sum(label_counts)
    probs = [(count + 1) / (total_count + len(label_counts)) for count in label_counts]
    return - sum(prob * math.log2(prob) for prob in probs)


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
        for label, prob in sorted(probs.items()):
            print('', label, prob, sep='\t', end='', file=file)
        print(file=file)


class _DecisionTreeNode:
    def __init__(self, depth: int = 0, label_set: set = {}) -> None:
        self.depth = depth
        self.label_set = label_set
        self.split_feat = None
        self.present_child = None
        self.absent_child = None
        self.feature_set = set()
        self.label_counts = Counter()
        self.label_feat_counts = Counter()
        self.x = []
        self.y = []

    def is_leaf(self) -> bool:
        return self.present_child is None and self.absent_child is None

    def add_data(self, x: List[Dict[str, int]], y: List[str]) -> None:
        self.x += x
        self.y += y

        for xi, yi in zip(x, y):
            self.feature_set |= set(xi.keys())
            self.label_set.add(yi)
            for feat, val in xi.items():
                if val > 0:
                    self.label_feat_counts[yi, feat] += 1
            self.label_counts[yi] += 1

    def predict_probabilities(self, x: List[Dict[str, int]]) -> List[Dict[str, float]]:
        current = self
        while not current.is_leaf():
            if x[current.split_feat] > 0:
                current = current.present_child
            else:
                current = current.absent_child
        return {label: current.label_counts[label] / len(current.y) for label in self.label_set}

    def grow(self, max_depth: int = None, min_gain: float = 0., debug: bool = False) -> None:
        if debug:
            print(self.depth, file=sys.stderr)
        if max_depth is not None and self.depth >= max_depth:
            return

        orig_entropy = entropy(self.label_counts.values())

        # find best feature to split on
        best_info_gain = 0.
        best_split_feat = None
        for split_feat in self.feature_set:
            present_counts = [self.label_feat_counts[label, split_feat] for label in self.label_set]
            absent_counts = [self.label_counts[label] - self.label_feat_counts[label, split_feat] for label in self.label_set]

            new_entropy = entropy(present_counts) * sum(present_counts) + entropy(absent_counts) * sum(absent_counts)
            new_entropy /= len(self.y)

            info_gain = orig_entropy - new_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split_feat = split_feat

        if debug:
            print(f'Best info gain: {best_info_gain} (splitting on {best_split_feat})', file=sys.stderr)
        if best_info_gain < min_gain:
            return

        present = [idx for idx, xi in enumerate(self.x) if xi[best_split_feat] > 0]
        absent = [idx for idx, xi in enumerate(self.x) if xi[best_split_feat] == 0]

        if len(present) == 0 or len(absent) == 0:
            return

        self.split_feat = best_split_feat
        if debug:
            print(f'Splitting on {best_split_feat}', file=sys.stderr)

        x_present = [self.x[i] for i in present]
        y_present = [self.y[i] for i in present]

        y_absent = [self.y[i] for i in absent]
        x_absent = [self.x[i] for i in absent]

        self.present_child = _DecisionTreeNode(depth=self.depth + 1, label_set=self.label_set)
        self.present_child.add_data(x_present, y_present)
        self.present_child.grow(max_depth, min_gain, debug)
        self.absent_child = _DecisionTreeNode(depth=self.depth + 1, label_set=self.label_set)
        self.absent_child.add_data(x_absent, y_absent)
        self.absent_child.grow(max_depth, min_gain, debug)

    def dump(self, path: str, file: IO) -> None:
        if self.is_leaf():
            print(path, len(self.y), end='', file=file)
            for label, count in sorted(self.predict_probabilities({}).items()):
                print('', label, count, end='', file=file)
            print(file=file)
        if self.absent_child:
            new_path = f'!{self.split_feat}' if len(path) == 0 else f'{path}&!{self.split_feat}'
            self.absent_child.dump(new_path, file=file)
        if self.present_child:
            new_path = f'{self.split_feat}' if len(path) == 0 else f'{path}&{self.split_feat}'
            self.present_child.dump(new_path, file=file)


class DecisionTree:
    def __init__(self, max_depth: int, min_gain: float) -> None:
        self.max_depth: int = max_depth
        self.min_gain: float = min_gain
        self.root: _DecisionTreeNode = _DecisionTreeNode(depth=0)
        self.label_set: set = set()

    def train(self, x: List[Dict[str, int]], y: List[str]) -> None:
        start_time = time.time()
        self.label_set = set(y)
        self.root.label_set = self.label_set
        self.root.add_data(x, y)
        self.root.grow(self.max_depth, self.min_gain)
        end_time = time.time()
        run_time = end_time - start_time
        print(f"DT training finished in {run_time / 60:.2f} min.", file=sys.stderr)

    def predict_probabilities(self, x: List[Dict[str, int]]) -> List[Dict[str, float]]:
        return [self.root.predict_probabilities(xi) for xi in x]

    def predict(self, x: List[Dict[str, int]]) -> List[str]:
        preds = []
        for probs in self.predict_probabilities(x):
            preds.append(max(self.label_set, key=probs.get))
        return preds

    def dump(self, file: IO = sys.stdout) -> None:
        self.root.dump('', file=file)


if __name__ == '__main__':
    TRAIN_DATA, TEST_DATA, MAX_DEPTH, MIN_GAIN, MODEL_FILE, SYS_OUTPUT = sys.argv[1:7]
    MAX_DEPTH, MIN_GAIN = int(MAX_DEPTH), float(MIN_GAIN)

    x_train, y_train = load_data(TRAIN_DATA)
    x_test, y_test = load_data(TEST_DATA)

    dt = DecisionTree(max_depth=MAX_DEPTH, min_gain=MIN_GAIN)
    dt.train(x_train, y_train)
    with open(MODEL_FILE, 'w') as model_file:
        dt.dump(model_file)

    prob_train = dt.predict_probabilities(x_train)
    prob_test = dt.predict_probabilities(x_test)
    with open(SYS_OUTPUT, 'w') as sys_output_file:
        print('%%%%%%%%%% training data:', file=sys_output_file)
        dump_sys_output(prob_train, file=sys_output_file)
        print(file=sys_output_file)
        print(file=sys_output_file)
        print('%%%%%%%%%% test data:', file=sys_output_file)
        dump_sys_output(prob_test, file=sys_output_file)

    pred_train = dt.predict(x_train)
    pred_test = dt.predict(x_test)
    confusion_matrix_train = Counter()
    confusion_matrix_test = Counter()
    for yi, pi in zip(y_train, pred_train):
        confusion_matrix_train[yi, pi] += 1
    for yi, pi in zip(y_test, pred_test):
        confusion_matrix_test[yi, pi] += 1

    print('Confusion matrix for the training data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_train, labels=dt.label_set)
    print()
    print(f' Training accuracy={accuracy(y_train, pred_train)}')
    print()
    print()
    print('Confusion matrix for the test data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_test, labels=dt.label_set)
    print()
    print(f' Test accuracy={accuracy(y_test, pred_test)}')
