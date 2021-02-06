import sys
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Any, IO


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


def dump_sys_output(prob_list: List[Dict[str, float]], pred_list: List[str], file: IO = sys.stdout) -> None:
    for idx, (probs, pred) in enumerate(zip(prob_list, pred_list)):
        print(f'array:{idx} {pred}', end='', file=file)
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            print('', label, prob, sep=' ', end='', file=file)
        print(file=file)


class MaxEntDecoder:
    DEFAULT_FEAT = '<default>'

    def __init__(self, model_file_path: str) -> None:
        self.weights = defaultdict(float)
        self.label_set = set()
        with open(model_file_path, 'r') as f:
            current_label = None
            for line in f:
                if len(line.strip()) > 0:
                    line = line.rstrip()
                    if line.startswith('FEATURES FOR CLASS'):
                        current_label = line.split()[-1]
                        self.label_set.add(current_label)
                    else:
                        feat, weight = line.split()
                        weight = float(weight)
                        self.weights[current_label, feat] = weight

    def predict_probabilities(self, x: List[Dict[str, int]]) -> List[Dict[str, float]]:
        return [self.predict_probability(xi) for xi in x]

    def predict_probability(self, xi: Dict[str, int]) -> Dict[str, float]:
        exponents = dict()
        for label in self.label_set:
            exponent = self.weights[label, self.DEFAULT_FEAT]
            for feat, value in xi.items():
                if value > 0:
                    exponent += self.weights[label, feat]
            exponents[label] = exponent
        numerators = {k: math.exp(v) for k, v in exponents.items()}
        normalizer = sum(numerators.values())
        probs = {k: v / normalizer for k, v in numerators.items()}
        return probs

    def predict(self, x: List[Dict[str, int]], pred_probs: List[Dict[str, float]] = None) -> List[str]:
        preds = []
        for probs in pred_probs if pred_probs is not None else self.predict_probabilities(x):
            preds.append(max(self.label_set, key=probs.get))
        return preds


if __name__ == '__main__':
    TEST_DATA, MODEL_FILE, SYS_OUTPUT = sys.argv[1:4]
    x_test, y_test = load_data(TEST_DATA)

    decoder = MaxEntDecoder(MODEL_FILE)

    prob_test = decoder.predict_probabilities(x_test)
    pred_test = decoder.predict(x_test, pred_probs=prob_test)

    with open(SYS_OUTPUT, 'w') as sys_output_file:
        print('%%%%% test data:', file=sys_output_file)
        dump_sys_output(prob_test, pred_test, file=sys_output_file)

    confusion_matrix_test = Counter()
    for yi, pi in zip(y_test, pred_test):
        confusion_matrix_test[yi, pi] += 1

    print('Confusion matrix for the test data:')
    print('row is the truth, column is the system output')
    dump_confusion_matrix(confusion_matrix_test, labels=decoder.label_set)
    print()
    print(f' Test accuracy={accuracy(y_test, pred_test)}')
