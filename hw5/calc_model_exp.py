import sys
import math
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple, Set, Any, IO


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
    TRAIN_DATA, OUTPUT_FILE = sys.argv[1:3]
    if len(sys.argv) == 4:
        MODEL_FILE = sys.argv[3]
        model = MaxEntDecoder(MODEL_FILE)
    else:
        model = None

    x_train, y_train = load_data(TRAIN_DATA)
    label_set = set(y_train)
    feat_set = set()
    for xi in x_train:
        for feat, value in xi.items():
            feat_set.add(feat)

    default_probs = {label: 1. / len(label_set) for label in label_set}

    raw_counts = defaultdict(float)
    for xi, _ in zip(x_train, y_train):
        probs = model.predict_probability(xi) if model is not None else default_probs
        for feat, value in xi.items():
            if value > 0:
                for label in label_set:
                    raw_counts[label, feat] += probs[label]
    expectations = defaultdict(float, {k: v / len(y_train) for k, v in raw_counts.items()})
    with open(OUTPUT_FILE, 'w') as of:
        for label in sorted(label_set):
            for feat in sorted(feat_set):
                print(label, feat, f'{expectations[label, feat]:.5f}',
                      f'{raw_counts[label, feat]:.5f}',
                      file=of)
