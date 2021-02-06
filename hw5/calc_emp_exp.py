import sys
from collections import Counter, defaultdict
from typing import Tuple

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

if __name__ == '__main__':
    TRAIN_DATA, OUTPUT_FILE = sys.argv[1:3]
    x_train, y_train = load_data(TRAIN_DATA)
    raw_counts = Counter()
    for xi, yi in zip(x_train, y_train):
        for feat, value in xi.items():
            if value > 0:
                raw_counts[yi, feat] += 1
    expectations = defaultdict(float, {k: v / len(y_train) for k, v in raw_counts.items()})
    label_set = set(y_train)
    feat_set = set()
    for xi in x_train:
        for feat, value in xi.items():
            feat_set.add(feat)
    with open(OUTPUT_FILE, 'w') as of:
        for label in sorted(label_set):
            for feat in sorted(feat_set):
                print(label, feat, f'{expectations[label, feat]:.5f}', raw_counts[label, feat],
                      file=of)
