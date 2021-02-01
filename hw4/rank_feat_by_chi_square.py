import sys
from typing import List, Dict, Set
from collections import Counter, defaultdict

def load_data():
    x, y = [], []
    for line in sys.stdin:
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


def calculate_chi2(x: List[Dict[str, int]],
                   y: List[str],
                   feat_set: Set[str],
                   label_set: Set[str]):
    feat_label_counts = Counter()
    doc_freqs = Counter()
    label_counts = Counter()
    for xi, yi in zip(x, y):
        label_counts[yi] += 1
        for feat, value in xi.items():
            if value > 0:
                feat_label_counts[feat, yi] += 1
                doc_freqs[feat] += 1
    # print(feat_label_counts, label_counts, doc_freqs)
    chi2 = dict()
    for feat in feat_set:
        feat_chi2 = 0.
        for label in label_set:
            o = feat_label_counts[feat, label]
            not_o = label_counts[label] - o
            e = doc_freqs[feat] * label_counts[label] / len(x)
            not_e = (len(x) - doc_freqs[feat]) * label_counts[label] / len(x)
            feat_chi2 += (o - e) ** 2 / e
            feat_chi2 += (not_o - not_e) ** 2 / not_e
        chi2[feat] = feat_chi2
    return chi2, doc_freqs


if __name__ == '__main__':
    x, y = load_data()
    label_set = set(y)
    feat_set = set()
    for xi in x:
        for feat in xi.keys():
            feat_set.add(feat)
    chi2, doc_freqs = calculate_chi2(x, y, feat_set, label_set)
    sorted_feat = sorted(list(feat_set), key=lambda feat: -chi2[feat])
    for feat in sorted_feat:
        print(feat, chi2[feat], doc_freqs[feat])
