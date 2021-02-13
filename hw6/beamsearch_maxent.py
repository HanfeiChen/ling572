import sys
import math
import tqdm
from copy import copy, deepcopy
from collections import defaultdict
from typing import Optional, List, Dict, IO, Tuple


class BeamSearchNode:
    sent_id: int
    token_id: int
    features: Dict[str, int]
    pred_label: Optional[float]
    prob_label: Optional[float]
    lg_prob_label: Optional[float]
    prob_hist: Optional[float]
    lg_prob_hist: Optional[float]
    prev: 'BeamSearchNode'

    def __init__(self, instance: 'Instance', prev: Optional['BeamSearchNode']) -> None:
        self.sent_id = instance.sent_id
        self.token_id = instance.token_id
        self.features = copy(instance.features)
        self.prev = prev

class Instance:
    sent_id: int
    token_id: int
    instance_name: str
    features: Dict[str, int]
    label: Optional[str]

    def __init__(self,
                 sent_id: int,
                 token_id: int,
                 instance_name: str,
                 features: Dict[str, int],
                 label: Optional[str] = None) -> None:
        self.name = instance_name
        self.sent_id = sent_id
        self.token_id = token_id
        self.features = features
        self.label = label

    def add_feature(self, feat: str, value: int) -> None:
        self.features[feat] = value

    def __repr__(self) -> str:
        return f"Instance({self.sent_id},{self.token_id}, name={self.name})"

    def add_additional_features(self, prev_node: BeamSearchNode = None):
        if prev_node is None and self.token_id != 0:
            raise UserWarning('This should only be called after setting prev or if token_id = 0')
        if self.token_id == 0:
            self.features['prevT=BOS'] = 1
            self.features['prevTwoTags=BOS+BOS'] = 1
        elif self.token_id == 1:
            self.features[f'prevT={prev_node.pred_label}'] = 1
            self.features[f'prevTwoTags=BOS+{prev_node.pred_label}'] = 1
        else:
            self.features[f'prevT={prev_node.pred_label}'] = 1
            self.features[f'prevTwoTags={prev_node.prev.pred_label}+{prev_node.pred_label}'] = 1



def load_test_data(file_path: str, boundaries: List[int]) -> List[Instance]:
    instances = []

    curr_sent_id = 0
    curr_token_id = -1
    with open(file_path, 'r') as in_file:
        for line in in_file:
            if len(line.strip()) > 0:
                parts = line.split()
                instance_name, label, feature_strings = parts[0], parts[1], parts[2:]
                feature_names, feature_values = parts[2::2], parts[3::2]
                features = {feat: int(val) for feat, val in zip(feature_names, feature_values)}

                curr_token_id += 1
                if curr_token_id > boundaries[curr_sent_id]:
                    curr_sent_id += 1
                    curr_token_id = 0

                instance = Instance(sent_id=curr_sent_id + 1,
                                    token_id=curr_token_id,
                                    instance_name=instance_name,
                                    features=features,
                                    label=label)

                instances.append(instance)

    return instances


def load_boundary_file(file_path: str) -> List[int]:
    boundaries = []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            if len(line.strip()) > 0:
                boundary = int(line.strip())
                boundaries.append(boundary)
    return boundaries



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


class MaxEntBeamSearchDecoder:
    def __init__(self, model: MaxEntDecoder, beam_size: int, top_n: int, top_k: int) -> None:
        self.model = model
        self.beam_size = beam_size
        self.top_n = top_n
        self.top_k = top_k

    def beam_search_sentence(self, sentence: List[Instance]) -> Tuple[List, List]:
        kept_nodes = []
        for idx, instance in enumerate(sentence):
            new_nodes = []
            if idx == 0:
                instance.add_additional_features()
                model_preds = self.model.predict_probability(instance.features)
                for label, prob in sorted(model_preds.items(), key=lambda x: -x[1])[:self.top_n]:
                    node = BeamSearchNode(instance, None)
                    node.pred_label = label
                    node.prob_hist = prob
                    node.lg_prob_hist = math.log10(prob)
                    node.prob_label = prob
                    node.lg_prob_label = math.log10(prob)
                    new_nodes.append(node)
                kept_nodes = new_nodes
            else:
                for prev_node in kept_nodes:
                    instance_copy = deepcopy(instance)
                    instance_copy.add_additional_features(prev_node)
                    model_preds = self.model.predict_probability(instance_copy.features)
                    for label, prob in sorted(model_preds.items(), key=lambda x: -x[1])[:self.top_n]:
                        node = BeamSearchNode(instance, prev_node)
                        node.pred_label = label
                        node.prob_hist = prob * prev_node.prob_hist
                        node.lg_prob_hist = math.log10(prob) + prev_node.lg_prob_hist
                        node.prob_label = prob
                        node.lg_prob_label = math.log10(prob)
                        new_nodes.append(node)

                # prune
                surviving_nodes = []
                max_lg_prob = max(x.lg_prob_hist for x in new_nodes)
                for node in sorted(new_nodes, key=lambda x: -x.lg_prob_hist)[:self.top_k]:
                    if node.lg_prob_hist + self.beam_size >= max_lg_prob:
                        surviving_nodes.append(node)
                kept_nodes = surviving_nodes

        # choose best node in kept_nodes
        best_node = max(kept_nodes, key=lambda x: x.lg_prob_hist)
        # decode best_node into list of dicts
        preds, probs = [], []
        current: BeamSearchNode = best_node
        while current is not None:
            probs.append(current.prob_label)
            preds.append(current.pred_label)
            current = current.prev
        return reversed(preds), reversed(probs)

    def beam_search_all(self, data: List[Instance]) -> Tuple[List, List]:
        preds, probs = [], []
        current_sentence: List[Instance] = []
        for instance in tqdm.tqdm(data):
            if len(current_sentence) == 0 or current_sentence[-1].sent_id == instance.sent_id:
                current_sentence.append(instance)
            else:
                sent_preds, sent_probs = self.beam_search_sentence(current_sentence)
                preds.extend(sent_preds)
                probs.extend(sent_probs)
                current_sentence = [instance]
        # one more
        sent_preds, sent_probs = self.beam_search_sentence(current_sentence)
        preds.extend(sent_preds)
        probs.extend(sent_probs)
        current_sentence = []
        return preds, probs


def dump_sys_output(instances: List[Instance], pred_list: List[float], prob_list: List[str], file: IO = sys.stdout) -> None:
    print('%%%%% test data:', file=file)
    for instance, pred, prob in zip(instances, pred_list, prob_list):
        print(f'{instance.name} {instance.label} {pred} {prob}', file=file)


if __name__ == '__main__':
    import time
    start = time.time()
    TEST_DATA, \
        BOUNDARY_FILE, \
        MODEL_FILE, \
        SYS_OUTPUT, \
        BEAM_SIZE, \
        TOP_N, \
        TOP_K = sys.argv[1:8]

    # TEST_DATA, \
    #     BOUNDARY_FILE, \
    #     MODEL_FILE, \
    #     SYS_OUTPUT, \
    #     BEAM_SIZE, \
    #     TOP_N, \
    #     TOP_K = 'examples/ex/test.txt examples/ex/boundary.txt examples/m1.txt sys_out 0 1 1'.split()

    BEAM_SIZE, TOP_N, TOP_K = int(BEAM_SIZE), int(TOP_N), int(TOP_K)

    boundaries: List[int] = load_boundary_file(BOUNDARY_FILE)
    model = MaxEntDecoder(MODEL_FILE)
    test_data = load_test_data(TEST_DATA, boundaries)
    decoder = MaxEntBeamSearchDecoder(model, BEAM_SIZE, TOP_N, TOP_K)
    preds, probs = decoder.beam_search_all(test_data)

    with open(SYS_OUTPUT, 'w') as sys_output:
        dump_sys_output(test_data, preds, probs, sys_output)

    print(f'Took {time.time() - start}s')
