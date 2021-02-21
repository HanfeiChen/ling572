import sys
import numpy as np
import tqdm
from collections import Counter
from typing import List, Dict, Tuple, Any, IO


def load_data(file_path: str) -> Tuple:
    x, y = [], []
    with open(file_path, 'r') as in_file:
        for line in in_file:
            if len(line.strip()) > 0:
                parts = line.split()
                label, feature_strings = int(parts[0]), parts[1:]
                features = Counter()
                for feat_str in feature_strings:
                    feat, val = feat_str.split(':')
                    features[int(feat)] = int(val)
                x.append(features)
                y.append(label)
    return x, y


def accuracy(y: List[Any], pred: List[Any]) -> float:
    correct = 0
    for yi, pi in zip(y, pred):
        if yi == pi:
            correct += 1
    return correct / len(y)


def dump_sys_output(gold_list: List[int], pred_list: List[str], fx_list: List[float], file: IO = sys.stdout) -> None:
    for (gold, pred, fx) in zip(gold_list, pred_list, fx_list):
        print(gold, pred, fx, file=file)


class SVMClassifier:
    def __init__(self, model_file: str) -> None:
        self.svs: List[np.array] = list()
        self.sv_weights: List[float] = list()
        self.sv_dim: int = 0

        header: Dict[str, Any] = dict()
        sv_dicts: List[Dict] = list()
        segment = 'header'
        with open(model_file, 'r') as in_file:
            for line in in_file:
                if len(line.strip()) > 0:
                    line = line.strip()
                    parts = line.split()
                    if line == 'SV':
                        segment = 'sv'
                        continue
                    if segment == 'header':
                        if len(parts) > 2:
                            header[parts[0]] = tuple(parts[1:])
                        else:
                            header[parts[0]] = parts[1]
                    else:
                        weight = float(parts[0])
                        sv = [fv.split(':') for fv in parts[1:]]
                        sv = {int(fv[0]): int(fv[1]) for fv in sv}
                        self.sv_dim = max(self.sv_dim, max(sv.keys()))
                        self.sv_weights.append(weight)
                        sv_dicts.append(sv)
        self.sv_dim += 1

        self.kernel_type = header['kernel_type']
        self.rho = float(header['rho'])
        self.gamma = float(header['gamma']) if 'gamma' in header else None
        self.coef0 = float(header['coef0']) if 'coef0' in header else None
        self.degree = int(header['degree']) if 'degree' in header else None
        if self.kernel_type == 'linear':
            self.kernel_func = lambda x, z: np.inner(x, z)
        elif self.kernel_type == 'polynomial':
            self.kernel_func = lambda x, z: np.power(self.gamma * np.inner(x, z) + self.coef0, self.degree)
        elif self.kernel_type == 'rbf':
            self.kernel_func = lambda x, z: np.exp(-self.gamma * np.linalg.norm(x - z) * np.linalg.norm(x - z))
        elif self.kernel_type == 'sigmoid':
            self.kernel_func = lambda x, z: np.tanh(self.gamma * np.inner(x, z) + self.coef0)
        else:
            raise UserWarning('unsupported kernel type')

        self.svs = [self._to_numpy(feat_dict) for feat_dict in sv_dicts]

    def _to_numpy(self, feat_dict: Dict[int, float]) -> np.array:
        vec = np.zeros(self.sv_dim)
        for k, v in feat_dict.items():
            if k < self.sv_dim:
                vec[k] = v
        return vec

    def compute_fx_one(self, x: Dict[str, float]) -> float:
        x_vec = self._to_numpy(x)
        k_values = [self.kernel_func(xi, x_vec) for xi in self.svs]
        f = sum(wi * ki for wi, ki in zip(self.sv_weights, k_values)) - self.rho
        return f

    def compute_fx(self, x: List[Dict[str, float]]) -> float:
        return [self.compute_fx_one(xi) for xi in tqdm.tqdm(x)]

    def predict_one(self, x: Dict[str, float], fx: float = None) -> int:
        if fx is None:
            fx = self.compute_fx_one(x)
        return 0 if fx > 0 else 1

    def predict(self, x: List[Dict[str, float]], fx: List[float] = None) -> int:
        if fx is not None:
            return [self.predict_one(xi, fxi) for xi, fxi in zip(x, fx)]
        else:
            return [self.predict_one(xi) for xi in x]


if __name__ == '__main__':
    TEST_DATA, MODEL_FILE, SYS_OUTPUT = sys.argv[1:4]
    # TEST_DATA, MODEL_FILE, SYS_OUTPUT = 'examples/test', 'model.1', 'sys.1'

    x_test, y_test = load_data(TEST_DATA)

    decoder = SVMClassifier(MODEL_FILE)

    fx_test = decoder.compute_fx(x_test)
    pred_test = decoder.predict(x_test, fx=fx_test)

    with open(SYS_OUTPUT, 'w') as sys_output_file:
        dump_sys_output(y_test, pred_test, fx_test, file=sys_output_file)

    acc = accuracy(y_test, pred_test)
    print('Accuracy:', acc)
