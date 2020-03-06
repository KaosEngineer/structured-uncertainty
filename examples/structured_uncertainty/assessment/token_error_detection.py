import numpy as np
import os
import argparse
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve

sns.set()

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain uncertainties.')


def get_error_labels(path):
    labels = []
    with open(os.path.join(path, 'error_labels.txt'), 'r') as f:
        for line in f.readlines():
            labels.extend([int(tok) for tok in line[:-1].split()])

    return np.asarray(labels)


def get_token_uncertainties(path):
    uncertainty_names = ['word_eoe',
                         'word_exe',
                         'word_mi',
                         'word_epkl',
                         'word_scores']

    uncertainties = {}
    for uname in uncertainty_names:
        unc = []
        with open(os.path.join(path, uname + '.txt'), 'r') as f:
            for line in f.readlines():
                unc.extend([float(tok) for tok in line[:-1].split()])
            if uname == 'word_scores':
                uncertainties[uname] = -np.asarray(unc, dtype=np.float32)
            else:
                uncertainties[uname] = np.asarray(unc, dtype=np.float32)

    return uncertainties


def main():
    args = parser.parse_args()

    error_labels = get_error_labels(args.path)
    uncertainties = get_token_uncertainties(args.path)

    results={}
    for key in uncertainties.keys():
        precision, recall, thresholds = precision_recall_curve(error_labels, uncertainties[key])
        aupr = auc(recall, precision)
        results[key] = np.round(aupr*100, 3)

    with open(os.path.join(args.path, 'results.txt'), 'a') as f:
        f.write(f'Errors: {np.sum(error_labels)} / Words {error_labels.shape[0]}\n')
        for key in results.keys():
            f.write(f'{key}: {results[key]}\n')





if __name__ == '__main__':
    main()
