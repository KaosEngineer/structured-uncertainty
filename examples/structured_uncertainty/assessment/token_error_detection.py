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
                         'word_aep_tu',
                         'word_exe',
                         'word_aep_du',
                         'word_mi',
                         'word_epkl',
                         'word_aep_ku']

    uncertainties = {}
    for uname in uncertainty_names:
        unc = []
        with open(os.path.join(path, uname + '.txt'), 'r') as f:
            for line in f.readlines():
                unc.extend([float(tok) for tok in line[:-1].split()[:-1]])
            uncertainties[uname] = np.asarray(unc, dtype=np.float32)
    uncertainties['word_mkl'] = uncertainties['word_epkl']-uncertainties['word_mi']

    return uncertainties


def main():
    args = parser.parse_args()

    error_labels = get_error_labels(args.path)
    uncertainties = get_token_uncertainties(args.path)
    results={}
    for key in uncertainties.keys():
        print(error_labels.shape, uncertainties[key].shape)
        precision, recall, thresholds = precision_recall_curve(error_labels, uncertainties[key])
        aupr = auc(recall, precision)
        results[key] = np.round(aupr*100, 3)

    with open(os.path.join(args.path, 'results_token.txt'), 'a') as f:
        f.write(f'--TOKEN ERROR DETECT --\n')
        f.write(f'Errors: {np.sum(error_labels)} / Words {error_labels.shape[0]}\n')
        for key in results.keys():
            f.write(f'{key}: {results[key]}\n')





if __name__ == '__main__':
    main()
