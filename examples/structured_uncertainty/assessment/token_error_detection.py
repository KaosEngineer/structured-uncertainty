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
    uncertainty_names = ['word_pe_eoe',
                         'word_pe_sTU',
                         'word_ep_eoe',
                         'word_ep_sTU',
                         'word_exe',
                         'word_sDU',
                         'word_pe_mi',
                         'word_ep_mi',
                         'word_pe_epkl',
                         'word_ep_epkl',
                         'word_pe_mkl',
                         'word_ep_mkl',
                         'word_pe_sMKL',
                         'word_ep_sMKL']

    uncertainties = {}
    for uname in uncertainty_names:
        unc = []
        with open(os.path.join(path, uname + '.txt'), 'r') as f:
            for line in f.readlines():
                unc.extend([float(tok) for tok in line[:-1].split()[:-1]])
            uncertainties[uname] = np.asarray(unc, dtype=np.float64)

    return uncertainties


def normalized_cross_entropy(error_labels, probs):
    error_labels = np.asarray(error_labels, dtype=np.float64)
    error_prob = np.float(np.sum(error_labels)) / np.float(error_labels.shape[0])
    probs = np.exp(-probs)

    levenstein_entropy = - error_prob*np.log(error_prob) \
                         - (1.0-error_prob)*np.log(1.0-error_prob)

    cross_entropy = -np.mean(error_labels*np.log(1.0-probs+1e-6) + (1.0-error_labels)*np.log(probs+1e-6))

    nce = (levenstein_entropy - cross_entropy)/levenstein_entropy
    return nce

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

        if key in ['word_pe_sTU', 'word_ep_sTU']:
            nce=normalized_cross_entropy(error_labels, uncertainties[key])
            if key == 'word_pe_sTU':
                results['NCE_PE'] = nce
            elif key == 'word_ep_sTU':
                results['NCE_EP'] = nce

    with open(os.path.join(args.path, 'results_token.txt'), 'a') as f:
        f.write(f'--TOKEN ERROR DETECT --\n')
        f.write(f'Errors: {np.sum(error_labels)} / Words {error_labels.shape[0]}\n')
        for key in results.keys():
            f.write(f'{key}: {results[key]}\n')





if __name__ == '__main__':
    main()
