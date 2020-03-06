#! /usr/bin/env python

import os, sys

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import seaborn as sns
import argparse
from pathlib import Path

sns.set()
sns.set(font_scale=1.25)

parser = argparse.ArgumentParser(description='Assess ood detection performance')

parser.add_argument('id_path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('ood_path', type=str,
                    help='Path of directory containing out-of-domain uncertainties.')
parser.add_argument('output_path', type=str,
                    help='Path of directory where to save results.')


def load_uncertainties(path):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float32)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float32)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float32)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float32)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float32)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float32)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float32)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float32)
    unc_dict = {'entropy_of_expected': eoe,
            'expected_entropy': exe,
            'mutual_informaiton': mi,
            'EPKL': epkl,
            'score': score,
            'AEP_TU': aep_tu,
            'AEP_DU': aep_du,
            'NPMI': npmi,
            'SCORE-NPMI': aep_du-score}
    if os.path.exists(os.path.join(path, 'xbleu.txt')):
        xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float32)
        unc_dict['XBLEU'] = xbleu
    if os.path.exists(os.path.join(path, 'xwer.txt')):
        xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float32)
        unc_dict['XWER'] = xwer

    return unc_dict


def eval_ood_detect(in_uncertainties, out_uncertainties, save_path):
    for mode in ['PR', 'ROC']:
        for key in in_uncertainties.keys():
            ood_detect(in_uncertainties[key],
                       out_uncertainties[key],
                       measure_name=key,
                       mode=mode,
                       save_path=save_path)


def ood_detect(in_measure, out_measure, measure_name, save_path, mode):

    # if out_measure.shape[0] > in_measure.shape[0]:
    #     out_measure = out_measure[:in_measure.shape[0]]
    # assert out_measure.shape[0] == in_measure.shape[0]
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)

    domain_labels = np.concatenate((np.zeros_like(in_measure, dtype=np.int32),
                                   np.ones_like(out_measure, dtype=np.int32)), axis=0)

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUPR using ' + measure_name + ": " + str(np.round(aupr * 100.0, 1)) + '\n')
        np.savetxt(os.path.join(save_path, measure_name + '_recall.txt'), recall)
        np.savetxt(os.path.join(save_path, measure_name + '_precision.txt'), precision)

        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'PR_' + measure_name + '.png'))
        plt.close()

    elif mode == 'ROC':
        fpr, tpr, thresholds = roc_curve(domain_labels, scores)
        roc_auc = roc_auc_score(domain_labels, scores)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUROC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')
        np.savetxt(os.path.join(save_path, measure_name + '_trp.txt'), tpr)
        np.savetxt(os.path.join(save_path, measure_name + '_frp.txt'), fpr)

        plt.plot(fpr, tpr)
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'ROC_' + measure_name + '.png'))
        plt.close()


def main():
    args = parser.parse_args()
    # if not os.path.isdir('CMDs'):
    #     os.mkdir('CMDs')
    # with open('CMDs/ood_detect.cmd', 'a') as f:
    #     f.write(' '.join(sys.argv) + '\n')
    #     f.write('--------------------------------\n')
    # if os.path.isdir(args.output_path) and not args.overwrite:
    #     print(f'Directory {args.output_path} exists. Exiting...')
    #     sys.exit()
    # elif os.path.isdir(args.output_path) and args.overwrite:
    #     os.remove(args.output_path + '/*')
    # else:
    #     os.makedirs(args.output_path)

    # Get dictionary of uncertainties.
    id_uncertainties = load_uncertainties(args.id_path)
    ood_uncertainties = load_uncertainties(args.ood_path)

    eval_ood_detect(in_uncertainties=id_uncertainties,
                    out_uncertainties=ood_uncertainties,
                    save_path=args.output_path)


if __name__ == '__main__':
    main()
