#! /usr/bin/env python

import os

import matplotlib

#from examples.structured_uncertainty.assessment.utils import load_uncertainties
from scipy.special import softmax
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import seaborn as sns
import argparse

sns.set()
sns.set(font_scale=1.25)

parser = argparse.ArgumentParser(description='Assess ood detection performance')

parser.add_argument('id_path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('ood_path', type=str,
                    help='Path of directory containing out-of-domain uncertainties.')
parser.add_argument('output_path', type=str,
                    help='Path of directory where to save results.')
parser.add_argument('--beam_width', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--beam_search', action='store_true',
                    help='Path of directory where to save results.')
parser.add_argument('--calibrate', action='store_true',
                    help='Path of directory where to save results.')



def load_uncertainties(path, n_best=5, beam_width=5, beam_search=True):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float32)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float32)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float32)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float32)
    mkl = np.loadtxt(os.path.join(path, 'mkl.txt'), dtype=np.float32)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float32)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float32)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float32)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float32)
    lprobs = np.loadtxt(os.path.join(path, 'log_probs.txt'), dtype=np.float32)

    ep_eoe = np.loadtxt(os.path.join(path, 'ep_entropy_expected.txt'), dtype=np.float32)
    ep_mi = np.loadtxt(os.path.join(path, 'ep_mutual_information.txt'), dtype=np.float32)
    ep_epkl = np.loadtxt(os.path.join(path, 'ep_epkl.txt'), dtype=np.float32)
    ep_mkl = np.loadtxt(os.path.join(path, 'ep_mkl.txt'), dtype=np.float32)


    var = np.loadtxt(os.path.join(path, 'var.txt'), dtype=np.float64)
    varcombo = np.loadtxt(os.path.join(path, 'varcombo.txt'), dtype=np.float64)
    logvar = np.loadtxt(os.path.join(path, 'logvar.txt'), dtype=np.float64)
    logcombo = np.loadtxt(os.path.join(path, 'logcombo.txt'), dtype=np.float64)

    unc_dict = {'Total Uncertainty-PE': eoe,
                'Total Uncertainty-EP': ep_eoe,
                'SCR-PE': score,
                'SCR-EP': aep_tu,
                'Data Uncertainty': exe,
                'Mutual Information-PE': mi,
                'Mutual Information-EP': ep_mi,
                'EPKL-PE': epkl,
                'EPKL-EP': ep_epkl,
                'MKL': mkl,
                'ep_MKL': ep_mkl,
                'sMKL-PE': aep_du - score,
                'sMKL-EP': npmi,
                'var': var,
                'varcombo': varcombo,
                'logvar': logvar,
                'logcomvo': logcombo
                }
    # }
    if os.path.exists(os.path.join(path, 'xbleu.txt')):
        xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float32)
        unc_dict['XBLEU'] = xbleu
    if os.path.exists(os.path.join(path, 'xwer.txt')):
        xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float32)
        unc_dict['XWER'] = xwer


    weights = softmax(lprobs.reshape([-1, beam_width])[:, :n_best], axis=1)
    for key in unc_dict.keys():
        uncertainties = unc_dict[key]
        if beam_search:
            unc_dict[key] = np.sum(weights * np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
        else:
            unc_dict[key] = np.mean(np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
    return unc_dict



def load_uncertainties_calibrate(path, n_best=5, beam_width=5, beam_search=True, temp=1):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float64)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float64)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float64)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float64)
    mkl = np.loadtxt(os.path.join(path, 'mkl.txt'), dtype=np.float64)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float64)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float64)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float64)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float64)
    lprobs = np.loadtxt(os.path.join(path, 'log_probs.txt'), dtype=np.float64)

    ep_eoe = np.loadtxt(os.path.join(path, 'ep_entropy_expected.txt'), dtype=np.float64)
    ep_mi = np.loadtxt(os.path.join(path, 'ep_mutual_information.txt'), dtype=np.float64)
    ep_epkl = np.loadtxt(os.path.join(path, 'ep_epkl.txt'), dtype=np.float64)
    ep_mkl = np.loadtxt(os.path.join(path, 'ep_mkl.txt'), dtype=np.float64)


    var = np.loadtxt(os.path.join(path, 'var.txt'), dtype=np.float64)
    varcombo = np.loadtxt(os.path.join(path, 'varcombo.txt'), dtype=np.float64)
    logvar = np.loadtxt(os.path.join(path, 'logvar.txt'), dtype=np.float64)
    logcombo = np.loadtxt(os.path.join(path, 'logcombo.txt'), dtype=np.float64)

    unc_dict = {'Total Uncertainty-PE': eoe,
                'Total Uncertainty-EP': ep_eoe,
                'SCR-PE': score,
                'SCR-EP': aep_tu,
                'Data Uncertainty': exe,
                'Mutual Information-PE': mi,
                'Mutual Information-EP': ep_mi,
                'EPKL-PE': epkl,
                'EPKL-EP': ep_epkl,
                'MKL': mkl,
                'ep_MKL': ep_mkl,
                'sMKL-PE': aep_du - score,
                'sMKL-EP': npmi,
                'var': var,
                'varcombo': varcombo,
                'logvar': logvar,
                'logcomvo': logcombo
                }
    # }
    if os.path.exists(os.path.join(path, 'xbleu.txt')):
        xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float64)
        unc_dict['XBLEU'] = xbleu
    if os.path.exists(os.path.join(path, 'xwer.txt')):
        xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float64)
        unc_dict['XWER'] = xwer


    weights = softmax(lprobs.reshape([-1, beam_width])[:, :n_best]/temp, axis=1)
    for key in unc_dict.keys():
        uncertainties = unc_dict[key]
        if beam_search:
            unc_dict[key] = np.sum(weights * np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
        else:
            unc_dict[key] = np.mean(np.reshape(uncertainties, [-1, beam_width])[:, :n_best], axis=1)
    return unc_dict

def eval_ood_detect(in_uncertainties, out_uncertainties, save_path):
    for mode in ['ROC']:
        for key in in_uncertainties.keys():
            ood_detect(in_uncertainties[key],
                       out_uncertainties[key],
                       measure_name=key,
                       mode=mode,
                       save_path=save_path)


def ood_detect(in_measure, out_measure, measure_name, save_path, mode, beam_search=True):
    # if out_measure.shape[0] > in_measure.shape[0]:
    #     out_measure = out_measure[:in_measure.shape[0]]
    # assert out_measure.shape[0] == in_measure.shape[0]
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    results_path = save_path
    save_path = os.path.split(save_path)[0]

    domain_labels = np.concatenate((np.zeros_like(in_measure, dtype=np.int32),
                                    np.ones_like(out_measure, dtype=np.int32)), axis=0)

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        with open(results_path, 'a') as f:
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
        with open(results_path, 'a') as f:
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
    if args.calibrate:
        output_path = os.path.join(args.output_path, 'results_ood_temp.txt')

        for temp in np.arange(1.0, 51.0, 1.0):
            id_uncertainties = load_uncertainties_calibrate(args.id_path, n_best=args.nbest, beam_width=args.beam_width,
                                                  beam_search=True, temp=temp)
            ood_uncertainties = load_uncertainties_calibrate(args.ood_path, n_best=args.nbest, beam_width=args.beam_width,
                                                   beam_search=True, temp=temp)
            with open(output_path, 'a') as f:
                f.write(f'--OOD DETECT N-BEST {args.nbest} BW: {args.beam_width}, IW:{True}, Temp: {temp}--\n')
            eval_ood_detect(in_uncertainties=id_uncertainties,
                            out_uncertainties=ood_uncertainties,
                            save_path=output_path)


    else:
        id_uncertainties = load_uncertainties(args.id_path, n_best=args.nbest, beam_width=args.beam_width,
                                              beam_search=args.beam_search)
        ood_uncertainties = load_uncertainties(args.ood_path, n_best=args.nbest, beam_width=args.beam_width,
                                               beam_search=args.beam_search)


        if args.beam_search:
            output_path = os.path.join(args.output_path, 'results_ood_bs.txt')
        else:
            output_path = os.path.join(args.output_path, 'results_ood_mc.txt')
        with open(output_path, 'a') as f:
            f.write(f'--OOD DETECT N-BEST {args.nbest} BW: {args.beam_width}, BS:{args.beam_search}--\n')
        eval_ood_detect(in_uncertainties=id_uncertainties,
                        out_uncertainties=ood_uncertainties,
                        save_path=output_path)


if __name__ == '__main__':
    main()
