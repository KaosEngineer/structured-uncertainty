import numpy as np
import os, sys
from sacrebleu import corpus_bleu, sentence_bleu
import argparse
import subprocess
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

sns.set()

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('--wer', action='store_true',
                    help='Whether to evaluate using WER instead of BLEU')
parser.add_argument('--nbest', type=int, default=5,
                    help='Path of directory where to save results.')
parser.add_argument('--top', action='store_true',
                    help='Path of directory where to save results.')

def get_wer(i, path):
    subprocess.run(f"~/sclite -r {path}/sorted_refs.txt -h {path}/hypos_{i}.txt -i rm -o all", shell=True)
    return np.float(subprocess.run(
        f"grep 'Sum/Avg' {path}/hypos_{i}.txt.sys | sed 's/|//g' | sed 's/\+/ /g' | awk -F ' ' '{{print $8}}' ",
        capture_output=True, shell=True).stdout)


def get_sentence_wer(path, refs, hypos):

    if not os.path.exists(f"{path}/wers.txt"):
        print("Getting WERS...)")
        subprocess.run(f"~/sclite -r {path}/refs.txt -h {path}/hypos.txt -i rm -o all", shell=True)
    subprocess.run(
        f" grep -v '-'  {path}/hypos.txt.sys | egrep -v '[A-Z]' | egrep -v '=' | sed 's/|//g' | egrep '[0-9]+' | awk '{{print $8}}' > {path}/wers.txt",
        shell=True)

    return np.loadtxt(f"{path}/wers.txt", dtype=np.float32)


def reject_predictions_wer(refs, hypos, measure, path,
                           rev: bool = False,
                           corpus_wer: bool = False):  # , measure_name: str, save_path: str,  show=True):

    refs = np.asarray(refs)
    hypos = np.asarray(hypos)

    if rev:
        inds = np.argsort(measure)
    else:
        inds = np.argsort(measure)[::-1]

    assert measure[inds[0]] >= measure[inds[-1]]

    swers = get_sentence_wer(path, refs, hypos)
    sorted_swers = swers[inds]
    # sorted_refs = refs[inds]
    # sorted_hypos = hypos[inds]

    # with open(f'{path}/sorted_refs.txt', 'w') as f:
    #     f.write('\n'.join(sorted_refs) + '\n')
    #
    # for i in range(len(hypos)):
    #     tmp_hypo = np.concatenate((sorted_refs[:i], sorted_hypos[i:]), axis=0)
    #     with open(f'{path}/hypos_{i}.txt', 'w') as f:
    #         f.write('\n'.join(tmp_hypo) + '\n')

    total_data = np.float(len(hypos))
    wers, percentages = [], []

    if corpus_wer:
        # This is where things can go epically wrong. I may be abusing the queue in weird and wonderful ways...
        wers = Parallel(n_jobs=-1)(delayed(get_wer)(i, path) for i in range(len(hypos)))
    else:
        min_swers = np.zeros_like(sorted_swers)
        wers = [np.mean(np.concatenate((min_swers[:i], sorted_swers[i:]), axis=0)) for i in range(len(hypos))]

    for i in range(len(hypos)):
        percentages.append(float(i + 1) / total_data * 100.0)

    wers, percentages = np.asarray(wers), np.asarray(percentages)

    base_wer = np.mean(swers)
    n_items = wers.shape[0]

    random_wer = np.asarray([base_wer * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
                            dtype=np.float32)

    sorted_swers = np.sort(swers)[::-1]
    perfect_wer = np.asarray(
        [np.mean(np.concatenate((min_swers[:i], sorted_swers[i:]), axis=0)) for i in range(len(hypos))])

    print(random_wer.shape, wers.shape, perfect_wer.shape)
    auc_rnd = 1.0 - auc(percentages / 100.0, random_wer / 100.0)
    auc_uns = 1.0 - auc(percentages / 100.0, wers / 100.0)
    auc_orc = 1.0 - auc(percentages / 100.0, perfect_wer / 100.0)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
    return wers, random_wer, percentages, perfect_wer, rejection_ratio


def get_corpus_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return corpus_bleu(tmp_hypo, [refs]).score


def get_bleu(refs, hypos, i):
    tmp_hypo = np.concatenate((refs[:i], hypos[i:]), axis=0)
    return np.mean([sentence_bleu([tmp_hypo[i]], [[refs[i]]], smooth_value=1.0).score for i in range(hypos.shape[0])])
    # return corpus_bleu(tmp_hypo, [refs]).score


def reject_predictions_bleu(refs, hypos, measure, path,
                            rev: bool = False, corpus_bleu=False):  # , measure_name: str, save_path: str,  show=True):
    # Get predictions

    refs = np.asarray(refs)
    hypos = np.asarray(hypos)

    if rev:
        inds = np.argsort(measure)
    else:
        inds = np.argsort(measure)[::-1]

    assert measure[inds[0]] >= measure[inds[-1]]

    sorted_refs = refs[inds]
    sorted_hypos = hypos[inds]

    #bleus = np.asarray([sentence_bleu([hypos[i]], [[refs[i]]], smooth_value=1.1).score for i in range(hypos.shape[0])])
    if not os.path.exists(f"{path}/sbleus.txt"):
        sbleus = np.asarray(
            [sentence_bleu([hypos[i]], [[refs[i]]], smooth_value=1.1).score for i in range(hypos.shape[0])])
        np.savetxt(f"{path}/sbleus.txt", sbleus)
    else:
        sbleus = np.loadtxt(f"{path}/sbleus.txt", dtype=np.float32)

    sorted_sbleus = sbleus[inds]
    # sorted_refs = refs[inds]
    # sorted_hypos = hypos[inds]

    inds = np.squeeze(inds)

    total_data = np.float(len(hypos))
    bleus, percentages = [], []

    if corpus_bleu:
        bleus = Parallel(n_jobs=-1)(delayed(get_bleu)(sorted_refs, sorted_hypos, i) for i in range(len(hypos)))
    else:
        max_sbleus = 100 * np.ones_like(sorted_sbleus)
        bleus = [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))]

    for i in range(len(hypos)):
        # tmp_ref = np.concatenate((sorted_hypos[:i], sorted_refs[i:]), axis=0)
        # score = corpus_bleu(tmp_ref, [sorted_hypos]).score
        # bleus.append(score)
        percentages.append(float(i + 1) / total_data * 100.0)

    bleus, percentages = np.asarray(bleus), np.asarray(percentages)

    base_bleu = bleus[0]
    n_items = bleus.shape[0]
    # auc_uns = 1.0 - auc(percentages / 100.0, bleus[::-1] / 100.0)

    random_bleu = np.asarray(
        [base_bleu * (1.0 - float(i) / float(n_items)) + 100.0 * (float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32)

    sorted_sbleus = np.sort(sbleus)
    print(sorted_sbleus.shape)
    perfect_bleu = np.asarray(
        [np.mean(np.concatenate((max_sbleus[:i], sorted_sbleus[i:]), axis=0)) for i in range(len(hypos))])

    auc_rnd = 1.0 - auc(percentages / 100.0, random_bleu / 100.0)
    auc_uns = 1.0 - auc(percentages / 100.0, bleus / 100.0)
    auc_orc = 1.0 - auc(percentages / 100.0, perfect_bleu / 100.0)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

    return bleus, random_bleu, percentages, perfect_bleu, rejection_ratio
    #
    #
    # orc_rejection = np.asarray(
    #     [base_error * (1.0 - float(i) / float(base_error / 100.0 * n_items)) for i in
    #      range(int(base_error / 100.0 * n_items))], dtype=np.float32)
    # orc = np.zeros_like(bleus)
    #
    # orc[0:orc_rejection.shape[0]] = orc_rejection

    #
    # random_rejection = np.squeeze(random_rejection)
    # orc = np.squeeze(orc)
    # bleus = np.squeeze(bleus)
    #
    #
    #
    # if show:
    #     plt.plot(percentages, orc, lw=2)
    #     plt.fill_between(percentages, orc, random_rejection, alpha=0.5)
    #     plt.plot(percentages, bleus[::-1], lw=2)
    #     plt.fill_between(percentages, bleus[::-1], random_rejection, alpha=0.0)
    #     plt.plot(percentages, random_rejection, 'k--', lw=2)
    #     plt.legend(['Oracle', 'Uncertainty', 'Random'])
    #     plt.xlabel('Percentage of predictions rejected to oracle')
    #     plt.ylabel('Classification Error (%)')
    #     plt.savefig('Rejection-Curve-oracle.png', bbox_inches='tight', dpi=300)
    #     # plt.show()
    #     plt.close()
    #
    #     plt.plot(percentages, orc, lw=2)
    #     plt.fill_between(percentages, orc, random_rejection, alpha=0.0)
    #     plt.plot(percentages, bleus[::-1], lw=2)
    #     plt.fill_between(percentages, bleus[::-1], random_rejection, alpha=0.5)
    #     plt.plot(percentages, random_rejection, 'k--', lw=2)
    #     plt.legend(['Oracle', 'Uncertainty', 'Random'])
    #     plt.xlabel('Percentage of predictions rejected to oracle')
    #     plt.ylabel('Classification Error (%)')
    #     plt.savefig('Rejection-Curve-uncertainty.png', bbox_inches='tight', dpi=300)
    #     # plt.show()
    #     plt.close()
    #

    # with open(os.path.join(save_path, 'results.txt'), 'a') as f:
    #     f.write(f'Rejection Ratio using {measure_name}: {np.round(rejection_ratio, 1)}\n')


# def load_uncertainties(path, n):
#     eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float32)
#     exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float32)
#     mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float32)
#     epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float32)
#     score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float32)
#     aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float32)
#     aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float32)
#     npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float32)
#     unc_dict = {'Total Uncertainty': eoe,
#             'SCR-PE': score,
#             'Data Uncertainty': exe,
#             'E-SCR': aep_du,
#             'Mutual Information': mi,
#             'EPKL': epkl,
#             'SCR-EP': aep_tu,
#             'EPMI-EP': npmi,
#             'EPMI-PE': aep_du - score}
#     if os.path.exists(os.path.join(path, 'xbleu.txt')):
#         xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float32)
#         unc_dict['XBLEU'] = xbleu
#     if os.path.exists(os.path.join(path, 'xwer.txt')):
#         xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float32)
#         unc_dict['XWER'] = xwer
#
#     return unc_dict

def load_uncertainties(path, n_best=5, top=False):
    eoe = np.loadtxt(os.path.join(path, 'entropy_expected.txt'), dtype=np.float32)
    exe = np.loadtxt(os.path.join(path, 'expected_entropy.txt'), dtype=np.float32)
    mi = np.loadtxt(os.path.join(path, 'mutual_information.txt'), dtype=np.float32)
    epkl = np.loadtxt(os.path.join(path, 'epkl.txt'), dtype=np.float32)
    score = np.loadtxt(os.path.join(path, 'score.txt'), dtype=np.float32)
    aep_tu = np.loadtxt(os.path.join(path, 'aep_tu.txt'), dtype=np.float32)
    aep_du = np.loadtxt(os.path.join(path, 'aep_du.txt'), dtype=np.float32)
    npmi = np.loadtxt(os.path.join(path, 'npmi.txt'), dtype=np.float32)
    unc_dict = {'Total Uncertainty': eoe,
            'SCR-PE': score,
            'Data Uncertainty': exe,
            'E-SCR': aep_du,
            'Mutual Information': mi,
            'EPKL': epkl,
            'SCR-EP': aep_tu,
            'EPMI-EP': npmi,
            'EPMI-PE': aep_du - score}
    if os.path.exists(os.path.join(path, 'xbleu.txt')):
        xbleu = np.loadtxt(os.path.join(path, 'xbleu.txt'), dtype=np.float32)
        unc_dict['XBLEU'] = xbleu
    if os.path.exists(os.path.join(path, 'xwer.txt')):
        xwer = np.loadtxt(os.path.join(path, 'xwer.txt'), dtype=np.float32)
        unc_dict['XWER'] = xwer

    for key in unc_dict.keys():
        uncertainties = unc_dict[key]
        if top:
            unc_dict[key] = np.reshape(uncertainties, [-1, n_best])[:, 0]
            print('Went to top...')
        else:
            unc_dict[key] = np.mean(np.reshape(uncertainties, [-1, n_best]), axis=1)
    return unc_dict


def load_text(path, nbest=5):
    refs, hypos = [], []
    with open(os.path.join(path, 'refs.txt'), 'r') as f:
        for line in f.readlines():
            refs.append(line[1:-1])

    with open(os.path.join(path, 'hypos.txt'), 'r') as f:
        count=0
        for line in f.readlines():
            if count % nbest == 0:
                hypos.append(line[1:-1])
            count += 1

    return refs, hypos


def main():
    args = parser.parse_args()


    if args.wer:
        refs, hypos = load_text(args.path, nbest=1)
        uncertainties = load_uncertainties(args.path, n_best=args.nbest, top=args.top)
        wer_dict = {}
        for key in uncertainties.keys():
            wer, random_wer, percentage, perfect_wer, auc_rr = reject_predictions_wer(refs, hypos, uncertainties[key],
                                                                                      args.path)
            wer_dict[key] = [wer, percentage, random_wer, perfect_wer, auc_rr]
        plt.plot(wer_dict[key][1], wer_dict[key][3], 'k--', lw=2)
        for key in wer_dict.keys():
            plt.plot(wer_dict[key][1], wer_dict[key][0])
        plt.plot(wer_dict[key][1], wer_dict[key][2], 'k--', lw=2)
        plt.ylim(0.0, wer_dict[key][2][0])
        plt.xlim(0.0, 100.0)
        plt.xlabel('Percentage Rejected')
        plt.ylabel('WER (%)')
        plt.legend(['Oracle'] + list(wer_dict.keys()) + ['Expected Random'])
        plt.savefig(os.path.join(args.path, 'seq_reject.png'), bbox_inches='tight', dpi=300)
        plt.close()

        with open(os.path.join(args.path, 'results.txt'), 'a') as f:
            f.write(f'--SEQ ERROR DETECT N-BEST {args.top} --\n')
            for key in wer_dict.keys():
                f.write('AUC-RR using ' + key + ": " + str(np.round(wer_dict[key][-1], 1)) + '\n')
    else:
        refs, hypos = load_text(args.path, nbest=args.nbest)
        uncertainties = load_uncertainties(args.path, n_best=args.nbest, top=args.top)
        bleu_dict = {}
        for key in uncertainties.keys():
            bleus, random_bleu, percentage, perfect_bleu, auc_rr = reject_predictions_bleu(refs, hypos,
                                                                                           uncertainties[key], args.path)
            bleu_dict[key] = [bleus, percentage, random_bleu, perfect_bleu, auc_rr]
        plt.plot(bleu_dict[key][1], bleu_dict[key][3], 'k--', lw=2)
        for key in bleu_dict.keys():
            plt.plot(bleu_dict[key][1], bleu_dict[key][0])
        plt.plot(bleu_dict[key][1], bleu_dict[key][2], 'k--', lw=2)
        plt.ylim(bleu_dict[key][2][0], 100.0)
        plt.xlim(0.0, 100.0)
        plt.xlabel('Percentage Rejected')
        plt.ylabel('Bleu')
        plt.legend(['Oracle'] + list(bleu_dict.keys()) + ['Expected Random'])
        plt.savefig(os.path.join(args.path, 'seq_reject.png'), bbox_inches='tight', dpi=300)
        plt.close()

        with open(os.path.join(args.path, 'results.txt'), 'a') as f:
            f.write(f'--SEQ ERROR DETECT N-BEST {args.top} --\n')
            for key in bleu_dict.keys():
                f.write('AUC-RR using ' + key + ": " + str(np.round(bleu_dict[key][-1], 1)) + '\n')


if __name__ == '__main__':
    main()
