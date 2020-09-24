import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy.stats


def parse_uncertainty_metrics(f, report_nbest=None):
    result = dict()
    for line in f:
        stripped_line = line.strip()
        group_start = re.search(r'N-BEST\s+(\d)', stripped_line)
        if group_start:
            n = int(group_start.group(1))
        else:
            metric_res = re.search(r'using\s*(.*):\s*(-?\d+.\d+)', stripped_line)
            metric = metric_res.group(1)
            value = float(metric_res.group(2))
            if not metric.endswith('EP') and not metric.endswith('ep_MKL') and (report_nbest is None or n in report_nbest):
                result[f'{n}_{metric}'] = value
    return result


def main(decode_path, reference_path, output_path, ensemble_decode_path):
    bleu_stats = dict()
    for testset in ('test', 'test_ens_pred', 'test5_ens_pred', 'test9_ens_pred', 'test12_ens_pred', 'test14_ens_pred'):
        with open(decode_path / f'results-{testset}.txt') as test_preds_f:
            result_line = test_preds_f.read()
            test_bleu = float(re.search(r"BLEU = (\d*.\d*)", result_line).group(1))
            bleu_stats[f'{testset} BLEU'] = test_bleu

    scores = np.loadtxt(reference_path / 'test' / 'score.txt', dtype=np.float64)

    ood_metrics = dict()
    for testset in ['test9', 'test12', 'test14']:
        with open(decode_path / testset / 'results_ood_bs.txt') as f:
            decode_bs_seq = parse_uncertainty_metrics(f, report_nbest=(5,))
        with open(reference_path / testset / 'results_ood_mc.txt') as f:
            ref_bs_seq = parse_uncertainty_metrics(f, report_nbest=(1,))

        ood_metrics.update({
            **{f'ood_{testset}_decode_bs_{key}': value for key, value in decode_bs_seq.items()},
            **{f'ood_{testset}_reference_bs_{key}': value for key, value in ref_bs_seq.items()},
        })

    seq_error_metrics = dict()
    for testset in ['test']:
        with open(decode_path / testset / 'results_seq_mc.txt') as f:
            decode_mc_seq = parse_uncertainty_metrics(f, report_nbest=(1,))

        seq_error_metrics.update({
            **{f'seq_error_{testset}_{key}': value for key, value in decode_mc_seq.items()},
        })

    sequence_spearman_metrics = dict()
    if ensemble_decode_path is not None:
        for testset in ('test', 'test12', 'test14'):
            for metric in ('epkl', 'mkl', 'score', 'entropy_expected', 'score_npmi'):
                try:
                    uncertainties_for_model = np.loadtxt(decode_path / testset / f'{metric}.txt')
                    uncertainties_for_ensemble = np.loadtxt(ensemble_decode_path / testset / f'{metric}.txt')
                except Exception as e:
                    if os.path.exists(ensemble_decode_path / testset):
                        print(os.listdir(ensemble_decode_path / testset))
                    if os.path.exists(decode_path / testset):
                        print(os.listdir(decode_path / testset))
                    print(testset, metric, e)
                    continue
                metric_name = metric if metric != 'score_npmi' else 'smkl'
                spearman_correlation = scipy.stats.spearmanr(uncertainties_for_model, uncertainties_for_ensemble).correlation
                sequence_spearman_metrics[f'spearman_{testset}_{metric_name}'] = spearman_correlation

    result_dict = {
        **bleu_stats,
        'test NLL': scores.mean(),
        **ood_metrics,
        **seq_error_metrics,
        **sequence_spearman_metrics,
    }
    with open(output_path, 'w+') as output_f:
        json.dump(result_dict, output_f, indent=2, sort_keys=True)
        output_f.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--decode-path', required=True, type=Path)
    parser.add_argument('-r', '--reference-path', required=True, type=Path)
    parser.add_argument('-o', '--output-path', required=True, type=Path)
    parser.add_argument('-e', '--ensemble-decode-path', type=Path)
    args = parser.parse_args()
    main(args.decode_path, args.reference_path, args.output_path, args.ensemble_decode_path)
