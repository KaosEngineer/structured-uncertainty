import numpy as np
import os
from sacrebleu import corpus_bleu, sentence_bleu
import argparse
import subprocess
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str,
                    help='Path of directory containing in-domain uncertainties.')
parser.add_argument('n_models', type=int,
                    help='Number of models to evaluate')
parser.add_argument('--wer', action='store_true',
                    help='Whether to evaluate using WER instead of BLEU')


def load_text(path, m):
    hypos = []
    with open(os.path.join(path, f'hypos_{m}.txt'), 'r') as f:
        for line in f.readlines():
            hypos.append(line[:-1])

    return hypos

def get_sentence_wer(path, i, j):
    #if not os.path.exists(f"{path}/wers.txt"):
    subprocess.run(f"~/sclite -r {path}/whypos_{i}.txt -h {path}/whypos_{j}.txt -i rm -o all", shell=True)
    subprocess.run(
        f" grep -v '-'  {path}/whypos_{j}.txt.sys | egrep -v '[A-Z]' | egrep -v '=' | sed 's/|//g' | egrep '[0-9]+' | awk '{{print $8}}' > {path}/wers.txt",
        shell=True)

    return np.loadtxt(f"{path}/wers.txt", dtype=np.float32)

def main():
    args = parser.parse_args()

    if not args.wer:
        hypo_dict = {}
        for m in range(args.n_models):
            hypos = load_text(args.path, m)
            for hypo in hypos:
                hypo = hypo.split()
                if hypo[0] not in hypo_dict.keys():
                    hypo_dict[hypo[0]] = [' '.join(hypo[1:])]
                else:
                    hypo_dict[hypo[0]].append(' '.join(hypo[1:]))

        cross_bleu_dict = {}

        for key in hypo_dict.keys():
            counter = 0.0
            bleu = 0.0
            for i in range(args.n_models):
                for j in range(args.n_models):
                    if i != j:
                        try:
                            bleu += (100.0 - sentence_bleu(hypo_dict[key][i], hypo_dict[key][j]).score)**2
                        except:
                            print(m, i, j, key, args.path)
                        counter += 1.0

            cross_bleu_dict[key] = np.round(bleu / counter, 1)

        with open(os.path.join(args.path, 'cross_bleu.txt'), 'w') as f:
            for key in cross_bleu_dict.keys():
                f.write(f'{key}\t{cross_bleu_dict[key]}\n')
    else:
        wer = []
        for i in range(args.n_models):
            for j in range(args.n_models):
                if i != j:
                    try:
                        wer.append(get_sentence_wer(args.path, i, j) ** 2)
                    except:
                        print(i, j, args.path)
        wer = np.mean(np.stack(wer, axis=0), axis=0)

        with open(os.path.join(args.path, 'cross_wer_tmp.txt'), 'w') as f:
            for w, i in zip(wer, range(wer.shape[0])):
                f.write(f'{w}\n')


if __name__ == '__main__':
    main()
