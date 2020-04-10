import argparse

from tqdm import tqdm


def safe_index(toks, index, default):
    try:
        return toks[index]
    except IndexError:
        return default


def main():
    parser = argparse.ArgumentParser(description=(
        'Extract predictions from the stdout of fairseq-generate. '
    ))
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='output prefix')
    parser.add_argument('--srclang', required=True, help='source language (extracted from H-* lines)')
    parser.add_argument('--tgtlang', required=True, help='target language (extracted from S-* lines)')
    args = parser.parse_args()

    tgt_set = set()

    with open(args.output + '.' + args.srclang, 'w') as src_h, \
            open(args.output + '.' + args.tgtlang, 'w') as tgt_h, \
            open(args.input) as inp_h:
        for line in tqdm(inp_h):
            if line.startswith('S-'):
                for tgt in tgt_set:
                    print(src, file=src_h)
                    print(tgt, file=tgt_h)
                src = safe_index(line.rstrip().split('\t'), 1, '')
                tgt_set = set()
            elif line.startswith('H-'):
                tgt_set.add(safe_index(line.rstrip().split('\t'), 2, ''))


if __name__ == '__main__':
    main()
