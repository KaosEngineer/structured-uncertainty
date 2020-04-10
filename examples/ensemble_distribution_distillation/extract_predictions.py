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
        'Keep all hypotheses '
    ))
    parser.add_argument('--output', required=True, help='output prefix')
    parser.add_argument('--srclang', required=True, help='source language (extracted from H-* lines)')
    parser.add_argument('--tgtlang', required=True, help='target language (extracted from S-* lines)')
    parser.add_argument('files', nargs='*', help='input files')
    args = parser.parse_args()

    with open(args.output + '.' + args.srclang, 'w') as src_h, \
            open(args.output + '.' + args.tgtlang, 'w') as tgt_h, \
            open(args.input) as inp_h:
        for line in tqdm(inp_h):
            if line.startswith('S-'):
                src = safe_index(line.rstrip().split('\t'), 1, '')
            elif line.startswith('H-'):
                if src is not None:
                    tgt = safe_index(line.rstrip().split('\t'), 2, '')
                    print(src, file=src_h)
                    print(tgt, file=tgt_h)


if __name__ == '__main__':
    main()
