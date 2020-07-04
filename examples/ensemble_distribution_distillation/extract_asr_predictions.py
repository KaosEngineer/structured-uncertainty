import argparse
import json
import re

from fairseq.data import Dictionary


def main():
    parser = argparse.ArgumentParser(description=(
        'Extract deduplicated predictions from the stdout of fairseq-generate. '
    ))
    parser.add_argument('--words-path', required=True)
    parser.add_argument('--units-path', required=True)
    parser.add_argument('--dict-path', required=True)
    parser.add_argument('--orig-path', required=True)
    parser.add_argument('--result-path', required=True)
    args = parser.parse_args()

    # Set dictionary
    tgt_dict = Dictionary.load(args.dict_path)

    with open(args.orig_path) as f:
        data_samples = json.load(f)

    new_words = dict()
    new_units = dict()
    new_tokens = dict()

    # OCEAN REIGNED SUPREME (5105_28241-5105-28241-0010)
    word_pattern = re.compile(r'(.*)\s\(\d*_\d*-(.*)\)')

    with open(args.words_path) as words_fh:
        for line in words_fh:
            m = word_pattern.match(line.strip())
            if m:
                words = m.group(1)
                id = m.group(2)

                new_words[id] = words + '\n'
            else:
                print(f'Matching failed for {line.strip()}')

    with open(args.units_path) as units_fh:
        for line in units_fh:
            m = word_pattern.match(line.strip())
            if m:
                units = m.group(1)
                id = m.group(2)
                if id in new_words:
                    # '▁OCEAN' -> \u2581OCEAN
                    new_units[id] = units
                    tokens_tensor = tgt_dict.encode_line(units, append_eos=False)
                    new_tokens[id] = ', '.join(map(str, tokens_tensor.tolist()))
                else:
                    print(f'id {id} was not present in new_words')
            else:
                print(f'Matching failed for {line.strip()}')

    for id, data in data_samples['utts'].items():
        data['output']['text'] = new_words[id]
        data['output']['token'] = new_units[id]
        data['output']['tokenid'] = new_tokens[id]

    with open(args.result_path, 'w+') as f:
        json.dump(data_samples, f, indent=4)


if __name__ == '__main__':
    main()
