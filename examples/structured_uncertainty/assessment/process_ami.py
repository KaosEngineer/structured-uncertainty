import os
import argparse

parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('stm_path', type=str,
                    help='Path to stm file.')
parser.add_argument('wav_path', type=str,
                    help='Path of directory containing wav files.')
parser.add_argument('out_path', type=str,
                    help='Path of directory where to write everything.')


def load_text(path):
    utts = []
    with open(os.path.join(path), 'r') as f:
        for line in f.readlines():
            line = line[0:-1].split()
            utt = {'file': line[0],
                   'spkr': line[2],
                   'start': float(line[3]),
                   'end': float(line[4]),
                   'text': ' '.join(line[5:])
                   }
            utts.append(utt)

    return utts


def main():
    args = parser.parse_args()

    utts = load_text(args.stm_path)

    for utt, i in zip(utts, range(len(utts))):
        file = utt['file']
        start = utt['start']
        duration = utt['end']-start
        tgt_dir = os.path.join(args.out_path, file)
        file_path = os.path.join(args.wav_path, file)
        tgt_name = file+f"-{i}.wav"
        if not os.path.exists(tgt_dir):
            os.mkdir(tgt_dir)
        tgt_path = os.path.join(tgt_dir, tgt_name)
        os.system(f'sox {file_path}.wav {tgt_path} trim {start} {duration}')
        with open(os.path.join(tgt_dir, f'{file}.trans.txt'), 'a') as f:
            f.write(f"{file}-{i} "+utt['text']+'\n')

if __name__ == '__main__':
    main()
