import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Assess ood detection performance')
parser.add_argument('path', type=str, help='Path of directory containing in-domain uncertainties.')

def main():
    args = parser.parse_args()
    data = np.loadtxt(args.path)
    print(f"{args.path.split('/')[-1]}, {np.round(np.mean(data), 1)} tiny{{$\pm$ {np.round(2.0*np.std(data), 1)}}}")

if __name__ == '__main__':
    main()