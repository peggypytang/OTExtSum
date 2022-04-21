""" Evaluate the baselines ont ROUGE/METEOR"""
import argparse
import json
import os
from os.path import join, exists

from evaluate import eval_meteor, eval_rouge


def main(args):
    dec_dir = join(args.decode_dir, args.decode_folder)
    print("dec_dir", dec_dir)
    split = 'test'
    ref_dir = join(args.ref_dir, split)
    assert exists(ref_dir)

    if args.rouge:
        dec_pattern = r'(\d+).dec'
        if args.multi_ref:
            ref_pattern = '#ID#.[A-Z].ref'
        else:
            ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'rouge'
    else:
        dec_pattern = '[0-9]+.dec'
        if args.multi_ref:
            ref_pattern = '[0-9]+.A.ref'
        else:
            ref_pattern = '[0-9]+.ref'
        output = eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir)
        metric = 'meteor'
    print(output)
    with open(join(args.decode_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the output files for the RL full models')

    # choose metric to evaluate
    metric_opt = parser.add_mutually_exclusive_group(required=True)
    metric_opt.add_argument('--rouge', action='store_true',
                            help='ROUGE evaluation')
    metric_opt.add_argument('--meteor', action='store_true',
                            help='METEOR evaluation')
    parser.add_argument('--ref_dir', action='store', required=True,
                        help='directory of ref summaries')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--decode_folder', action='store', required=True,
                        help='folder of decoded summaries')
    parser.add_argument('--multi_ref', action='store', default=False,
                        help='Multiple or single reference summaries')

    args = parser.parse_args()
    main(args)
