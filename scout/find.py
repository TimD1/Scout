#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main(args):
    print(args.pileup_bam)
    print(args.model)

def argparser():
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    parser.add_argument("pileup_bam", default="")
    parser.add_argument("model", default="")

    return parser
