#!/usr/bin/env python3

"""
Scout training data generation.

$ scout data_gen reads_to_draft.bam draft_error_catalogue_db.txt data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main(args):
    print(args.pileup_bam)
    print(args.error_catalogue)

def argparser():
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    parser.add_argument("pileup_bam", default="")
    parser.add_argument("error_catalogue", default="")
    parser.add_argument("output_data_folder", default="")
    parser.add_argument("--max_error_size", default=5, type=int)

    return parser
