#!/usr/bin/env python3

"""
Scout CNN model training.

$ scout train model data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def main(args):
    print(args.model)
    print(args.data_folder)

def argparser():
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    parser.add_argument("model", default="")
    parser.add_argument("data_folder", default="")

    return parser
