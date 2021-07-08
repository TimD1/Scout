#!/usr/bin/env python3

"""
Scout training data generation.

$ scout data_gen reads_to_draft.bam draft_error_catalogue_db.txt data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import multiprocessing as mp
import numpy as np
import re, os, sys, toml, pysam

from scout.blocks import *
from scout.util import get_fasta

try: import grouper.config as cfg
except: import scout.config as cfg


def validate(args):

    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
        sys.exit(-1)

    if not os.path.isfile(args.draft_consensus):
        print("ERROR: draft_consensus '{}' does not exist.".format(args.draft_consensus))
        sys.exit(-1)

    if args.error_catalogue and not os.path.isfile(args.error_catalogue):
        print("ERROR: error_catalogue '{}' does not exist.".format(args.error_catalogue))
        sys.exit(-1)

    # set default contig using alignment file
    if args.region_contig == "default":
        calls_to_draft = pysam.AlignmentFile(args.calls_to_draft, "rb")
        args.region_contig = calls_to_draft.references[0]

    # set region to search for errors, ignoring genome start/end
    args.base_radius = (args.base_window-1) // 2
    genome_end = len(get_fasta(args.draft_consensus, args.region_contig))
    if not args.region_end:
        args.region_end = genome_end - args.base_radius
    args.region_end = min(args.region_end, genome_end - args.base_radius)
    args.region_start = max(args.region_start, args.base_radius)

    os.makedirs(args.output_dir, exist_ok=True)
    args.routine = "data_gen"
    cfg.args = args


def main(args):

    # validate arguments and store them
    print("> processing command-line arguments")
    validate(args)

    # get list of positions with high error rate
    cand_positions = get_candidate_positions()

    # get list of ground-truth (polishable) errors using pomoxis error catalogue
    print("> retrieving known variant positions")
    actual_positions = get_variant_positions(args.gold_vcf)
    # actual_positions = get_error_positions(args.error_catalogue)
    
    # generate training dataset
    print("> generating training data")
    blocks, targets = generate_training_data(cand_positions, actual_positions)

    # save training dataset
    print("\n> saving training data")
    argsdict = dict(data_gen = vars(args))
    toml.dump({**argsdict}, open(f'{args.output_dir}/config.toml', 'w'))
    np.save(f'{args.output_dir}/blocks', blocks)
    np.save(f'{args.output_dir}/targets', targets)



def argparser():

    # initialize parser
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    # required arguments
    parser.add_argument("--output_dir")

    # diploid arguments
    parser.add_argument("--gold_vcf")

    # haploid arguments
    parser.add_argument("--calls_to_draft")
    parser.add_argument("--draft_consensus")
    parser.add_argument("--error_catalogue")

    # parameters for training dataset
    parser.add_argument("--base_window", default=41, type=int)
    parser.add_argument("--max_num_blocks", default=100000, type=int)
    parser.add_argument("--max_error_ratio", default=20, type=int)

    # limit search to specific region
    parser.add_argument("--region_contig", default="default")
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)

    # selecting candidate positions
    parser.add_argument("--pileup_min_error", default=0.2, type=float)
    parser.add_argument("--pileup_min_hp", default=0, type=int)
    parser.add_argument("--load_candidates", action="store_true")

    # selecting polish positions
    parser.add_argument("--max_error_size", default=5, type=int)

    return parser
