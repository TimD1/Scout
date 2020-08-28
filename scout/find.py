#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model_dir
"""

import sys, time, torch
from datetime import timedelta 

import numpy as np 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import scout.config as cfg
from scout.util import load_model, get_fasta
from scout.blocks import *


def validate(args):

    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
        sys.exit(-1)

    if not args.region_end:
        args.region_end = len(get_fasta(args.draft_consensus))

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.args = args



def main(args):

    # validate arguments and store them
    print("> processing command-line arguments")
    validate(args)

    # load model from disk
    print("> loading model")
    model = load_model(args.model_directory, args.device, 
            weights=int(args.weights), half=args.half)

    # pre-filtering using basic pileup heuristics
    cand_positions = get_candidate_positions()

    # use model to select positions to polish
    chosen_positions = choose_positions(cand_positions, model, args.device)

    print("\n>saving results")
    np.save(os.path.join(args.output_dir, "positions"), chosen_positions)



def argparser():

    # initialize parser
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    # i/o arguments
    parser.add_argument("calls_to_draft")
    parser.add_argument("draft_consensus")
    parser.add_argument("output_dir")

    # limit search to specific region
    parser.add_argument("--region_contig", default="chromosome",)
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)

    # selecting candidate positions
    parser.add_argument("--base_window", default=21, type=int)
    parser.add_argument("--pileup_min_error", default=0.3, type=float)
    parser.add_argument("--pileup_min_hp", default=5, type=int)
    parser.add_argument("--use_existing_candidates", action="store_true")

    # model arguments
    parser.add_argument("--model_directory", default="dna_r941_21bp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--half", action="store_true", default=False)

    return parser
