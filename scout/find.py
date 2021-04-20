#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model_dir
"""

import sys, time, torch, pysam
from datetime import timedelta 

import numpy as np 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scout.util import load_model, get_fasta
from scout.blocks import *

try: import grouper.config as cfg
except: import scout.config as cfg


def validate(args):

    print("> processing command-line arguments")

    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
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
    cfg.args = args



def main(args):

    # validate arguments and store them
    validate(args)

    # load model from disk
    model = load_model(args.model_dir, args.device, 
            weights=int(args.weights), half=args.half)

    # pre-filtering using basic pileup heuristics
    cand_positions = get_candidate_positions()

    # use model to select positions to polish
    error_probs = get_pileup_scout_error_probs(cand_positions, model, args.device)
    chosen_positions = np.array(cand_positions)[error_probs > args.threshold]

    print("\n> saving results")
    argsdict = dict(find = vars(args))
    train_config = toml.load(os.path.join(args.model_dir, "config.toml"))
    toml.dump({**argsdict, **train_config}, open(os.path.join(args.output_dir, "config.toml"), "w"))
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
    parser.add_argument("--region_contig", default="default",)
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)

    # selecting candidate positions
    parser.add_argument("--base_window", default=41, type=int)
    parser.add_argument("--pileup_min_error", default=0.2, type=float)
    parser.add_argument("--pileup_min_hp", default=0, type=int)
    parser.add_argument("--use_existing_candidates", action="store_true")

    # model arguments
    parser.add_argument("--model_dir", default="dna_r941_41bp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--half", action="store_true", default=False)

    return parser
