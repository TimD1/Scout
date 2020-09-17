#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model_dir
"""

import sys, time, torch, os, pysam
from datetime import timedelta 
import matplotlib.pyplot as plt
import numpy as np 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from scout.util import load_model, get_fasta
from scout.blocks import *

try: import grouper.config as cfg
except: import scout.config as cfg


def validate(args):

    print("> processing command-line arguments")

    # check that alignment file exists
    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: --calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
        sys.exit(-1)

    # set contig default using alignment file
    if args.region_contig == "default":
        calls_to_draft = pysam.AlignmentFile(args.calls_to_draft, "rb")
        args.region_contig = calls_to_draft.references[0]

    # set region to search for errors, ignoring genome start/end
    args.base_radius = (args.base_window-1) // 2
    genome_end = len(get_fasta(args.draft_consensus))
    if not args.region_end:
        args.region_end = genome_end - args.base_radius
    args.region_end = min(args.region_end, genome_end - args.base_radius)
    args.region_start = max(args.region_start, args.base_radius)

    # check position selection method
    methods = ["pileup", "medaka", "scout", "pileup_scout", "sweep_racon", "sweep_medaka"]
    if args.method not in methods:
        print("ERROR: --method must be one of {}, not '{}'".format(
            methods, args.method))
    if args.method == "medaka" or args.method == "sweep_medaka" and args.medaka_hdf5_out is None:
        print("ERROR: must provide --medaka_hdf5_out file with --method '{}'".format(args.method))
        sys.exit(-1)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.args = args



def count_useful(cand_pos, error_pos):
    ''' 
    Given lists of candidate and actual error positions, returns the number of 
    errors polished and number of useful positions.

    These values are not necessarily the same because an error can be identified 
    by calling a nearby base an error. Stats are dependent upon how many 
    re-called bases Grouper will merge into the updated assembly.
    '''
    err_idx = 0
    pol_idx = 0
    polished_errors = 0
    good_candidates = 0
    radius = (cfg.args.merge_center - 1) // 2

    # iterate over each polished position
    while pol_idx < len(cand_pos):

        # no more errors remaining, all remaining polish regions are incorrect
        if err_idx >= len(error_pos):
            break

        # we missed a position we should have polished
        if error_pos[err_idx] < cand_pos[pol_idx] - radius:
            err_idx += 1

        # we successfully polished some errors
        elif error_pos[err_idx] >= cand_pos[pol_idx] - radius and \
                error_pos[err_idx] <= cand_pos[pol_idx] + radius:

            # this chosen position is correct, we polished at least one error
            good_candidates += 1
            err_idx += 1
            polished_errors += 1
            pol_idx += 1

            # count how many errors we polished
            if err_idx >= len(error_pos): break
            while error_pos[err_idx] <= cand_pos[pol_idx-1]+radius:
                err_idx += 1
                polished_errors += 1
                if err_idx >= len(error_pos): break

        # this polish position didn't fix any errors
        else:
            pol_idx += 1

    return polished_errors, good_candidates



def print_stats(error_pos, polish_pos, cand_pos=None):
    '''
    Prints prec-recall statistics for identifying errors which need polishing.

    Allows an optional two-step filtering process, identifying candidate 
    positions prior to using Scout to narrow polish positions further.
    '''

    # count useful positions and errors retained
    polish_errors, good_polish_pos = count_useful(polish_pos, error_pos)
    actual_errors = len(error_pos)

    if cand_pos is not None:
        cand_errors, good_cand_pos = count_useful(cand_pos, error_pos)
        print("\nCANDIDATE SELECTION [err: {}\thp: {}]".format(
            cfg.args.pileup_min_error, cfg.args.pileup_min_hp))
        print("recall:\t{:.2f}% ({} of {})".format(
            cand_errors*100.0/actual_errors, cand_errors, actual_errors))
        print("prec:\t{:.2f}% ({} of {})".format(
            good_cand_pos*100.0/len(cand_pos), good_cand_pos, len(cand_pos)))
        print("keep:\t{:.2f}% of total ({}*{} of {})".format(
            len(cand_pos)*cfg.args.merge_center*100.0 / (cfg.args.region_end-cfg.args.region_start),
            len(cand_pos), cfg.args.merge_center, cfg.args.region_end-cfg.args.region_start
        ))

        print("\nPOLISHING SELECTION [model: {}]".format(cfg.args.model_dir))
        print("recall:\t{:.2f}% ({} of {})".format(
            polish_errors*100.0/cand_errors, polish_errors, cand_errors))
        print("prec:\t{:.2f}% ({} of {})".format(
            good_polish_pos*100.0/len(polish_pos), good_polish_pos, len(polish_pos)))
        print("keep:\t{:.2f}% of candidates ({} of {})".format(
            len(polish_pos)*100.0 / len(cand_pos),
            len(polish_pos), len(cand_pos)
        ))

    print("\nOVERALL SELECTION")
    print("recall:\t{:.2f}% ({} of {})".format(
        polish_errors*100.0/actual_errors, polish_errors, actual_errors))
    print("prec:\t{:.2f}% ({} of {})".format(
        good_polish_pos*100.0/len(polish_pos), good_polish_pos, len(polish_pos)))
    print("keep:\t{:.2f}% of total ({}*{} of {})".format(
        len(polish_pos)*cfg.args.merge_center*100.0 / (cfg.args.region_end-cfg.args.region_start),
        len(polish_pos), cfg.args.merge_center, cfg.args.region_end-cfg.args.region_start
    ))
    print(" ")


def main(args):

    validate(args)
    error_positions = get_error_positions(args.error_catalogue)
    print("> evaluating '{}'".format(args.method))

    if args.method == "scout":
        # still have to write this, scout on entire genome
        print("WARNING: not implemented yet...")

    elif args.method == "pileup_scout":
        # scout with pileup pre-filtering
        cand_positions = get_candidate_positions()
        model = load_model(args.model_dir, args.device, 
                weights=int(args.weights), half=args.half)
        error_probs = get_pileup_scout_error_probs(cand_positions, model, args.device)
        polish_positions = np.array(cand_positions)[error_probs > args.threshold]
        print_stats(error_positions, polish_positions, cand_positions)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xticks(range(0,101,10))
        ax.set_yticks(range(0,101,10))
        recall, prec = [], []
        thresholds = 1 / (1 + np.exp(-np.linspace(-10, 10, 101)))
        for threshold in thresholds:
            polish_pos = np.array(cand_positions)[error_probs > threshold]
            polish_errors, good_polish_pos = count_useful(polish_pos, error_positions)
            recall.append(100 if not len(error_positions) else polish_errors*100.0 / len(error_positions))
            prec.append(100 if not len(polish_pos) else good_polish_pos*100.0 / len(polish_pos))
        ax.plot(recall, prec, label="pileup_scout")
        ax.legend()
        fig.savefig(os.path.join(cfg.args.output_dir, "prec_recall.png"))

    elif args.method == "pileup":
        # pileup based position selection using error rate and hp length
        polish_positions = get_candidate_positions()
        print_stats(error_positions, polish_positions)

    elif args.method == "medaka":
        # choose positions based on medaka's output confidence
        polish_positions = get_medaka_positions(args.medaka_hdf5_out)
        print_stats(error_positions, polish_positions)

    elif args.method == "sweep_racon" or args.method == "sweep_medaka":

        # initialize graph
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xticks(range(0,101,10))
        ax.set_yticks(range(0,101,10))

        print("> now evaluating 'pileup_scout'")
        prec, recall = [], []
        pileup_positions = get_candidate_positions()
        print_stats(error_positions, pileup_positions)
        model = load_model(args.model_dir, args.device, 
                weights=int(args.weights), half=args.half)
        error_probs = get_pileup_scout_error_probs(pileup_positions, model, args.device)
        thresholds = 1 / (1 + np.exp(-np.linspace(-10, 10, 101)))
        for threshold in thresholds:
            polish_pos = np.array(pileup_positions)[error_probs > threshold]
            polish_errors, good_polish_pos = count_useful(polish_pos, error_positions)
            recall.append(100 if not len(error_positions) else polish_errors*100.0 / len(error_positions))
            prec.append(100 if not len(polish_pos) else good_polish_pos*100.0 / len(polish_pos))
        ax.plot(recall, prec, label="pileup_scout")

        if args.method == "sweep_medaka":
            print("> now evaluating 'medaka'")
            prec, recall = [], []
            thresholds = np.linspace(0, 50, 21)
            for idx, threshold in enumerate(thresholds):
                print("iteration {} of {}\r".format(idx+1, len(thresholds)), end="")
                cfg.args.max_qscore_medaka = threshold
                polish_pos = get_medaka_positions(args.medaka_hdf5_out)
                polish_errors, good_polish_pos = count_useful(polish_pos, error_positions)
                recall.append(100 if not len(error_positions) else polish_errors*100.0 / len(error_positions))
                prec.append(100 if not len(polish_pos) else good_polish_pos*100.0 / len(polish_pos))
            ax.plot(recall, prec, label="medaka")

        print("> now evaluating 'pileup'")
        prec, recall = [], []
        thresholds = np.linspace(0.1, 1, 19)
        for threshold in thresholds:
            cfg.args.pileup_min_error = threshold
            polish_pos = get_candidate_positions()
            polish_errors, good_polish_pos = count_useful(polish_pos, error_positions)
            recall.append(100 if not len(error_positions) else polish_errors*100.0 / len(error_positions))
            prec.append(100 if not len(polish_pos) else good_polish_pos*100.0 / len(polish_pos))
        ax.plot(recall, prec, label="pileup")

        ax.legend()
        fig.savefig(os.path.join(cfg.args.output_dir, "prec_recall.png"))



def argparser():

    # initialize parser
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    # i/o arguments
    parser.add_argument("calls_to_draft")
    parser.add_argument("draft_consensus")
    parser.add_argument("error_catalogue")
    parser.add_argument("output_dir")

    # limit search to specific region
    parser.add_argument("--region_contig", default="default",)
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)
    parser.add_argument("--max_error_size", default=5, type=int)

    # selecting candidate positions
    parser.add_argument("--method", default="pileup_scout")
    parser.add_argument("--base_window", default=21, type=int)
    parser.add_argument("--merge_center", default=3, type=int)
    parser.add_argument("--pileup_min_error", default=0.2, type=float)
    parser.add_argument("--pileup_min_hp", default=5, type=int)
    parser.add_argument("--use_existing_candidates", action="store_true")
    parser.add_argument("--max_qscore_medaka", default=15, type=float)
    parser.add_argument("--medaka_hdf5_out")

    # model arguments
    parser.add_argument("--model_dir", default="dna_r941_21bp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--half", action="store_true", default=False)

    return parser
