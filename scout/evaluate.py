#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model_dir
"""

import sys, time, torch, os
from datetime import timedelta 
import matplotlib.pyplot as plt
import numpy as np 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import scout.config as cfg
from scout.util import load_model, get_fasta
from scout.blocks import *


def validate(args):

    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
        sys.exit(-1)

    # set region to search for errors, ignoring genome start/end
    args.base_radius = (args.base_window-1) // 2
    genome_end = len(get_fasta(args.draft_consensus))
    if not args.region_end:
        args.region_end = genome_end - args.base_radius
    args.region_end = min(args.region_end, genome_end - args.base_radius)
    args.region_start = max(args.region_start, args.base_radius)

    os.makedirs(args.output_dir, exist_ok=True)
    cfg.args = args



def count_useful(cand_pos, error_pos):
    ''' 
    Returns the number of errors polished and number of useful positions retained.
    '''
    err_idx = 0
    pol_idx = 0
    polished_error = 0
    good_cand_pos = 0
    radius = (cfg.args.merge_center - 1) // 2

    # iterate over each polished position
    while pol_idx < len(cand_pos):

        # no more errors remaining, all remaining polish regions are incorrect
        if err_idx >= len(error_pos):
            break

        # we missed a position we should have polished
        if error_pos[err_idx] < cand_pos[pol_idx]:
            err_idx += 1

        # we successfully polished some errors
        elif error_pos[err_idx] >= cand_pos[pol_idx]-radius and \
                error_pos[err_idx] <= cand_pos[pol_idx]+radius:

            # this chosen position is correct, we polished at least one error
            good_cand_pos += 1
            err_idx += 1
            polished_error += 1
            pol_idx += 1

            # count how many errors we polished
            if err_idx >= len(error_pos): break
            while error_pos[err_idx] <= cand_pos[pol_idx-1]+radius:
                err_idx += 1
                polished_error += 1
                if err_idx >= len(error_pos): break

        # this polish position didn't fix any errors
        else:
            pol_idx += 1

    return polished_error, good_cand_pos



def plot_prec_recall(cand_pos, error_probs, error_pos):

    prec = []
    recall = []
    thresholds = np.linspace(0, 1, 101)
    cand_errors, good_cand_pos = count_useful(cand_pos, error_pos)
    for threshold in thresholds:
        polish_pos = np.array(cand_pos)[error_probs > threshold]
        polish_errors, good_polish_pos = count_useful(polish_pos, error_pos)
        recall.append(100 if not cand_errors else polish_errors*100.0 / cand_errors)
        prec.append(100 if not len(polish_pos) else good_polish_pos*100.0 / len(polish_pos))

    plt.plot(recall, prec)
    plt.grid(True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xticks(range(0,101,10))
    plt.yticks(range(0,101,10))
    plt.savefig(os.path.join(cfg.args.output_dir, "prec_recall.png"))




def print_stats(error_pos, cand_pos, polish_pos):

    # count useful positions and errors retained
    cand_errors, good_cand_pos = count_useful(cand_pos, error_pos)
    polish_errors, good_polish_pos = count_useful(polish_pos, error_pos)
    actual_errors = len(error_pos)

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

    # validate arguments and store them
    print("> processing command-line arguments")
    validate(args)

    # load model from disk
    print("> loading model")
    model = load_model(args.model_dir, args.device, 
            weights=int(args.weights), half=args.half)

    # get error positions
    print("> loading known error positions")
    error_positions = get_error_positions(args.error_catalogue, 
            args.max_error_size, args.region_start, args.region_end)

    # pre-filtering using basic pileup heuristics
    cand_positions = get_candidate_positions()

    # use model to select positions to polish
    error_probs = call_error_probs(cand_positions, model, args.device)
    polish_positions = np.array(cand_positions)[error_probs > args.threshold]

    # print statistics to evaluate candidate heuristic and model
    print("> calculating summary statistics")
    plot_prec_recall(cand_positions, error_probs, error_positions)
    print_stats(error_positions, cand_positions, polish_positions)



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
    parser.add_argument("--region_contig", default="chromosome",)
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)
    parser.add_argument("--max_error_size", default=5, type=int)

    # selecting candidate positions
    parser.add_argument("--base_window", default=21, type=int)
    parser.add_argument("--merge_center", default=3, type=int)
    parser.add_argument("--pileup_min_error", default=0.3, type=float)
    parser.add_argument("--pileup_min_hp", default=5, type=int)
    parser.add_argument("--use_existing_candidates", action="store_true")

    # model arguments
    parser.add_argument("--model_dir", default="dna_r941_21bp")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--half", action="store_true", default=False)

    return parser
