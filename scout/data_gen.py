#!/usr/bin/env python3

"""
Scout training data generation.

$ scout data_gen reads_to_draft.bam draft_error_catalogue_db.txt data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import multiprocessing as mp
import numpy as np
import re, os, sys


# define global counters
test_count = mp.Value('i', 0) # total positions to test
cand_count = mp.Value('i', 0) # candidate positions selected
pos_count = mp.Value('i', 0)  # positions actually selected


def get_candidate_positions(region_start):
    ''' Return list of all positions to generate training data for, based
    on the command-line constraints provided.'''

    # create pysam alignment file
    calls_to_draft = pysam.AlignmentFile(args.calls_to_draft, "rb")

    # calculate region
    center_col_start = region_start + (args.base_window-1)//2
    center_col_end = center_col_start + \
            min(args.region_batch_size, args.region_end-region_start)

    # select positions with sufficient error rate and depth
    positions = []
    for center_col in calls_to_draft.pileup(args.region_contig, \
            center_col_start, center_col_end, min_base_quality=1):
        start_col = center_col.pos - (args.base_window-1)//2

        # only test pileups for positions within region
        if center_col.pos < center_col_start: continue
        if center_col.pos >= center_col_end: break

        # if this is a long homopolymer region, polish it no matter what
        if args.pileup_min_homopolymer:

            # grab reference chunk using read pileup
            for read in center_col.pileups:
                pos = center_col.pos - read.alignment.reference_start
                potential_hp = read.alignment.get_reference_sequence().upper() \
                        [pos : pos + args.pileup_min_homopolymer]
                break

            # check if it's a homopolymer
            if len(potential_hp) == args.pileup_min_homopolymer and \
                    len(set(potential_hp)) == 1:
                if args.print_progress:
                    with cand_count.get_lock(): 
                        cand_count.value += 1
                    with test_count.get_lock(): 
                        test_count.value += 1
                        sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                                .format(cand_count.value, test_count.value,
                                        args.region_end-args.region_start))
                positions.append(start_col)
                continue

        # check percentage of reads which err at this position
        wrong = 0
        for read in center_col.pileups:
            ref_base = read.alignment.get_reference_sequence().upper() \
                    [center_col.pos-read.alignment.reference_start]
            read_base = 'N' if read.query_position is None else \
                    read.alignment.query_sequence[read.query_position]
            if read_base != ref_base or read.indel > 0: 
                wrong += 1

        # if low error rate, continue
        if wrong / float(center_col.nsegments) < args.pileup_min_error:
            if args.print_progress:
                with test_count.get_lock():
                    test_count.value += 1
                    sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                            .format(cand_count.value, test_count.value, 
                                args.region_end-args.region_start))
            continue

        if args.print_progress:
            with cand_count.get_lock(): 
                cand_count.value += 1
            with test_count.get_lock(): 
                test_count.value += 1
                sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                        .format(cand_count.value, test_count.value,
                                args.region_end-args.region_start))
        positions.append(start_col)

    return positions



def get_error_positions(error_catalogue, max_error_size):

    # parse the error catalogue, creating a list of all SNP positions
    error_positions = []
    with open(error_catalogue, 'r') as error_stats_file:
        header = True
        for line in error_stats_file:

            # skip first line of error database
            if header: header = False; continue

            # for each error, parse the important information
            fields = re.split(r'\t+', line)
            chunk = fields[3]
            chunk_id = int(re.findall(r'\d+$', chunk)[0])
            pos = int(fields[4].strip('~')) + chunk_id * 100000

            # we can't polish large structural variants, so don't even try
            errs = max(int(fields[8]), int(fields[9]), int(fields[10]))
            if errs > max_error_size: continue

            # append the start of this region to list of error positions
            error_positions.append(pos+1-(args.base_window-1)//2)

    return error_positions



def generate_block(ref_start):

    calls_to_draft = pysam.AlignmentFile(args.calls_to_draft, 'rb')
    block = np.zeros( (args.base_window, 8, 4) )
    ref_end = ref_start + args.base_window

    for col in calls_to_draft.pileup(args.region_contig, 
            ref_start, ref_end, min_base_quality=1):

        if col < ref_start: continue
        if col >= ref_end: break

        for read in col.pileups:
            ref_base = read.alignment.get_reference_sequence().upper() \
                    [center_col.pos-read.alignment.reference_start]
            read_base = 'N' if read.query_position is None else \
                    read.alignment.query_sequence[read.query_position]
            if read_base != ref_base or read.indel > 0:
                wrong += 1
                





def generate_training_data(cand_positions, error_positions):

    # filter any actual errors from the candidate positions
    not_error_positions = np.setdiff1d(cand_positions - error_positions)

    # limit ratio of actual errors to candidate positions for training
    np.random.shuffle(not_error_positions)
    not_error_positions = not_error_positions[:args.max_error_ratio*len(error_positions)]

    # merge actual errors with candidate errors
    positions = np.array(not_error_positions + error_positions)
    truths = np.concatenate(np.zeros(len(not_error_positions)), np.ones(len(error_positions)))

    # shuffle together and limit examples, keeping track of ground truth
    state = np.random.get_state()
    for array in (positions, truths):
        np.random.set_state(state)
        np.random.shuffle(array)
        array = array[:args.max_num_blocks]

    # generate the blocks for all positions
    block_pool = mp.Pool()
    blocks = list(block_pool.map(generate_block, positions))
    valid_indices = [ block is not None for block in blocks ]
    return blocks[valid_indices], truths[valid_indices]



def get_fasta(ref_fasta):
    ''' Get length of sequence in FASTA file. '''
    with open(ref_fasta, 'r') as fasta:
        return ''.join(fasta.read().split('\n')[1:])



def validate_arguments():

    if not os.path.isfile(args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(args.calls_to_draft))
        sys.exit(-1)

    if not os.path.isfile(args.draft_consensus):
        print("ERROR: draft_consensus '{}' does not exist.".format(args.draft_consensus))
        sys.exit(-1)

    if not os.path.isfile(args.error_catalogue):
        print("ERROR: error_catalogue '{}' does not exist.".format(args.error_catalogue))
        sys.exit(-1)

    if not args.region_end:
        args.region_end = len(get_fasta(args.draft_consensus))


def main(args):

    # validate arguments and make calculations if necessary
    print("> processing command-line arguments")
    validate_arguments()

    # get list of candidate positions with high recall, using pileup heuristics
    print("> finding candidate positions")
    cand_positions = None
    if args.use_existing_candidates:
        cand_positions = np.load(os.path.join(args.output_data_folder, 'candidates'))
    else
        candidate_pool = mp.Pool()
        cand_positions = list(filter(None, candidate_pool.map(get_candidate_positions,
            list(range(args.region_start, args.region_end, args.region_batch_size)))))
        print("> saving candidate positions")
        np.save(os.path.join(args.output_data_folder, 'candidates'), cand_positions)

    # get list of ground-truth (polishable) errors using pomoxis error catalogue
    print("> retrieving known error positions")
    error_positions = get_error_positions(args.error_catalogue, args.max_error_size)
    
    # generate training dataset
    print("> generating training data")
    blocks, truth = generate_training_data(cand_positions, error_positions)

    # save training dataset
    print("> saving training data")
    np.save(os.path.join(args.output_data_folder, 'blocks'), blocks)
    np.save(os.path.join(args.output_data_folder, 'truth'), truth)



def argparser():

    # initialize parser
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    # required arguments
    parser.add_argument("calls_to_draft", default="")
    parser.add_argument("draft_consensus", default="")
    parser.add_argument("error_catalogue", default="")
    parser.add_argument("output_data_folder", default="")

    # parameters for training dataset
    parser.add_argument("--base_window", default=21, type=int)
    parser.add_argument("--max_num_blocks", default=100000, type=int)
    parser.add_argument("--max_error_ratio", default=20, type=float)

    # limit search to specific region
    parser.add_argument("--region_contig", default="chromosome")
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)

    # selecting candidate positions
    parser.add_argument("--pileup_min_error", default=0.1, type=float)
    parser.add_argument("--pileup_min_homopolymer", default=0.1, type=float)
    parser.add_argument("--use_existing_candidates", action="store_true")

    # selecting polish positions
    parser.add_argument("--max_error_size", default=5, type=int)

    return parser
