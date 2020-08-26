#!/usr/bin/env python3

"""
Scout training data generation.

$ scout data_gen reads_to_draft.bam draft_error_catalogue_db.txt data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import multiprocessing as mp
import numpy as np
import re, os, sys, toml
import pysam

# create a module-specific global to attach args to
mod = sys.modules[__name__]

# define globals
test_count = mp.Value('i', 0) # total positions to test
cand_count = mp.Value('i', 0) # candidate positions selected
pos_count = mp.Value('i', 0)  # positions actually selected


def get_candidate_positions(region_start):
    ''' Return list of all positions to generate training data for, based
    on the command-line constraints provided.'''

    # create pysam alignment file
    calls_to_draft = pysam.AlignmentFile(mod.args.calls_to_draft, "rb")

    # calculate region
    center_col_start = region_start + (mod.args.base_window-1)//2
    center_col_end = center_col_start + \
            min(mod.args.region_batch_size, mod.args.region_end-region_start)

    # select positions with sufficient error rate and depth
    positions = []
    for center_col in calls_to_draft.pileup(mod.args.region_contig, \
            center_col_start, center_col_end, min_base_quality=1):
        start_col = center_col.pos - (mod.args.base_window-1)//2

        # only test pileups for positions within region
        if center_col.pos < center_col_start: continue
        if center_col.pos >= center_col_end: break

        # if this is a long homopolymer region, polish it no matter what
        if mod.args.pileup_min_homopolymer:

            # grab reference chunk using read pileup
            for read in center_col.pileups:
                pos = center_col.pos - read.alignment.reference_start
                potential_hp = read.alignment.get_reference_sequence().upper() \
                        [pos : pos + mod.args.pileup_min_homopolymer]
                break

            # check if it's a homopolymer
            if len(potential_hp) == mod.args.pileup_min_homopolymer and \
                    len(set(potential_hp)) == 1:
                with cand_count.get_lock(): 
                    cand_count.value += 1
                with test_count.get_lock(): 
                    test_count.value += 1
                    sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                            .format(cand_count.value, test_count.value,
                                    mod.args.region_end-mod.args.region_start))
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
        if wrong / float(center_col.nsegments) < mod.args.pileup_min_error:
            with test_count.get_lock():
                test_count.value += 1
                sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                        .format(cand_count.value, test_count.value, 
                            mod.args.region_end-mod.args.region_start))
            continue

        with cand_count.get_lock(): 
            cand_count.value += 1
        with test_count.get_lock(): 
            test_count.value += 1
            sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                    .format(cand_count.value, test_count.value,
                            mod.args.region_end-mod.args.region_start))
        positions.append(start_col)

    return positions



def get_error_positions(error_catalogue, max_error_size, start, end):

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
            error_positions.append(pos+1-(mod.args.base_window-1)//2)

    return list(filter(lambda x: x >= start and x < end, error_positions))



def generate_block(ref_start):

    # initialize training data block and encoding scheme
    calls_to_draft = pysam.AlignmentFile(mod.args.calls_to_draft, 'rb')
    block = np.zeros( (mod.args.base_window, 8, 4) )
    ref_end = ref_start + mod.args.base_window
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    feat_idx = {'REF': 0, 'INS': 1, 'DEL': 2, 'SUB': 3}

    for col in calls_to_draft.pileup(mod.args.region_contig, 
            ref_start, ref_end, min_base_quality=1):

        # ignore columns outside of ROI
        if col.pos < ref_start: continue
        if col.pos >= ref_end: break
        pos_offset = col.pos - ref_start

        for read in col.pileups:
            strand_offset = 4 if read.alignment.is_reverse else 0
            ref_base = read.alignment.get_reference_sequence().upper() \
                    [col.pos-read.alignment.reference_start]
            read_base = 'N' if read.query_position is None else \
                    read.alignment.query_sequence[read.query_position]

            # deletion
            if read_base == 'N':
                block[pos_offset][base_idx[ref_base] + strand_offset][feat_idx['DEL']] += 1
            else:

                # insertion
                if read.indel > 0:
                    # add all inserted bases, which are skipped by ref col iteration
                    inserted_bases = read.alignment.query_sequence \
                            [read.query_position+1 : read.query_position+1+read.indel]
                    for ins_base, ins_offset in zip(inserted_bases, range(1,read.indel+1)):
                        # only worry about insertions within ROI
                        if pos_offset + ins_offset >= mod.args.base_window:
                            break
                        block[pos_offset+ins_offset][base_idx[ins_base]+strand_offset][feat_idx['INS']] += 1

                # substitution or correct
                block[pos_offset][base_idx[ref_base]  + strand_offset][feat_idx['REF']] += 1
                block[pos_offset][base_idx[read_base] + strand_offset][feat_idx['INS']] += 1
                block[pos_offset][base_idx[ref_base]  + strand_offset][feat_idx['DEL']] += 1
                block[pos_offset][base_idx[read_base] + strand_offset][feat_idx['SUB']] += 1

    # features are relative to reference counts
    for pos in range(mod.args.base_window):
        for base in range(len(base_idx)*2):
            block[pos][base][feat_idx['INS']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['DEL']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['SUB']] -= block[pos][base][feat_idx['REF']]

    with pos_count.get_lock(): 
        pos_count.value += 1
        sys.stderr.write('{} of {} blocks generated\r' \
                .format(pos_count.value, mod.args.max_num_blocks))

    return block[np.newaxis,:]



def generate_training_data(cand_positions, error_positions):

    # filter any actual errors from the candidate positions
    correct_positions = list(set(cand_positions)-set(error_positions))

    # limit ratio of actual errors to candidate positions for training
    np.random.shuffle(correct_positions)
    correct_positions = correct_positions[:mod.args.max_error_ratio*len(error_positions)]

    # merge actual errors with candidate errors
    positions = np.array(correct_positions + error_positions)
    targets = np.concatenate((np.zeros(len(correct_positions)), np.ones(len(error_positions))))

    # shuffle together
    state = np.random.get_state()
    np.random.set_state(state)
    np.random.shuffle(positions)
    np.random.set_state(state)
    np.random.shuffle(targets)

    # limit data size, keeping track of ground truth
    if len(targets) < mod.args.max_num_blocks:
        print("WARNING: {} blocks requested, generating {}".\
                format(mod.args.max_num_blocks, len(targets)))
        mod.args.max_num_blocks = len(targets)
    positions = positions[:mod.args.max_num_blocks]
    targets = targets[:mod.args.max_num_blocks]

    # generate the blocks for all positions
    block_pool = mp.Pool()
    blocks = list(block_pool.map(generate_block, positions))
    return np.vstack(blocks), targets



def get_fasta(ref_fasta):
    ''' Get length of sequence in FASTA file. '''
    with open(ref_fasta, 'r') as fasta:
        return ''.join(fasta.read().split('\n')[1:])



def validate_arguments():

    if not os.path.isfile(mod.args.calls_to_draft):
        print("ERROR: calls_to_draft '{}' does not exist.".format(mod.args.calls_to_draft))
        sys.exit(-1)

    if not os.path.isfile(mod.args.draft_consensus):
        print("ERROR: draft_consensus '{}' does not exist.".format(mod.args.draft_consensus))
        sys.exit(-1)

    if not os.path.isfile(mod.args.error_catalogue):
        print("ERROR: error_catalogue '{}' does not exist.".format(mod.args.error_catalogue))
        sys.exit(-1)

    if not mod.args.region_end:
        mod.args.region_end = len(get_fasta(mod.args.draft_consensus))

    # keep here for later searches in output folder
    os.makedirs(args.output_data_folder, exist_ok=True)


def main(args):

    # validate arguments and make calculations if necessary
    print("> processing command-line arguments")
    mod.args = args
    validate_arguments()

    # get list of candidate positions with high recall, using pileup heuristics
    cand_positions = None
    if args.use_existing_candidates:
        print("> loading candidate positions")
        candidates_file = os.path.join(args.output_data_folder, 'candidates.npy')
        if os.path.isfile(candidates_file):
            cand_positions = np.load(candidates_file)
        else:
            print("ERROR: candidates file '{}' does not exist.".format(candidates_file))
            sys.exit(-1)
    else:
        print("> finding candidate positions")
        candidate_pool = mp.Pool()
        cand_positions = list(filter(None, candidate_pool.map(get_candidate_positions,
            list(range(args.region_start, args.region_end, args.region_batch_size)))))
        cand_positions = [item for sublist in cand_positions for item in sublist]
        print("\n> saving candidate positions")
        np.save(os.path.join(args.output_data_folder, 'candidates'), cand_positions)

    # get list of ground-truth (polishable) errors using pomoxis error catalogue
    print("> retrieving known error positions")
    error_positions = get_error_positions(args.error_catalogue, 
            args.max_error_size, args.region_start, args.region_end)
    print("{} errors found in range {}-{}".format(len(error_positions), 
            args.region_start, args.region_end))
    
    # generate training dataset
    print("> generating training data")
    blocks, target = generate_training_data(cand_positions, error_positions)

    # save training dataset
    print("\n> saving training data")
    argsdict = dict(data_gen = vars(args))
    toml.dump({**argsdict}, open(os.path.join(args.output_data_folder, 'config.toml'), 'w'))
    np.save(os.path.join(args.output_data_folder, 'blocks'), blocks)
    np.save(os.path.join(args.output_data_folder, 'targets'), target)



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
    parser.add_argument("--max_error_ratio", default=20, type=int)

    # limit search to specific region
    parser.add_argument("--region_contig", default="chromosome")
    parser.add_argument("--region_start", default=0, type=int)
    parser.add_argument("--region_end", default=0, type=int)
    parser.add_argument("--region_batch_size", default=10000, type=int)

    # selecting candidate positions
    parser.add_argument("--pileup_min_error", default=0.3, type=float)
    parser.add_argument("--pileup_min_homopolymer", default=5, type=int)
    parser.add_argument("--use_existing_candidates", action="store_true")

    # selecting polish positions
    parser.add_argument("--max_error_size", default=5, type=int)

    return parser
