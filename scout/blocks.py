import multiprocessing as mp
import numpy as np
import re, os, sys, toml
import pysam, torch

import scout.config as cfg
from scout.util import *


def find_candidate_positions(region_start):
    ''' Return list of all positions to generate training data for, based
    on the command-line constraints provided.'''

    # create pysam alignment file
    calls_to_draft = pysam.AlignmentFile(cfg.args.calls_to_draft, "rb")

    # calculate region
    center_col_start = region_start + (cfg.args.base_window-1)//2
    center_col_end = center_col_start + \
            min(cfg.args.region_batch_size, cfg.args.region_end-region_start)

    # select positions with sufficient error rate and depth
    positions = []
    for center_col in calls_to_draft.pileup(cfg.args.region_contig, \
            center_col_start, center_col_end, min_base_quality=1):
        start_col = center_col.pos - (cfg.args.base_window-1)//2

        # only test pileups for positions within region
        if center_col.pos < center_col_start: continue
        if center_col.pos >= center_col_end: break

        # if this is a long hp region, polish it no matter what
        if cfg.args.pileup_min_hp:

            # grab reference chunk using read pileup
            for read in center_col.pileups:
                pos = center_col.pos - read.alignment.reference_start
                potential_hp = read.alignment.get_reference_sequence().upper() \
                        [pos : pos + cfg.args.pileup_min_hp]
                break

            # check if it's a homopolymer
            if len(potential_hp) == cfg.args.pileup_min_hp and \
                    len(set(potential_hp)) == 1:
                with cfg.cand_count.get_lock(): 
                    cfg.cand_count.value += 1
                with cfg.test_count.get_lock(): 
                    cfg.test_count.value += 1
                    sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                            .format(cfg.cand_count.value, cfg.test_count.value,
                                    cfg.args.region_end-cfg.args.region_start))
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
        if wrong / float(center_col.nsegments) < cfg.args.pileup_min_error:
            with cfg.test_count.get_lock():
                cfg.test_count.value += 1
                sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                        .format(cfg.cand_count.value, cfg.test_count.value, 
                            cfg.args.region_end-cfg.args.region_start))
            continue

        with cfg.cand_count.get_lock(): 
            cfg.cand_count.value += 1
        with cfg.test_count.get_lock(): 
            cfg.test_count.value += 1
            sys.stderr.write('{} candidates selected, {:010d} of {} tested\r' \
                    .format(cfg.cand_count.value, cfg.test_count.value,
                            cfg.args.region_end-cfg.args.region_start))
        positions.append(start_col)

    return positions



def get_candidate_positions():

    cand_positions = None
    if cfg.args.use_existing_candidates:

        print("> loading candidate positions")
        candidates_file = os.path.join(cfg.args.output_dir, 'candidates.npy')

        if os.path.isfile(candidates_file):
            cand_positions = np.load(candidates_file)
        else:
            print("ERROR: candidates file '{}' does not exist.".format(candidates_file))
            sys.exit(-1)

    else:
        print("> finding candidate positions")
        candidate_pool = mp.Pool()
        cand_positions = list(filter(None, candidate_pool.map(
            find_candidate_positions, 
            list(range(cfg.args.region_start, cfg.args.region_end, 
                cfg.args.region_batch_size)))))
        cand_positions = [item for sublist in cand_positions for item in sublist]

        print("\n> saving candidate positions")
        np.save(os.path.join(cfg.args.output_dir, 'candidates'), cand_positions)

    return cand_positions



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
            error_positions.append(pos+1-(cfg.args.base_window-1)//2)

    # return error positions within range
    results = list(filter(lambda x: x >= start and x < end, error_positions))
    print("{} errors found in range {}-{}".format(len(results),
        cfg.args.region_start, cfg.args.region_end))
    return results



def generate_block(ref_start):

    # initialize training data block and encoding scheme
    calls_to_draft = pysam.AlignmentFile(cfg.args.calls_to_draft, 'rb')
    block = np.zeros( (cfg.args.base_window, 8, 4) )
    ref_end = ref_start + cfg.args.base_window
    base_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    feat_idx = {'REF': 0, 'INS': 1, 'DEL': 2, 'SUB': 3}

    for col in calls_to_draft.pileup(cfg.args.region_contig, 
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
                        if pos_offset + ins_offset >= cfg.args.base_window:
                            break
                        block[pos_offset+ins_offset][base_idx[ins_base]+strand_offset][feat_idx['INS']] += 1

                # substitution or correct
                block[pos_offset][base_idx[ref_base]  + strand_offset][feat_idx['REF']] += 1
                block[pos_offset][base_idx[read_base] + strand_offset][feat_idx['INS']] += 1
                block[pos_offset][base_idx[ref_base]  + strand_offset][feat_idx['DEL']] += 1
                block[pos_offset][base_idx[read_base] + strand_offset][feat_idx['SUB']] += 1

    # features are relative to reference counts
    for pos in range(cfg.args.base_window):
        for base in range(len(base_idx)*2):
            block[pos][base][feat_idx['INS']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['DEL']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['SUB']] -= block[pos][base][feat_idx['REF']]

    with cfg.pos_count.get_lock(): 
        cfg.pos_count.value += 1
        sys.stderr.write('{} blocks generated\r'.format(cfg.pos_count.value))

    return block[np.newaxis,:]



def generate_training_data(cand_positions, error_positions):

    # filter any actual errors from the candidate positions
    correct_positions = list(set(cand_positions)-set(error_positions))

    # limit ratio of actual errors to candidate positions for training
    np.random.shuffle(correct_positions)
    correct_positions = correct_positions[:cfg.args.max_error_ratio*len(error_positions)]

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
    if len(targets) < cfg.args.max_num_blocks:
        print("WARNING: {} blocks requested, generating {}".\
                format(cfg.args.max_num_blocks, len(targets)))
    positions = positions[:cfg.args.max_num_blocks]
    targets = targets[:cfg.args.max_num_blocks]

    # generate the blocks for all positions
    block_pool = mp.Pool()
    blocks = list(block_pool.map(generate_block, positions))
    return np.vstack(blocks), targets



def call_error_probs(cand_positions, model, device):

    # initialize device and model
    model.eval()
    init(cfg.args.seed, device)
    device = torch.device(device)
    dtype = np.float16 if cfg.args.half else np.float32

    # generate all candidate blocks and merge
    print("> generating blocks")
    choose_pool = mp.Pool()
    cand_blocks = choose_pool.map(generate_block, cand_positions)
    cand_blocks = np.vstack([block.astype(dtype) for block in cand_blocks])

    # chunk all blocks in to batches
    data = ChunkBlock(cand_blocks)
    data_loader = torch.utils.data.DataLoader(
            data, batch_size = cfg.args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
    )

    # determine whether candidates are worth polishing
    print("\n> calling blocks")
    results = np.zeros(len(cand_positions))
    called = 0
    with torch.no_grad():
        for block in data_loader:
            block = block.to(device)
            result = model(block)
            results[called:called+len(block)] = \
                    torch.sigmoid(result).cpu().numpy().flatten()
            called += len(block)
            print("{} blocks called\r".format(called), end="")

    return results
