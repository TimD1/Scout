import multiprocessing as mp
import numpy as np
import re, os, sys, toml, pysam, torch, h5py

try: import grouper.config as cfg
except: import scout.config as cfg
from scout.util import *


def get_pileup_positions(center_col_start):
    '''
    Returns list of all pileup positions exceeding minimum error rate or 
    homopolymer length.
    '''

    # create pysam alignment file
    calls_to_draft = pysam.AlignmentFile(cfg.args.calls_to_draft, "rb")

    # calculate region end
    center_col_end = min(center_col_start + cfg.args.region_batch_size, cfg.args.region_end)

    # select positions with sufficient error rate and depth
    positions = []
    for center_col in calls_to_draft.pileup(cfg.args.region_contig, \
            center_col_start, center_col_end, min_base_quality=1):

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
                    print('{} candidates selected, {:010d} of {} tested\r' \
                            .format(cfg.cand_count.value, cfg.test_count.value,
                                    cfg.args.region_end-cfg.args.region_start), end="", flush=True)
                positions.append(center_col.pos)
                continue

        # check percentage of reads which err at this position
        hap_diffs = [0, 0]
        hap_totals = [0, 0]
        for read in center_col.pileups:
            if read.alignment.has_tag('HP'):
                ref_base = read.alignment.get_reference_sequence().upper() \
                        [center_col.pos-read.alignment.reference_start]
                read_base = 'N' if read.query_position is None else \
                        read.alignment.query_sequence[read.query_position]
                if read_base != ref_base or read.indel > 0: 
                    hap_diffs[read.alignment.get_tag('HP')-1] += 1
                hap_totals[read.alignment.get_tag('HP')-1] += 1

        # if low error rate, continue
        if calculate_error_rate(hap_diffs, hap_totals) < cfg.args.pileup_min_error:
            with cfg.test_count.get_lock():
                cfg.test_count.value += 1
                print('{} candidates selected, {:010d} of {} tested\r' \
                        .format(cfg.cand_count.value, cfg.test_count.value, 
                            cfg.args.region_end-cfg.args.region_start), end="", flush=True)
            continue

        # passed min error threshold, update number of candidates selected
        with cfg.cand_count.get_lock(): 
            cfg.cand_count.value += 1
        with cfg.test_count.get_lock(): 
            cfg.test_count.value += 1
            print('{} candidates selected, {:010d} of {} tested\r' \
                    .format(cfg.cand_count.value, cfg.test_count.value,
                            cfg.args.region_end-cfg.args.region_start), end="", flush=True)
        positions.append(center_col.pos)

    return positions



def calculate_error_rate(hap_diff_counts, hap_total_reads):

    # choose haploid with greatest error rate
    rate = 0
    for diff, total in zip(hap_diff_counts, hap_total_reads):
        # ignore this haploid if zero coverage
        if total:
            rate = max(rate, diff/float(total))
    return rate




def medaka_stitch(hdf5_filename):
    '''
    Written as test to ensure Medaka's consensus_probs.hdf5 is parsed correctly.
    '''

    # sort regions in HDF5 file by chromosomal position  
    code = ['', 'A', 'C', 'G', 'T']
    hdf5_file = h5py.File(hdf5_filename, 'r')
    regions = [ r for r in hdf5_file['samples']['data'] ]
    starts = []  
    for r in regions:
        pos = re.search('chromosome:(.*)-(.*)', r).group(1)  
        starts.append(float(pos))
    regions = [r for p,r in sorted(zip(starts,regions))] 

    # stitch together consensus
    consensus = ""
    for idx, region in enumerate(regions):   
        probs = hdf5_file['samples']['data'][region]['label_probs'][:]   
        first_pos = hdf5_file['samples']['data'][region]['positions'][0]
        last_pos = hdf5_file['samples']['data'][region]['positions'][-1]
        start = None
        end = None

        # get start index within this chunk
        if idx == 0:
            start = 0
        else:
            prev_positions = hdf5_file['samples']['data'][regions[idx-1]]['positions'][:]
            start = (args.chunk_size - np.where(prev_positions == first_pos)[0]).squeeze() // 2

        # get end index within this chunk
        if idx == len(regions)-1:
            end = args.chunk_size
        else:
            next_positions = hdf5_file['samples']['data'][regions[idx+1]]['positions'][:]
            end = args.chunk_size-1 - np.where(next_positions == last_pos)[0].squeeze() // 2
        probs = probs[start:end, :]  
        for chunk_idx in range(probs.shape[0]):
            consensus += code[np.argmax(probs[chunk_idx,:])]

    return consensus



def get_medaka_positions(hdf5_filename):

    # sort regions in HDF5 file by chromosomal position  
    hdf5_file = h5py.File(hdf5_filename, 'r')
    regions = [ r for r in hdf5_file['samples']['data'] ]
    starts = []  
    for r in regions:
        pos = re.search('chromosome:(.*)-(.*)', r).group(1)  
        starts.append(float(pos))
    regions = [r for p,r in sorted(zip(starts,regions))] 
    
    # report difficult positions, keeping running total of consensus length  
    position = 0 
    difficult_positions = [] 
    for idx, region in enumerate(regions):   
        
        # get confidence scores  
        probs = hdf5_file['samples']['data'][region]['label_probs'][:]   
        error_probs = np.clip(1-np.max(probs, axis=1), 0.00001, 1)   
        qscores = -10*np.log10(error_probs)  
        first_pos = hdf5_file['samples']['data'][region]['positions'][0]
        last_pos = hdf5_file['samples']['data'][region]['positions'][-1]
        medaka_chunk_size = 10000
        start, end = None, None

        # get start index within this chunk
        if idx == 0:
            start = 0
        else:
            prev_positions = hdf5_file['samples']['data'][regions[idx-1]]['positions'][:]
            start = (medaka_chunk_size - np.where(prev_positions == first_pos)[0]).squeeze() // 2

        # get end index within this chunk
        if idx == len(regions)-1:
            end = medaka_chunk_size
        else:
            next_positions = hdf5_file['samples']['data'][regions[idx+1]]['positions'][:]
            end = medaka_chunk_size-1 - np.where(next_positions == last_pos)[0].squeeze() // 2
        
        # trim
        probs = probs[start:end, :]  
        qscores = qscores[start:end] 
        
        # report difficult positions 
        for chunk_idx in range(len(qscores)):
            if qscores[chunk_idx] <= cfg.args.max_qscore_medaka: 
                difficult_positions.append(position) 
            if np.argmax(probs[chunk_idx,:]):
                position += 1

    results = list(filter(lambda x: x >= cfg.args.region_start and 
                x < cfg.args.region_end, difficult_positions))
    return results



def get_candidate_positions():

    cand_positions = None
    cfg.cand_count.value = 0
    cfg.test_count.value = 0
    if cfg.args.load_candidates:

        print("> loading candidate pileup positions")
        candidates_file = os.path.join(cfg.args.output_dir, 'candidates.npy')

        if os.path.isfile(candidates_file):
            cand_positions = np.load(candidates_file)
            print("loaded {} positions".format(len(cand_positions)))
        else:
            print("ERROR: candidates file '{}' does not exist.".format(candidates_file))
            sys.exit(-1)

    else:
        print("> finding candidate pileup positions")
        with mp.Pool() as pool:
            cand_positions = list(filter(None, pool.map(get_pileup_positions, 
                list(range(cfg.args.region_start, cfg.args.region_end, 
                    cfg.args.region_batch_size)))))
        cand_positions = [item for sublist in cand_positions for item in sublist]

        print("\n> saving candidate positions")
        np.save(os.path.join(cfg.args.output_dir, 'candidates'), cand_positions)

    return cand_positions



def get_variant_positions(vcf_filename, min_qual=0): 
    '''
    Parse VCF file and extract relevant positions. 
    - VCF filename must be zipped (supply <filename>.vcf.gz)
    - VCF index must be present (<filename>.vcf.gz.tbi) 
    '''
    draft_vcf = pysam.VariantFile(vcf_filename, "r") 
    return [record.start for record in 
        draft_vcf.fetch(cfg.args.region_contig, \
        cfg.args.region_start, cfg.args.region_end) if record.qual >= min_qual ] 



def get_error_positions(error_catalogue):

    # parse the error catalogue, creating a list of all SNP positions
    print("> loading known error positions")
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
            if errs > cfg.args.max_error_size: continue

            # append the start of this region to list of error positions
            error_positions.append(pos)

    # return error positions within range
    results = list(filter(lambda x: x >= cfg.args.region_start and 
                x < cfg.args.region_end, error_positions))
    print("{} errors found in range {}-{}".format(len(results),
        cfg.args.region_start, cfg.args.region_end))
    return results



def generate_block(block_center):

    # initialize training data block and encoding scheme
    calls_to_draft = pysam.AlignmentFile(cfg.args.calls_to_draft, 'rb')
    block = np.zeros( (cfg.args.base_window, 16, 4) )
    ref_start = block_center - cfg.args.base_radius
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
            haploid_offset = 8 if read.alignment.has_tag('HP') and read.alignment.get_tag('HP') == 1 else 0
            origin_offset = strand_offset + haploid_offset
            ref_base = read.alignment.get_reference_sequence().upper() \
                    [col.pos-read.alignment.reference_start]
            read_base = 'N' if read.query_position is None else \
                    read.alignment.query_sequence[read.query_position]

            # deletion
            if read_base == 'N':
                block[pos_offset][base_idx[ref_base] + origin_offset][feat_idx['DEL']] += 1
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
                        block[pos_offset+ins_offset][base_idx[ins_base]+origin_offset][feat_idx['INS']] += 1

                # substitution or correct
                block[pos_offset][base_idx[ref_base]  + origin_offset][feat_idx['REF']] += 1
                block[pos_offset][base_idx[read_base] + origin_offset][feat_idx['INS']] += 1
                block[pos_offset][base_idx[ref_base]  + origin_offset][feat_idx['DEL']] += 1
                block[pos_offset][base_idx[read_base] + origin_offset][feat_idx['SUB']] += 1

    # features are relative to reference counts
    for pos in range(cfg.args.base_window):
        for base in range(len(base_idx)*2):
            block[pos][base][feat_idx['INS']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['DEL']] -= block[pos][base][feat_idx['REF']]
            block[pos][base][feat_idx['SUB']] -= block[pos][base][feat_idx['REF']]

    with cfg.pos_count.get_lock(): 
        cfg.pos_count.value += 1
        print('{} blocks generated\r'.format(cfg.pos_count.value), end="", flush=True)

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
    with mp.Pool() as pool:
        blocks = list(pool.map(generate_block, positions))
    return np.vstack(blocks), targets



def get_scout_scores(blocks):

    # chunk all blocks in to batches
    data = ChunkBlock(blocks)
    data_loader = torch.utils.data.DataLoader(
            data, batch_size = cfg.args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
    )

    # determine whether candidates are worth polishing
    scores = np.zeros(len(blocks))
    called = 0
    with torch.no_grad():
        for block in data_loader:
            block = block.half().to(cfg.args.device)
            result = cfg.model(block)
            scores[called:called+len(block)] = \
                    torch.sigmoid(result).cpu().numpy().flatten()
            called += len(block)
            print("{} blocks called\r".format(called), end="", flush=True)

    return scores
