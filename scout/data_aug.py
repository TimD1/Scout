#!/usr/bin/env python3

"""
Merge training data from multiple datasets (adding augmentation later).

$ scout data_aug 
"""
import os, re, shutil, toml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np


def match(mylist):
    ''' Returns whether all values in a list are equal. '''
    return len(set(mylist)) == 1



def main(args):

    # print information on number of generated blocks
    args.routine = "data_aug"
    print('> selecting up to {} blocks (each) from {} directories'.format( \
            args.max_blocks_each, len(args.input_dirs.split(','))))

    # grab training data from all specified directories
    blocks, targets = [], []
    for input_dir in args.input_dirs.split(','):
        print('> processing {}'.format(input_dir))

        # check if valid directory
        if not os.path.isdir(input_dir):
            print('ERROR: {} is not a directory.'.format(input_dir))
            exit(1)

        # select some data from directory
        dir_blocks = np.load(os.path.join(input_dir, 'blocks.npy'))
        dir_targets= np.load(os.path.join(input_dir, 'targets.npy'))
        p = np.random.permutation(len(dir_blocks))
        dir_blocks, dir_targets = dir_blocks[p], dir_targets[p]
        blocks.append(dir_blocks[:args.max_blocks_each])
        targets.append(dir_targets[:args.max_blocks_each, np.newaxis])

    # check that all training data blocks are same dimensions
    block_depths = [block.shape[1] for block in blocks]
    if not match(block_depths):
        print('ERROR: All block depths must match')
        exit(1)
    block_widths = [block.shape[2] for block in blocks]
    if not match(block_widths):
        print('ERROR: All block widths must match')
        exit(1)

    # merge all data into single shuffled array
    print("> shuffling blocks")
    blocks = np.vstack(blocks)
    targets = np.vstack(targets)
    p = np.random.permutation(len(blocks))
    blocks, targets = blocks[p], targets[p]

    # prevent accidental overwriting
    if os.path.isdir(args.output_dir):
        if not args.force:
            print('WARNING: overwriting "{}".'.format(args.output_dir))
            if input("Continue? (y/n) ") != "y": exit(0)
        shutil.rmtree(args.output_dir)

    # save config log file of all parameters for data generation
    print("> saving config")
    os.makedirs(args.output_dir, exist_ok=True)
    argsdict = dict(data_aug = vars(args))
    blocks_config = {}
    for input_dir in args.input_dirs.split(','):
        block_config = {}
        block_config_file = os.path.join(input_dir, 'config.toml')
        if os.path.isfile(block_config_file):
            block_config = toml.load(block_config_file)
        blocks_config[input_dir] = block_config
    toml.dump({**argsdict, **blocks_config}, 
            open(os.path.join(args.output_dir, 'config.toml'), 'w'))

    # save data
    print("> saving data")
    np.save(os.path.join(args.output_dir, 'blocks'), blocks)
    np.save(os.path.join(args.output_dir, 'targets'), targets)



def argparser():

    parser = ArgumentParser(
            formatter_class = ArgumentDefaultsHelpFormatter,
            add_help = False
    )
    
    # set file paths
    parser.add_argument('input_dirs')
    parser.add_argument('output_dir')

    # set training data params
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--max_blocks_each', default=10000, type=int)

    return parser
