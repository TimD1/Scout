#!/usr/bin/env python3

"""
Plots training data for easier visualization.

$ scout plot_data train_data_folder/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import os, sys, random



def validate(args):

    if not os.path.exists(args.train_data_folder):
        print("ERROR: train_data_folder '{}' does not exist.".format(args.train_data_folder))
        sys.exit(-1)

    os.makedirs(args.plot_folder, exist_ok=True)



def plot(blocks, truth, plot_folder, max_plots):

    # randomly select data to plot
    max_plots = min(max_plots, truth.shape[0])
    indices = random.sample(range(truth.shape[0]), max_plots)
    titles = ['REF', 'INS', 'DEL', 'SUB']
    maxval = np.max(blocks)
    minval = np.min(blocks)

    # plot training data matrices
    fig, axs = plt.subplots(2,2, figsize=(20,8))
    plt.tight_layout()
    first = True
    for idx in indices:
        for plot in range(4):
            x = plot // 2
            y = plot % 2
            im = axs[x,y].imshow(np.transpose(blocks[idx,:,:,plot]), vmin=minval, \
                    vmax=maxval, cmap='jet', interpolation='nearest')
            axs[x,y].set_xticks([(blocks.shape[1]-1)//2])
            axs[x,y].set_xticklabels(['*'])
            axs[x,y].set_yticks(range(8))
            axs[x,y].set_yticklabels(['A','C','G','T',"A'","C'","G'","T'"])
            axs[x,y].set_title(titles[plot])
        plt.suptitle('Block {} was {}'.format(idx, 'error' if truth[idx] else 'correct'))
        if first:
            first = False
            plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.5)
        plt.savefig(os.path.join(plot_folder, '{}_{}.png'.format(idx, int(truth[idx]))))



def main(args):

    print("> validating arguments")
    validate(args)

    print("> loading data")
    blocks = np.load(os.path.join(args.train_data_folder, 'blocks.npy'))
    truth = np.load(os.path.join(args.train_data_folder, 'truth.npy'))

    print("> plotting data")
    plot(blocks, truth, args.plot_folder, args.max_plots)



def argparser():
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    parser.add_argument("train_data_folder", default="")
    parser.add_argument("plot_folder", default="")
    parser.add_argument("--max_plots", default=5, type=int)

    return parser
