#!/usr/bin/env python3

"""
Finds select positions using trained model.

$ scout find reads_to_draft.bam model_dir
"""

import sys, time, torch
from datetime import timedelta 

import numpy as np 
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter



def main(args):

    sys.stderr.write("> loading model\n")

    # model = load_model(
    #     args.model_directory, args.device, weights=int(args.weights),
    #     half=args.half, chunksize=args.chunksize, use_rt=args.cudart,
    # )

    # samples = 0
    # num_reads = 0
    # max_read_size = 4e6
    # dtype = np.float16 if args.half else np.float32

    # t0 = time.perf_counter() 
    # sys.stderr.write("> calling\n")

    # with torch.no_grad():

    #     while True:

    #         read = reader.queue.get()
    #         if read is None: 
    #             break

    #         if len(read.signal) > max_read_size: 
    #             sys.stderr.write("> skipping long read %s (%s samples)\n" % (read.read_id, len(read.signal)))
    #             continue 

    #         num_reads += 1 
    #         samples += len(read.signal)

    #         raw_data = torch.tensor(read.signal.astype(dtype)) 
    #         chunks = chunk(raw_data, args.chunksize, args.overlap) 

    #         posteriors = model(chunks.to(args.device)).cpu().numpy() 
    #         posteriors = stitch(posteriors, args.overlap // model.stride // 2) 

    #         writer.queue.put((read, posteriors[:raw_data.shape[0]])) 

    # duration = time.perf_counter() - t0

    # sys.stderr.write("> completed reads: %s\n" % num_reads)
    # sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration))) 
    # sys.stderr.write("> samples per second %.1E\n" % (samples / duration)) 
    # sys.stderr.write("> done\n") 



def argparser():

    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    parser.add_argument("pileup_bam", default="")
    parser.add_argument("model_directory", default="")
    parser.add_argument("locations_out_dir", default="locations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--half", action="store_true", default=False)

    return parser
