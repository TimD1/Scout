#!/usr/bin/env python3

"""
Scout CNN model training.

$ scout train model data_dir/
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import toml, torch, csv, os, time
from datetime import datetime
from tqdm import tqdm

try: from apex import amp
except ImportError: pass

from scout.util import init, ChunkData, default_config
from scout.model import Model, FocalLoss

criterion = FocalLoss(apply_nonlin = torch.nn.Sigmoid(), alpha=[1,50], smooth=0.001)



def train(model, device, train_loader, optimizer, use_amp=False):

    # init block count, model, time, progress bar
    blocks = 0
    model.train()
    t0 = time.perf_counter()
    progress_bar = tqdm(
        total=len(train_loader), desc='[0/{}]'.format(len(train_loader.dataset)),
        ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
    )

    # iterate over training data in chunks of "batch size"
    with progress_bar:
        for data, targets in train_loader:

            # calculate forward step
            optimizer.zero_grad()
            blocks += data.shape[0]
            data = data.to(device)
            targets = targets.to(device)
            results = model(data)
            loss = criterion(results, targets)

            # calculate backward step
            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # update progress bar
            progress_bar.set_postfix(loss='%.4f' % loss.item())
            progress_bar.set_description("[{}/{}]".format(blocks, len(train_loader.dataset)))
            progress_bar.update()

    # return loss and time for this epoch
    return loss.item(), time.perf_counter() - t0



def test(model, device, test_loader):

    # initialize model for testing
    model.eval()
    test_loss = 0
    predictions = []
    all_targets = []

    # create list of all predictions and targets
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader, start=1):
            data, targets = data.to(device), targets.to(device)
            results = model(data)
            test_loss += criterion(results, targets)
            predictions.append(torch.sigmoid(results).round().cpu())
            all_targets.append(targets.cpu())

    # calculate accuracy metrics
    predictions = np.concatenate(predictions)
    all_targets = np.concatenate(all_targets)
    accuracy = 100*(1 - np.abs(predictions-all_targets).sum()/all_targets.shape[0])
    precision = 100 * all_targets[predictions == 1].sum() / predictions.sum()
    recall = 100 * predictions[all_targets == 1].sum() / all_targets.sum()

    # return loss and accuracy
    return test_loss.item()/batch_idx, accuracy, precision, recall



def main(args):

    # check that training directory doesn't exist already
    workdir = os.path.expanduser(args.training_dir)
    model_name = os.path.basename(os.path.normpath(args.training_dir))

    if os.path.exists(workdir) and not args.force:
        print("ERROR: training_dir '{}' exists, use -f to force training.".format(workdir))
        exit(1)

    if args.val_data_dir and args.train_val_split:
        print("ERROR: must choose separate validation dir OR train-val split")
        exit(1)
    args.routine = "train"

    # set device and initialize
    init(args.seed, args.device)
    device = torch.device(args.device)

    # load training data into np arrays
    print("> loading data")
    if args.val_data_dir:
        train_blocks = np.load(os.path.join(args.train_data_dir, "blocks.npy"))
        train_targets = np.load(os.path.join(args.train_data_dir, "targets.npy"))
        train_blocks = train_blocks[:args.max_train_blocks]
        train_targets = train_targets[:args.max_train_blocks].flatten()
        
        val_blocks = np.load(os.path.join(args.val_data_dir, "blocks.npy"))
        val_targets = np.load(os.path.join(args.val_data_dir, "targets.npy"))
        val_blocks = val_blocks[:args.max_val_blocks]
        val_targets = val_targets[:args.max_val_blocks].flatten()

    else:
        all_blocks = np.load(os.path.join(args.train_data_dir, "blocks.npy"))
        all_targets = np.load(os.path.join(args.train_data_dir, "targets.npy"))
        nblocks = min(len(all_targets), args.max_train_blocks)
        split = np.floor(nblocks * args.train_val_split).astype(np.int32)

        train_blocks = all_blocks[:split]
        train_targets = all_targets[:split]
        val_blocks = all_blocks[split:]
        val_targets = all_targets[split:]

    # split into torch-compatible train/test sets
    train_dataset = ChunkData(train_blocks, train_targets)
    test_dataset = ChunkData(val_blocks, val_targets)
    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
            batch_size=args.batch, num_workers=4, pin_memory=True)

    # load model config (architecture)
    print("> storing model & block params")
    config = toml.load(args.config)
    argsdict = dict(train = vars(args))

    # load block config (data_gen parameters)
    train_config = {}
    train_config_file = os.path.join(args.train_data_dir, 'config.toml')
    if os.path.isfile(train_config_file):
        train_config = toml.load(train_config_file)
    val_config = {}
    if args.val_data_dir:
        val_config_file = os.path.join(args.val_data_dir, 'config.toml')
        if os.path.isfile(val_config_file):
            val_config = toml.load(val_config_file)

    # merge config data and save in training directory
    os.makedirs(workdir, exist_ok=True)
    toml.dump({**config, **argsdict, **train_config, **val_config}, 
            open(os.path.join(workdir, 'config.toml'), 'w'))

    # generate model from config file
    print("> loading model")
    model = Model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=False, lr=args.lr)
    # TODO: add LR scheduler & resume training

    # allow multi-gpu training
    if args.multi_gpu:
        from torch.nn import DataParallel
        model = DataParallel(model)

    # use AMP to initialize model and optimizer if requested
    if args.amp:
        try:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        except NameError:
            print("ERROR: install AMP from https://github.com/NVIDIA/apex")
            exit(1)

    # train model for desired number of epochs
    for epoch in range(1, args.epochs+1):

        # train next iteration of model, allowing keyboard interrupt
        try:
            train_loss, duration = train(model, device, train_loader, optimizer, 
                    use_amp=args.amp)
            val_loss, val_acc, val_prec, val_rec = test(model, device, test_loader)
        except KeyboardInterrupt:
            break

        # print and save intermediate results
        print("[epoch {}] model={} loss={:.4f} val_acc={:.3f}% val_prec={:.3f}% val_rec={:.3f}%".format(
            epoch, model_name, val_loss, val_acc, val_prec, val_rec))
        model_state = model.state_dict() if not args.multi_gpu else model.module.state_dict()
        torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
        torch.save(optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))

        # record accuracy summary in training log file
        with open(os.path.join(workdir, 'training.csv'), 'a', newline='') as csvfile:
            csvw = csv.writer(csvfile, delimiter=',')
            if epoch == 1:
                csvw.writerow([
                    'time', 'duration', 'epoch', 'train_loss', 'val_loss', 
                    'val_acc', 'val_precision', 'val_recall'
                ])
            csvw.writerow([
                datetime.today(), int(duration), epoch, train_loss, val_loss, val_acc, val_prec, val_rec
            ])



def argparser():
    parser = ArgumentParser(
        formatter_class = ArgumentDefaultsHelpFormatter,
        add_help = False
    )

    # select directories
    parser.add_argument("training_dir")
    parser.add_argument("train_data_dir")
    parser.add_argument("--val_data_dir")
    parser.add_argument("--train_val_split", type=float, default=0.97)
    parser.add_argument("-f", "--force", action="store_true", default=False)

    # select data limits
    parser.add_argument("--max_train_blocks", default=10000000, type=int)
    parser.add_argument("--max_val_blocks", default=1000000, type=int)

    # model params
    parser.add_argument("--batch", default=512, type=int)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--epochs", default=400, type=int)

    # hardware params
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--multi-gpu", action="store_true", default=False)

    return parser
