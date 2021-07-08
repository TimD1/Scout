from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from scout import data_gen, plot, data_aug, train, find, evaluate
modules = ['data_gen', 'plot', 'data_aug', 'train', 'find', 'evaluate']


__version__ = '0.1.0'



def main():

    # create overall Scout parser
    parser = ArgumentParser(
        'scout', 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v', '--version', action='version',
        version='%(prog)s {}'.format(__version__)
    )

    # allow user to list subcommands
    subparsers = parser.add_subparsers(
        title='subcommands', description='valid commands',
        help='additional help', dest='command'
    )
    subparsers.required = True

    # add subparsers for each module
    for module in modules:
        mod = globals()[module]
        p = subparsers.add_parser(module, parents=[mod.argparser()])
        p.set_defaults(func=mod.main)

    args = parser.parse_args()
    args.func(args)
