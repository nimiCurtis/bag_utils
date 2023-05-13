import os
import argparse


class Parser():


    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser(description ='Bag Iterator')

        return parser

    @staticmethod
    def get_args(parser):

        parser.add_argument('-b', dest="single_bag", type=is_bag_file,
                            help="Use single bag file only")
        
        parser.add_argument('-a', dest="bag_batch_folder", type=is_bag_dir,
                            help="Use all bag files in the 'bag_batch' dir")
        
        parser.add_argument('-l','--list', nargs='+',dest="bag_batch_list", help='list of bag_batch folders', required=True)

        parser.add_argument('--all', dest="folder_of_batches", type=is_bag_dir,
                            help="Use all batches folder in the 'bag' dir")
        
        return parser.parse_args()

    @staticmethod
    def add_bool_arg(parser, name, default=False):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name:default})

    @staticmethod
    def add_arg(parser,arg, name, help):
        parser.add_argument(arg, dest=name,help=help)



def is_bag_file(arg_bag_str: str) -> str:
    """"""
    # check file validation
    if os.path.isfile(arg_bag_str) and arg_bag_str.split('.')[-1]=='bag':
        return arg_bag_str
    else:
        msg = f"Given bag file {arg_bag_str} is not valid! "
        raise argparse.ArgumentTypeError(msg)


def is_bag_dir(arg_bag_str:str):
    # check dir validation
    if os.path.isdir(arg_bag_str):
        return arg_bag_str
    else:
        msg = f"Given bag directory {arg_bag_str} is not valid! "
        raise argparse.ArgumentTypeError(msg)



def main():
    args = Parser.get_args()
    print('Single bag: {}'.format(args.single_bag))
    print('Multiple bags folder: {}'.format(args.bag_batch_folder))


if __name__ == '__main__':
    main()