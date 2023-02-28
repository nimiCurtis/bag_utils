import os
import argparse


class Parser():
    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description ='Bag Iterator')

        parser.add_argument('-b', dest="single_bag", type=is_bag_file,
                            help="Use single bag file only")
        
        parser.add_argument('-a', dest="multiple_bags_folder", type=is_bag_dir,
                            help="Use all bag files in the 'bag' dir")
        

        return parser.parse_args()


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
    print('Multiple bags folder: {}'.format(args.multiple_bags_folder))


if __name__ == '__main__':
    main()