import os
import subprocess
import sys

# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from bag_parser.bag_parser import Parser

class BagCompress:

    def __init__(self) -> None:
        pass
    
    @staticmethod
    def compress_batch(bag_folder_path):
        print(f"[INFO] compressing batch - {bag_folder_path}")
        # Loop through each file in the folder
        for filename in os.listdir(bag_folder_path):
            if filename.endswith(".bag"):
                # Compress the file using rosbag compress command
                subprocess.call(["rosbag", "compress", os.path.join(bag_folder_path, filename)])
                original_file_name = filename.rsplit('.')[0]+".orig.bag"
                # Delete the original file
                os.remove(os.path.join(bag_folder_path, original_file_name))
    
    @staticmethod
    def compress_folder(folder_path):
        print(f"[INFO] compressing folder - {folder_path}")
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.startswith("bag_batch"):
                batch_path = os.path.join(folder_path,filename)
                BagCompress.compress_batch(batch_path)



def main():

    # get arguments
    parser = Parser.get_parser()
    args = Parser.get_args(parser)

    bag_compressor = BagCompress()

    if args.bag_batch_folder is not None:
        bag_compressor.compress_batch(args.bag_batch_folder)

    elif args.folder_of_batches is not None:
        bag_compressor.compress_folder(args.folder_of_batches)



if __name__ == '__main__':
    main()