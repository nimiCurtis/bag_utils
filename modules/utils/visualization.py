import os
import sys

import numpy as np
import cv2
import pandas as pd

PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from bag_reader.bag_reader import BagReader
from bag_parser.bag_parser import Parser

class Visualizer():

    def __init__(self) -> None:
        pass

    def vis_images(self,folder_path):
        # Get list of all image files in folder
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.npy')]

        # Sort files alphabetically
        image_files.sort()

        # Initialize index to 0
        index = 0

        # Display first image
        file_ext = os.path.splitext(image_files[index])[1]
        if file_ext == '.jpg':
            img = cv2.imread(image_files[index])
        elif file_ext == '.npy':
            img = np.load(image_files[index])
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if img.ndim==2:
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        cv2.imshow('Image Viewer', img)

        # Loop until 'q' key is pressed
        while True:
            # Add frame number text
            cv2.putText(img, f'Frame {index+1}/{len(image_files)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display image
            cv2.imshow('Image Viewer', img)

            key = cv2.waitKey(0)

            if key == ord('q'):  # Quit
                break
            elif key == ord('a') and index > 0:  # Previous image
                index -= 1
                file_ext = os.path.splitext(image_files[index])[1]
                if file_ext == '.jpg':
                    img = cv2.imread(image_files[index])
                elif file_ext == '.npy':
                    img = np.load(image_files[index])
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    if img.ndim==2:
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

            elif key == ord('d') and index < len(image_files) - 1:  # Next image
                index += 1
                file_ext = os.path.splitext(image_files[index])[1]
                if file_ext == '.jpg':
                    img = cv2.imread(image_files[index])
                elif file_ext == '.npy':
                    img = np.load(image_files[index])
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    if img.ndim==2:
                        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        cv2.destroyAllWindows()

    
    def vis_bag(self,bag_file:str,depth=False, rgb=False):
        bag_obj = BagReader()
        bag_obj.bag = bag_file
        
        df_rgb = pd.read_csv(bag_obj.MetaData["rgb"],index_col=0)
        df_depth = pd.read_csv(bag_obj.MetaData["depth"],index_col=0)

        display_both = False
        
        # Initialize index to 0
        index = 0
        idx_limit = 0
        
        # Create a list of dataframes to iterate through
        dfs = []
        if rgb:
            dfs.append(df_rgb)
            idx_limit = len(df_rgb)
        elif depth:
            dfs.append(df_depth)
            idx_limit = len(df_depth)
        else:
            display_both = True
            dfs.append(df_rgb)
            dfs.append(df_depth)
            idx_limit = min(len(df_depth),len(df_rgb))

        # Loop until 'q' key is pressed
        while index<idx_limit:
            # Display images from each dataframe
            for i, df in enumerate(dfs):
                # Get image path
                img_path = df.iloc[index]['np_path']

                # Load image
                img = np.load(img_path)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                if img.ndim==2:
                    img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

                # Add file number text
                cv2.putText(img, f'{df.iloc[index].name+1}/{len(df)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Display image
                if display_both and i == 0:
                    both_img = img.copy()
                elif display_both and i == 1:
                    both_img = np.concatenate((both_img, img), axis=1)
                    cv2.imshow('File Viewer', both_img)
                else:
                    cv2.imshow('File Viewer', img)

            key = cv2.waitKey(0)

            if key == ord('q'):  # Quit
                break
            elif key == ord('a') and index > 0:  # Previous file
                index -= 1
            elif key == ord('d') and index < idx_limit - 1:  # Next file
                index += 1

        cv2.destroyAllWindows()

def main():
    visualizer = Visualizer()

    parser = Parser.get_parser()
    Parser.add_bool_arg(parser, 'rgb', default=False)
    Parser.add_bool_arg(parser, 'depth', default=False)
    Parser.add_arg(parser,arg='-f',name='images_folder', help='Enter folder name containing images of .jpg or .npy formats')
    args = Parser.get_args(parser)

    if args.single_bag is not None:
        visualizer.vis_bag(bag_file=args.single_bag,depth=args.depth, rgb=args.rgb)
    
    elif args.images_folder is not None:
        visualizer.vis_images(args.images_folder)

if __name__=='__main__':
    main()