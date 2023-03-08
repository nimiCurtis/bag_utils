
import sys

# reading bag files dependencies
from bagpy import bagreader
import numpy as np
import pandas as pd
import json
from functools import reduce

# extracting images dependendcies
from typing import Union
import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
import open3d as o3d
import ctypes
import struct


# import from parallel modules
PATH = os.path.join(os.path.dirname(__file__),'../')
sys.path.insert(0, PATH)
from bag_parser.bag_parser import Parser
from utils.image_data_handler import DepthHandler, ImageHandler
dp = DepthHandler()
ih = ImageHandler()

class BagReader():
    """A class for exporting/reading data from bags
        """    
    
    def __init__(self):
        """ Constructor of BagReader object

            Args:
                bag_file (string): path string of .bag file
            """            

        # initialize data frames dictionary
        self._bag_file = None
        self._dfs = {}

        # initialize counter
        self._frame_count = 0

    @property
    def bag(self):
        """Get bag file path

        Returns:
            str: bag file path
        """     

        return self._bag_file

    @bag.setter
    def bag(self,bag_file:str):
        """Set bag to handle

        Args:
            bag_file (str): bag file path
        """
        
        self._bag_file = bag_file
        
        # Use exist or create metadata
        self.bag_read = bagreader(self._bag_file) # get bagreader object using bagpy library
        self.metadata_file = os.path.join(self.bag_read.datafolder, "metadata.json")    
        self.MetaData ={}
        if os.path.exists(self.metadata_file): # if already exist load the content of its file to self.Metadata dict
            with open(self.metadata_file,"r") as json_file:
                self.MetaData = json.load(json_file)
        else:
            self.MetaData["exported"] = False


    def get_dfs(self)->dict:
        """Get data frames exported from bag

        Returns:
            dict: dictionary of dataframes
        """
        
        if self.MetaData["exported"]:
            self._dfs = self.read()
        else:
            self._dfs = self.export()

        return self._dfs


    def read(self)->dict:
        """Read data of existing bag datafolder

        Returns:
            dict: dictionary of dataframes


        -TODO: 
        - take topics from the record config file of the specific bag and check the metadata
        """


        print("[INFO]  Bag already exported, Reading data ...")
        
        names = ["imu","rgb","depth","confidence","disparity","pcl","tf","synced"] # change based on configuration
        dfs = {}
        
        # iterating on topics names and read data
        for name in names:
            if name in self.MetaData:
                dfs[name] = pd.read_csv(self.MetaData[name])

        return dfs

    def export(self,with_imgs:bool)->dict:
        """This function export the data from the bag and modify the data frames accordingly

        Returns:
            dict: dictionary of dataframes
        """     
        print("[INFO]  Bag doesn't exported, Exporting data ...")

        # initialize with_imgs
        self._with_imgs = with_imgs

        # set topic df
        self.topic_df = self.bag_read.topic_table 
        
        self.MetaData["labeled"] = [] # can delete this
        dfs = {}

        # read and set imu_df
        for index, topic_row in self.topic_df.iterrows():
            if (topic_row['Types']=='sensor_msgs/Imu') and  topic_row['Message Count']!=0: # stop when topic is the imu topic and its not empty
                    imu_file = self.bag_read.message_by_topic(topic_row['Topics']) # create the csv file  
                    dfs["imu"] = pd.read_csv(imu_file) # set the df
                    dfs["imu"].drop_duplicates(subset=['header.stamp.secs','header.stamp.nsecs'],ignore_index=True,inplace=True)
                    dfs["imu"].to_csv(imu_file) # rewrite imu csv
                    self.MetaData["imu"] = imu_file # save the path to metadata

            # export pcl data
            if ((topic_row['Types']=='sensor_msgs/PointCloud2') and topic_row['Message Count']!=0):
                dfs["pcl"] = self.export_pcl(topic_row['Topics'])
            
            # export images data
            if ((topic_row['Types']=='sensor_msgs/Image')or(topic_row['Types']=='stereo_msgs/DisparityImage')) and  topic_row['Message Count']!=0: # stop when topic is Image/Stereo kind and its not empty
                dfs = self.init_image_df(dfs,topic_row['Topics'])
                

        # get df of topics synced with imu
        if "imu" in self.MetaData:
            
            # wanted topics
            wanted_topics = ['imu', 'rgb', 'depth'] # The keys you want
            synced_topics = dict((k, dfs[k]) for k in wanted_topics if k in dfs)
            dfs["synced"] = self.sync_with_imu(synced_topics)
            synced_file = os.path.join(self.bag_read.datafolder,"synced.csv")
            dfs["synced"].to_csv(synced_file)
            # save the path to metadata
            self.MetaData["synced"] = synced_file  

        
        # change exported status and dump MetaData to a json file
        self.MetaData["exported"] = True
        with open(self.metadata_file, "w") as json_file:
            json.dump(self.MetaData, json_file, indent=3)

        return dfs

    def export_pcl(self,topic:str)->pd.DataFrame:
        """Export pcl data as ply file and return the df with the pathes

        Args:
            topic (str): pcl topic

        Returns:
            pd.DataFrame: df with of pcl files paths
        """  

        df = pd.DataFrame()
        path_buffer = []
        self._frame_count = 0
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        dir = os.path.join(self.bag_read.datafolder,"pcl")
        csv_file = os.path.join(self.bag_read.datafolder,"zed_pcl.csv")
        if not os.path.exists(dir):
            os.mkdir(dir)


        #tmp_file = self.bag_read.message_by_topic(topic)
        bag = rosbag.Bag(self._bag_file, "r") # read the bag file using rosbag library
        for topic, msg, t in bag.read_messages(topics=topic):


            pc = pc2.read_points(msg, skip_nans=True)

            int_data = list(pc)
            len_data = len(int_data)
            xyz = np.empty((len_data, 3))
            rgb = np.empty((len_data, 3))
            for idx in range(len_data):
                x = int_data[idx]
                test = x[3] 
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f' ,test)
                i = struct.unpack('>l',s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000)>> 16
                g = (pack & 0x0000FF00)>> 8
                b = (pack & 0x000000FF)
                # prints r,g,b values in the 0-255 range
                            # x,y,z can be retrieved from the x[0],x[1],x[2]
                # xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
                # rgb = np.append(rgb,[[r,g,b]], axis = 0)

                xyz[idx] = x[:3]
                rgb[idx] = [r/255,g/255,b/255]
            #pc = pypcd.PointCloud.from_msg(msg)
            frame_path = os.path.join(dir, "frame%06i.ply" % self._frame_count)
            path_buffer.append(frame_path)
            
            # pc.save_pcd(frame_path, compression='binary')
            #pc.save(frame_path)

            out_pcd = o3d.geometry.PointCloud()    
            out_pcd.points = o3d.utility.Vector3dVector(xyz)
            out_pcd.colors = o3d.utility.Vector3dVector(rgb)
            o3d.io.write_point_cloud(frame_path,out_pcd)
            self._frame_count+=1
            
        bag.close
        
        df["pcl_path"] = path_buffer
        df.to_csv(csv_file)
        self.MetaData["pcl"] = csv_file
        print("[INFO]  pcl folder saved")
        return df

    def init_image_df(self,dfs:dict,topic:str)->dict:
        """This function initializing image data frames using set_image_df function per topic

        Args:
            dfs (dict): dictionary of dataframes
            topic (str): name of the topic

        Returns:
            dict: modified dictionary dataframes
        """     

        topic_split = topic.split('/')

        names = ["rgb","depth","confidence","disparity","pointclod"] # change based on configuration

        # iterating on topics names and set df for every image topic
        for name in names:
            if name in topic_split:
                dfs[name] = self.set_image_df(topic,name)
                break
        
        return dfs

    def set_image_df(self,topic:str,img_type:str)->pd.DataFrame:
        """This function creating the image type dataframes and csv files including the outputs from 
        extract_images function

        Args:
            topic (str): type of image rgb/depth/confidence/disparity 
            img_type (str): _description_

        Returns:
            pd.DataFrame: pandas data frame depends on the type of image
        """

        self._frame_count = 0 # initialize counter every time using this function

        # set the img data folder and numpy values folder
        dir = os.path.join(self.bag_read.datafolder,img_type)
        dir_vals = os.path.join(dir,'vals')

        # crating data folder for the relevant image type
        if not os.path.exists(dir):
            os.mkdir(dir)
            if not os.path.exists(dir_vals) : # create values directory
                os.mkdir(dir_vals) 

        # create csv temp file using the bagreader library - temporarily because it deosnt handle good with imgs 
        tmp_file = self.bag_read.message_by_topic(topic)
        df = pd.read_csv(tmp_file)
        if (img_type == 'disparity'):
            df.columns = df.columns.str.replace('image.data','data',regex=True) # rename disparity columns starting with 'image.'

        df.drop('data',inplace = True , axis =1) # drop the data column because its containing garbage data
        df['frame_path'], df['np_path'] = self.extract_images(topic, dir, img_type)

        df.to_csv(tmp_file) # create updated csv file
        self.MetaData[img_type] = tmp_file # save to metadata

        return df

    def extract_images(self, topic:str, dir:str, img_type:str)->list:
        """extract images data based on topic and list

        Args:
            topic (str):  name of topic
            dir (str): path to the directory
            img_type (str): type of image rgb/depth/confidence/disparity

        Returns:
            list: of numpy data and images frames
        """

        bag = rosbag.Bag(self._bag_file, "r") # read the bag file using rosbag library
        bridge = CvBridge() # create bridge object

        # initialize paths lists
        frame_path_list = [] 
        numpy_path_list = []

        # iterate the topic msgs
        for topic, msg, t in bag.read_messages(topics=topic):

            # convert image msgs to opencv format
            try:
                if img_type != 'disparity':
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # "passthrough" = keep the same encoding format of the image
                else:
                    cv_img = bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)

            frame_path = os.path.join(dir, "frame%06i.jpg" % self._frame_count) 

            if (img_type != "rgb"): # if not rgb type save the real values as .npy file, and update np list
                values_array = np.array(cv_img, dtype=np.float32) # convert type to d_type --> extract depth values

                if (img_type == "depth"):
                    cv_img = dp.get_depth_normalization(values_array) # get normalized image of depth values

                if (img_type == "disparity"):
                    max_disparity = msg.max_disparity 
                    min_disparity = msg.min_disparity
                    cv_img = dp.get_disparity_colormap(values_array,min_disparity,max_disparity) # get color map of dispatity values

            else:
                values_array = np.array(cv_img[:,:,:3], dtype=np.int32)

            numpy_path_list.append(self.save_np_data(values_array,dir)) # save values

            
            frame_path_list.append(frame_path) # update frame path list
            
            if self._with_imgs:
                cv2.imwrite(frame_path, cv_img)    # save img

            self._frame_count += 1

        bag.close()
        print(f"[INFO]  {img_type} folder saved") # convert it to log in the future

        return frame_path_list, numpy_path_list

    def save_np_data(self,values_array:list,dir:str)->str: ### make it more generic to any np file
        """This function save values of the image into .npy file 

        Args:
            values_array (list): np values
            dir (str): path to directory

        Returns:
            str: path to the saved file
        """

        numpy_path = os.path.join(dir, "vals/np_values%06i.npy" % self._frame_count)
        np.save(numpy_path, values_array)
        return numpy_path

    def sync_with_imu(self,dfs:dict)->pd.DataFrame:
        """Get sync dataframe with imu data

        Args:
            dfs (dict): dictionary of dataframes (including imu)

        Returns:
            pd.DataFrame: synced dataframe
        """
        df_sync = dfs["imu"]
        #dfs_to_sync = [v for k, v in dfs.items() ]
        cols = ['np_path', 'frame_path']
        keys = [k for k, v in dfs.items() if k!='imu']
        
        for key in keys:
            df = dfs[key]
            df = df.rename(columns={c: c+f'_{key}' for c in df.columns if c in cols})
            df_sync = pd.merge(df_sync,df[["header.stamp.secs","header.stamp.nsecs",f"np_path_{key}",f"frame_path_{key}"]],on=["header.stamp.secs","header.stamp.nsecs"],how="right")    
            
        return df_sync

    def update_metadata(self,key:str,val)->None:
        """Update existed metadata file

        Args:
            key (str): key of meta dict
            val (_type_): value of meta dict
        
        -TODO:
        -check if existed if not raise error

        """        
        with open(self.metadata_file, "r+") as json_file:
                self.MetaData = json.load(json_file)
                self.MetaData[key] = val
                json_file.seek(0)
                json.dump(self.MetaData, json_file, indent=3)

def export_bag(bag_obj:BagReader,with_imgs:bool)->None:
    """Exporting bag data

    Args:
        bag_obj (BagReader): _description_
    """

    if bag_obj.MetaData["exported"] == False:
            bag_obj.export(with_imgs=with_imgs)
    else:
            print(f"[INFO]  Bag {bag_obj.bag} already exported. Not Exporting.")

def main():
    os.chdir(os.path.dirname(__file__))

    bag_obj = BagReader()
    # get arguments
    parser = Parser.get_parser()
    Parser.add_bool_arg(parser,'with_imgs',default=False)
    args = Parser.get_args(parser)

    bag_file = PATH+'../bag/2023-03-07-19-16-37.bag' # default for example and debug
    
    if args.bag_batch_folder is not None:
        for filename in os.scandir(args.bag_batch_folder): 
            if filename.is_file() and filename.path.split('.')[-1]=='bag':
                bag_file = filename.path
                bag_obj.bag = bag_file
                export_bag(bag_obj,args.with_imgs)

    else:
        if args.single_bag is not None:
            bag_file = args.single_bag
        
        bag_obj.bag = bag_file
        export_bag(bag_obj,args.with_imgs)

if __name__ == '__main__':
    main()
