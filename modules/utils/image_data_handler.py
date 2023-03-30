"""
TODO:
- erase things related to the grid and feature line
- refactoring
"""


# Import libraries
import os
from typing import Union
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import normalize

class ImageHandler():
    """_summary_
        """    
    
    def __init__(self):
        """_summary_
            """        
        
        # shape and size of the image grid
        self.grid_shape = (3,4)           
        self.number_of_regions = self.grid_shape[0]*self.grid_shape[1]
        
        

    def split_to_regions(self, img:np.ndarray)-> Union[np.ndarray,np.ndarray,np.ndarray]:
        """This function split generic image matrice to regions

            Args:
                img (np.ndarray): generic image matrice

            Returns:
                Union[np.ndarray,np.ndarray,np.ndarray]: regions, h_grid, w_grid 
            """        

        # split height and width indexes by requierd grid shape
        h_grid = np.linspace(0,img.shape[0],self.grid_shape[0]+1,dtype=np.uint16) 
        w_grid = np.linspace(0,img.shape[1],self.grid_shape[1]+1,dtype=np.uint16)

        # create array of the img regions
        regions_arr = [img[x:x+int(h_grid[1]),y:y+int(w_grid[1])] for x in range(0,img.shape[0],int(h_grid[1])) for y in range(0,img.shape[1],int(w_grid[1]))]
        regions = np.array(regions_arr)

        return regions, h_grid, w_grid

    def get_regions_mean(self,img:np.ndarray)->np.ndarray:
        """This function calculate image regions means

            Args:
                img (np.ndarray): splited image matrice

            Returns:
                np.ndarray: regions means of the splited image 
            """               

        mean_arr = np.array([np.nanmean(img[x]) for x in range(self.number_of_regions)]).reshape(self.grid_shape)

        return mean_arr
    
    def get_regions_std(self,img: np.ndarray)-> np.ndarray:
        """This function calculate image regions std
            Args:
                img (np.ndarray): splited image matrice

            Returns:
                np.ndarray: standard deviation of image splited regions
            """           

        std_arr = np.array([np.nanstd(img[x]) for x in range(self.number_of_regions)]).reshape(self.grid_shape)

        return std_arr

    def extract_features_singleImg(self,img: np.ndarray)->Union[np.ndarray,np.ndarray]:
        """This function extract relevant features from image matrice
        
            Args:
                img (np.ndarray): generic image matrice

            Returns:
                Union[np.ndarray,np.ndarray]: means array, std array
            """  

        # first split the image
        img_splited = self.split_to_regions(img) 

        # extract featurs
        mean = self.get_regions_mean(img_splited)
        std = self.get_regions_std(img_splited)

        return mean, std


    def GaussianBlur(self, img: np.ndarray, config: dict)->np.ndarray:
        """Implement gaussian blur using opencv builtin function

        Args:
            img (np.ndarray): image mat
            config (dict): configuration

        Returns:
            np.ndarray: blured image
        """

        gs_ksize = (config.ksize,config.ksize)
        sigmaX = config.sigmaX
        sigmaY = config.sigmaY
        gauss_blured = cv2.GaussianBlur(img,gs_ksize,sigmaX,sigmaY)

        return gauss_blured

    def GaborFilter(self, img: np.ndarray, config: dict)->np.ndarray:
        """Implement Gabor filter using opencv builtin functions

        Args:
            img (np.ndarray): image mat
            config (dict): configuration 

        Returns:
            np.ndarray: filtered image
        """

        gb_ksize = (config.ksize,config.ksize)
        sigma = config.sigma
        theta = np.deg2rad(config.theta)
        lambd = config.lambd  
        gamma = config.gamma
        psi = config.psi
        gabor_kernel = cv2.getGaborKernel(gb_ksize,sigma,theta,lambd,gamma,psi)
        gabor_filtered = cv2.filter2D(img,-1,gabor_kernel)

        return gabor_filtered

    def BilateralFilter(self,img: np.ndarray, config: dict)->np.ndarray: # docstring
        """Implement bilateral filter using opencv builtin functions

        Args:
            img (np.ndarray): image mat
            config (dict): configuration 

        Returns:
            np.ndarray: filtered image
        """

        d = config.d
        sigmaColor = config.sigmaColor 
        sigmaSpace = config.sigmaSpace
        bilateral_filterd = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

        return bilateral_filterd

    def Sobel(self,img: np.ndarray, config: dict)->np.ndarray:
        """Implement sobel filter using opencv builtin functions

        Args:
            img (np.ndarray): image mat
            config (dict): configuration 

        Returns:
            np.ndarray: filtered image
        """

        sobel_thresh = config.thresh
        s_ksize = config.ksize
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1,s_ksize) # get sobely img
        # normalized sobel matrice
        # take the abs because sobel consist negative values.
        #sobely_abs = np.abs(sobely)
        # shift the matrice by the min val
        # sobely_shifted = sobely - np.min(sobely)
        # scaled the matrice to fall between 0-255
        #sobely_scaled = (sobely_abs/np.max(sobely_abs))*255
        # convert matrice to uint8
        #sobely_u8 = sobely_scaled.astype("uint8")
        sobely_u8 = cv2.convertScaleAbs(sobely)
        sobely_u8[sobely_u8<sobel_thresh] = 0

        return sobely_u8

    def Canny(self,img: np.ndarray, config: dict)->np.ndarray: 
        """Implement canny edge detection using opencv builtin functions

        Args:
            img (np.ndarray): image mat
            config (dict): configuration 

        Returns:
            np.ndarray: edges image
        """

        canny_thresh1 = config.thresh1
        canny_thresh2 = config.thresh2
        aperture = config.aperture
        edges = cv2.Canny(img, canny_thresh1, canny_thresh2, aperture)

        return edges

    def Hough(self,img: np.ndarray, config: dict)->np.ndarray:
        """Implement Hough line detection using opencv builtin functions

        Args:
            img (np.ndarray): image mat
            config (dict): configuration 

        Returns:
            np.ndarray: lines
        """

        minLineLength = config.minLineLength
        maxLineGap = config.maxLineGap
        rho = config.rho
        theta = np.pi/config.theta 
        hough_thresh = config.thresh
        lines = cv2.HoughLinesP(img, rho, theta, hough_thresh, minLineLength, maxLineGap)

        return lines


    
    def imshow_with_grid(self,img_path:str, color:tuple=(0, 255, 0), thickness:int=1):
        """This function show image with the grid

        Args:
            img_path (str): path of image
            color (tuple, optional): RGB values of the grid lines. Defaults to (0, 255, 0)
            thickness (int, optional): grid lines thickness value. Defaults to 1.
        """

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        rows, cols = self.grid_shape
        dy, dx = h / rows, w / cols

        # draw vertical lines
        for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
            y = int(round(y))
            cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

        plt.imshow(img)
        plt.show()

    def write_video(self, folder_path:str, file_name:str, frames:list, fps:float):
        """Video wirter by opencv VideoWriter function

        Args:
            folder_path (str): folder path to save
            file_name (str): file name 
            frames (list): list of frame to convert to video
            fps (float): fps param
        """        
        
        if not os.path.exists(folder_path+"/video"):
            os.mkdir(folder_path+"/video")
        
        # file path formating
        now = datetime.now().strftime("%H-%M-%S")
        file_path = folder_path+f"/video/{now}_"+file_name+".mp4"
        
        # use opencv videowriter
        shape =  frames[0].shape
        if len(shape) == 3:
            h, w, _ = shape # unpack h,w dimensions from the first frame assuming frames dimensions are equall
        elif len(shape)==2:
            h, w = shape
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w,h))
        cv2.VideoWriter()
        for frame in frames:
            writer.write(frame)

        writer.release() 




class DepthHandler(ImageHandler):
    """_summary_

        Args:
            ImageHandler (object): parent class
        """        

    def __init__(self):
        """_summary_
            """        

        super().__init__()

    def mm2meter(self, depth:np.ndarray)->np.ndarray:
        """This function convert depth matrice to meters units

        Args:
            depth (np.ndarray): depth values in milimeters

        Returns:
            np.ndarray: depth values in meters
        """        

        return depth/1000

    def meter2mm(self, depth:np.ndarray)->np.ndarray:
        """This function convert depth matrice to mm units

        Args:
            depth (numpy array): depth values in meters

        Returns:
            depth (numpy array): depth values in milimeters
        """        

        return depth*1000

    def zero_to_nan(self, depth:np.ndarray)->np.ndarray:
        """This function convert zero values to nan values in given depth matrice

        Args:
            depth (np.ndarray): depth vals matrice

        Returns:
            np.ndarray: depth matrice with zero values converted to nan
        """         

        depth[depth==0] = np.nan
        return depth

    def extract_depth_features_single(self, depth:np.ndarray)->np.ndarray:
        """This function extract features from a depth matrice
        using the extract_features_singleImg

        Args:
            depth (np.ndarray): depth vals matrice

        Returns:
            np.ndarray: means and std array from 'get_regions_std()'
        """               

        depth0 = self.zero_to_nan(depth) # NOTE: different bitween formats of image.. need to check
        mean , std = self.extract_features_singleImg(depth0)
        return mean, std

    def extract_depth_features_batch(self, df:pd.DataFrame)->list:
        """This function extract features from given data frame of depth matrices
        Args:
            df (pd.DataFrame): data frame contains depth matrices

        Returns:
            list: means and std array off depth regions means
        """

        mean_arr = []
        std_arr = []

        for index in range(df.shape[0]):
            depth = np.load(df.np_path[index])
            depth = self.mm2meter(depth)                             # converted until config ready
            mean, std = self.extract_depth_features_single(depth)
            
            mean_arr.append(mean)
            std_arr.append(std)
        
        return mean_arr, std_arr


    def get_depth_features_df(self, df:pd.DataFrame)->pd.DataFrame:
        """This function add extracted regions features columns to a given depth images dataframe

        Args:
            df (pd.DataFrame): data frame contains depth matrices

        Returns:
            pd.DataFrame: modified dataframe to include features columns
        """       

        feat_df = df 
        feat_df["mean"], feat_df["std"] = self.extract_depth_features_batch(df)

        return feat_df

    def get_max_line(self,lines:np.ndarray)->np.ndarray:
        """Get the maximum line in the lines array

        Args:
            lines (np.ndarray): lines array

        Returns:
            np.ndarray: max line array
        """        
        # take the points of the lines
        p1 = lines[:,:2] 
        p2 = lines[:,2:]
        # calc dist = calc line length
        dist = np.linalg.norm(p1-p2,axis=1)
        # find max
        max_length = np.max(dist)
        max_line = lines[dist==max_length]

        return max_line 
    
    def get_mid(self,line:np.ndarray)->int:
        """Get the indexes of the middle of given line

        Args:
            line (np.ndarray): line 

        Returns:
            int: x, y integers of the middle
        """        
        x1,y1,x2,y2 = line
        return int((x2+x1)/2),int((y2+y1)/2)

    def get_feature_line(self,depth:np.ndarray,lines:np.ndarray=None)->Union[np.ndarray,np.ndarray,tuple]:
        """Get the feature line of given frame with lines detected

        Args:
            lines (np.ndarray): lines array
            depth (np.ndarray): depth values matrice

        Returns:
            Union[np.ndarray,np.ndarray,tuple]: [feature line depth vals, feature line indexes, middle indexes]
        """
        if lines is not None:
            max_line = self.get_max_line(lines)
            xmid,ymid = self.get_mid(max_line[0])
            feature_vals = depth[:,xmid]
            feature_index = np.ones((feature_vals.shape[0],2),dtype=np.int16)*xmid
            feature_index[:,1] = np.arange(0,feature_vals.shape[0])
        
        else:
            xmid,ymid = int(depth.shape[1]/2),(depth.shape[0]/2)
            feature_vals = depth[:,xmid]
            feature_index = np.ones((feature_vals.shape[0],2),dtype=np.int16)*xmid
            feature_index[:,1] = np.arange(0,feature_vals.shape[0])

        return [feature_vals,feature_index, (xmid,ymid)]

    def get_feature_region(self, lines:np.ndarray, depth:np.ndarray)->Union[np.ndarray,np.ndarray,tuple]:
        """Get the feature line from a region suround the featrue line of given frame with lines detected

        Args:
            lines (np.ndarray): lines array
            depth (np.ndarray): depth values matrice

        Returns:
            Union[np.ndarray,np.ndarray,tuple]: [feature line depth vals, feature line indexes, middle indexes]
        """

        max_line = self.get_max_line(lines)
        xmid,ymid = self.get_mid(max_line[0])

        # choose the width of the region 
        feature_vals = depth[:,xmid-5:xmid+5]
        feature_vals = feature_vals.mean(axis=1)
        #feature_vals = feature_vals[feature_vals!=0]
        feature_index = np.ones((feature_vals.shape[0],2),dtype=np.int16)*xmid
        feature_index[:,1] = np.arange(0,feature_vals.shape[0])
        
        return [feature_vals,feature_index, (xmid,ymid)]

    def feature_line_filter(self,depth_line:np.ndarray)->np.ndarray:
        fdepth_line = depth_line
        j_flag=False
        if fdepth_line[0]==0:fdepth_line[0]= fdepth_line[fdepth_line!=0][0]
        if fdepth_line[-1]==0:fdepth_line[-1]= fdepth_line[fdepth_line!=0][-1]

        for i in range(len(fdepth_line-1)):
            if fdepth_line[i]==0:
                if not j_flag:
                    for j in range(i+1,len(fdepth_line)):
                        if fdepth_line[j]!=0:
                            j_flag=True
                            fdepth_line[i]=np.mean([fdepth_line[i-1],fdepth_line[j]])
                            break
                else:
                    fdepth_line[i]=np.mean([fdepth_line[i-1],fdepth_line[j]])
            else:
                if j_flag:
                    j_flag=False
        
        return fdepth_line
    
    def gauss_filter1d(self, depth_line, k_size=3):

        return gaussian_filter1d(depth_line,k_size)
    
    def filter_depth_line(self,depth_line):

        d_line = self.feature_line_filter(depth_line)
        d_line = self.gauss_filter1d(d_line)

        return d_line
    
    def normalize(self,arr, norm = 'max'):
        
        return normalize(arr, norm=norm)

        


        

    def get_depth_normalization(self, img:np.ndarray)->np.ndarray:
        """Normalize the depth image to fall between 0 (black) and 1 (white)

        Args:
            img (np.ndarray): image matrice to be normalized

        Returns:
            np.ndarray: normalized image
        """        

        img_clone = img.copy() # make a copy of the values because cv2 only make a view and changing source

        # cv2.normalize(img_clone, img_clone, 0, 1, cv2.NORM_MINMAX)
        # img_clone = img_clone*255
        
        depth_img = np.array(256*img_clone/0x0fff,dtype=np.uint8)
        
        return depth_img
    
    def get_disparity_colormap(self, img:np.ndarray, min_disparity:int, max_disparity:int)->np.ndarray:
        """Get colormap image from diaprity values

        Args:
            img (np.ndarray): image disparity values
            min_disparity (int): min disparity
            max_disparity (int): max disparity

        Returns:
            np.ndarray: _description_
        """        
        
        img_clone = img.copy() # make a copy of the values because cv2 only make a view and changing source
        normal_dist = max_disparity - min_disparity
        shifted_disparity = (img_clone - min_disparity)                    # shift values to get rid from negetive vals | current_min = 0
        scaled_disparity = (shifted_disparity*255)/normal_dist            # normalize to fall between (0,255) 
        scaled_disparity = np.clip(scaled_disparity,0,255)                        # clip , not sure if totaly necssary
        scaled_disparity = scaled_disparity.astype(np.uint8)                      # change format 
        colormap_disparity= cv2.applyColorMap(scaled_disparity,cv2.COLORMAP_JET)  # apply colormap

        return cv2.cvtColor(colormap_disparity, cv2.COLOR_BGR2RGB)                # convert to rgb --> red = close dist, blue = far dist


def main(): 
    pass

if __name__ == '__main__':
    main()
