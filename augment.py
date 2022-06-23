#!/usr/bin/env python3

import os
import time
import shutil
import argparse
from tkinter.ttk import Progressbar

import scipy
import scipy.ndimage
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm

is_overwrite = True # If true, any data previously processed will be lost
is_resize_outputs = True
labels_available  = True
progressbar_enabled = False

output_imgsize = (144,144,144)

#region custom errors
class FilenameError(Exception):
        pass
#endregion custom errors

#region parse arguments
def get_args():
        parser = argparse.ArgumentParser(description="sMRI-augmentation", add_help=True,
                                 formatter_class=argparse.RawTextHelpFormatter)
        
        parser.add_argument("-input_dir", default = os.path.abspath(os.path.join(os.path.dirname(__file__),"data/")), 
                            help="Path to input data directory with brains and if available, labels.\n"
                                + "filename format: brains- subjectid_t1.nii | labels- subjectid_labels.nii\n"
                                + "directory structure:\ndata\n |- brains\n |- target_labels")
        
        parser.add_argument("-output_dir", default = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/data_aug")), 
                            help="Path to output directory to save augmented brains and if available, labels.")
        
        parser.add_argument("-aug_types", default = ['r','n','s','d','g'], nargs='+',
                            help= "Methods to augment input data: first letter of method(s) separated with space.\n"
                                + "Eg: to perform all augmentation methods: augment.py -aug_types r n s d g\n"
                                + " r - rotation\t n - noise\n s - spiking\t d - deformation \n g - ghosting")
        
        return parser.parse_args()
#endregion parse arguments

#region utilities
def resample_img(output_imgsize, img_file, msk_file):
        
        ref_file = sitk.GetImageFromArray(np.zeros((output_imgsize)))
        ref_file.SetOrigin(img_file.GetOrigin())
        ref_file.SetDirection(img_file.GetDirection())
        
        ref_physicalsize = np.zeros(img_file.GetDimension())
        ref_physicalsize[:] = [(sz-1)*spc if sz*spc > mx  else mx for sz, spc, mx in zip(img_file.GetSize(), img_file.GetSpacing(), ref_physicalsize)]
        
        ref_spacing   = [ phys_sz/(sz-1) for sz, phys_sz in zip(output_imgsize, ref_physicalsize) ]
        ref_file.SetSpacing(ref_spacing)
        
        img_resampled = sitk.Resample(img_file, ref_file)
        if not (msk_file == None):
                msk_resampled = sitk.Resample(msk_file, img_resampled)
                return img_resampled, msk_resampled
        else:
                return img_resampled, None
#endregion utilities

#region augmentation methods
def rotate(img_fname, msk_fname, angle, axes):
        
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)        
        img_arr  = sitk.GetArrayFromImage(img_file)
        
        dir = 'C' if angle >= 0 else 'A'
        rot_str = "r"+dir+str(f'{np.abs(angle):02}')+str(axes[0])+str(axes[1])
        
        if not (angle == 0):
                imgaug_arr = scipy.ndimage.rotate(img_arr, angle, axes=axes, reshape=True)   
        else:
                 imgaug_arr =  img_arr

        imgaug_arr = imgaug_arr[:,::-1,::-1]
        
        imgaug_file = sitk.GetImageFromArray(imgaug_arr)
        imgaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1))
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('r'):fname_arr_last.index('r')+6], rot_str)
        
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if not (angle == 0):
                img_arr = img_arr[:,::-1,::-1]
                img_file_in = sitk.GetImageFromArray(img_arr)
                img_file_in.SetDirection((-1,0,0,0,-1,0,0,0,-1))
                
                sitk.WriteImage(img_file_in, imgaug_fname.replace(rot_str,"r"+dir+str(f'{np.abs(angle):02}')+"00"))
                
        if labels_available and not (msk_fname==None):
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)
                msk_arr = sitk.GetArrayFromImage(msk_file)
                
                if not (angle == 0):
                        mskaug_arr = scipy.ndimage.rotate(msk_arr, angle, axes=axes, reshape=True)
                else:
                        mskaug_arr = msk_arr
                        
                mskaug_arr[mskaug_arr >0.3]= 1 ; mskaug_arr[mskaug_arr<=0.3]= 0
                mskaug_arr = scipy.ndimage.binary_closing(mskaug_arr).astype(int)
                mskaug_arr = mskaug_arr[:,::-1,::-1]
                
                mskaug_file = sitk.GetImageFromArray(mskaug_arr)
                mskaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1)) 
                
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                sitk.WriteImage(mskaug_file, mskaug_fname)
                
                if not (angle == 0):
                        msk_arr = scipy.ndimage.binary_closing(msk_arr).astype(int)
                        msk_arr = msk_arr[:,::-1,::-1]
                        
                        msk_file_in = sitk.GetImageFromArray(msk_arr)
                        msk_file_in.SetDirection((-1,0,0,0,-1,0,0,0,-1))
                
                        sitk.WriteImage(msk_file_in,mskaug_fname.replace(rot_str,"r"+dir+str(f'{np.abs(angle):02}')+"00"))
        
def noisify(img_fname, msk_fname, noise_perc):
        
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)                
        img_arr  = sitk.GetArrayFromImage(img_file)
        
        standard_dev = np.max(img_arr)*noise_perc
        
        gaussian_re = np.random.normal(loc=0, scale=standard_dev, size=img_file.GetSize())
        gaussian_im = np.random.normal(loc=0, scale=standard_dev, size=img_file.GetSize())
        
        imgaug_arr = np.sqrt(np.square(img_arr + gaussian_re) + np.square(gaussian_im))
        
        imgaug_file = sitk.GetImageFromArray(imgaug_arr).CopyInformation(img_file)
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('r'):fname_arr_last.index('r')+6], str(f'{noise_perc:02}'))
        
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        sitk.WriteImage(imgaug_file, imgaug_fname)

        if labels_available:
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                shutil.copy(msk_fname, mskaug_fname)
        
#endregion augmentation methods      

#region augmentation processes
def rotate_images(imgaug_list, mskaug_list):
        """
        Set min and max rotation angles and list axis of rotation to be used. 
        Rotate each image along each axis and in each of these cases, rotate 
        each image from min to max angle in increments of rot_inc.
                
        Inputs: List of files in output directories (resampled and renamed)
                data/data_aug
                |-brains
                |-target_labels
        """
        angle_min = -20 ; angle_max = 20
        rot_inc  = 4
        all_axes = [(1, 0), (1, 2)] #[(1, 0), (1, 2), (0, 2)]
        no_rot_files = int((len(imgaug_list)*((angle_max-angle_min)/rot_inc)*len(all_axes)))
        
        if progressbar_enabled: pbar = tqdm(total=no_rot_files,desc='Generating rotated images:')
        starttime_rot = time.time()
        for img_fname in imgaug_list:
                if labels_available and not (mskaug_list == None):
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                           
                for rot_angle in range(angle_min,angle_max+1,rot_inc):
                                for axes in all_axes:
                                        rotate(img_fname, msk_fname, rot_angle, axes)
                                        if progressbar_enabled: pbar.update(1)
        endtime_rot = time.time()
        rot_elapsedtime = round(endtime_rot-starttime_rot,3)
        
        if progressbar_enabled: pbar.close()  
        else: print("Generating rotated images complete.", no_rot_files,"files generated in",rot_elapsedtime,"seconds.")  
        
        return rot_elapsedtime

def noisify_images(imgaug_list, mskaug_list):
        
        noise_perc_min = 20
        noise_perc_max = 20
        noise_perc_inc =  5
        no_noised_files = int((len(imgaug_list)*((noise_perc_max-noise_perc_min)/noise_perc_inc)))
        
        if progressbar_enabled: pbar = tqdm(total=no_noised_files,desc='Adding noise to images:')
        starttime_noise = time.time()
        for img_fname in imgaug_list:
                if labels_available and not (mskaug_list == None):
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                
                for noise_perc in range(noise_perc_min, noise_perc_max+1, noise_perc_inc):
                        
                        noisify(img_fname, msk_fname, noise_perc)
                        if progressbar_enabled: pbar.update(1)
        
        endtime_noise = time.time()
        noise_elapsedtime = round(endtime_noise-starttime_noise,3)
        
        if progressbar_enabled: pbar.close()  
        else: print("Adding noise to images complete.", no_noised_files,"files generated in",noise_elapsedtime,"seconds.")  			
		
#endregion augmentation processes

def main():
        """
        # Input arguments: data directory, output directory, file type (nii or mnc)
        # If overwrite is true, delete any previously created output directores or files
        # Create an empty bool matrix to assign augmentation methods to perform in the
        # order: rotation (r), noise (n), spiking (s), deformation (d) and ghosting (g)
        # Rotate all images. Then start remaining augmentation processes in parallel.
        """
        
        global data_dir, output_dir, imgio_type
        
        data_dir    = get_args().input_dir
        augtypes_in = get_args().aug_types
        output_dir  = get_args().output_dir
        
        #region initialize directories
        if is_overwrite and (os.path.exists(output_dir)): 
                if os.path.exists(output_dir): shutil.rmtree(output_dir) 
                elif os.path.exists(os.path.join(data_dir,"data_aug")): shutil.rmtree(os.path.join(data_dir,"data_aug")) 
        
        if not os.path.exists(output_dir): 
                os.mkdir(output_dir)
                os.mkdir(os.path.join(output_dir,"brains"))
                
                if os.path.exists(os.path.join(data_dir,"target_labels")):
                        labels_available = True
                        os.mkdir(os.path.join(output_dir,"target_labels"))
                else: labels_available = False
        #endregion initialize directories
           
        files_list = os.listdir(os.path.join(data_dir,"brains"))
        no_files   = len(files_list)
        
        filename_suffix = "rC0000-n00-d0-sp0000-gh0"	
        
        augtypes = np.zeros(5,dtype=bool)
        for type in augtypes_in: augtypes[['r','n','s','d','g'].index(type)]=True
  
        #region augmentation - individual
        for fname in files_list:
                #region check file extension  
                try:
                        if not fname.endswith((".nii",".mnc")): 
                                raise FilenameError(fname.split(".")[1])
                        else: 
                                imgio_type = "NiftiImageIO" if fname.split(".")[1] == "nii" else "MINCImageIO"
                except FilenameError as e:  
                        print("Invalid extension. File must be .mnc or .nii. {} not processed".format(fname))
                #endregion check file extension 
                
                #region get filenames, load images
                img_fname   = os.path.join(data_dir,"brains",fname)
                imgaug_fname= os.path.join(output_dir,"brains",fname).replace("_t1.nii","_t1_"+filename_suffix+".nii")
                
                img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)
                
                if labels_available:
                        msk_fname   = os.path.join(data_dir,"target_labels",fname.replace("_t1.nii","_labels.nii"))
                        mskaug_fname= os.path.join(output_dir,"target_labels",fname.replace("_t1.nii","_labels.nii")).replace("_labels.nii","_labels_"+filename_suffix+".nii")
                        
                        msk_file    = sitk.ReadImage(os.path.join(data_dir,"target_labels",msk_fname), imageIO=imgio_type)
                else: msk_file = None
                #endregion get filenames, load images
                
                #region resize and save images in output directory
                if is_resize_outputs: 
                        imgaug_file, mskaug_file = resample_img(output_imgsize, img_file, msk_file) 
                        
                        mskaug_arr = sitk.GetArrayFromImage(mskaug_file)
                        
                        mskaug_arr[mskaug_arr >0.3]= 1 ; mskaug_arr[mskaug_arr<=0.3]= 0    
                        mskaugt_file = sitk.GetImageFromArray(mskaug_arr)
                        mskaugt_file.CopyInformation(mskaug_file)      
                        
                        sitk.WriteImage(imgaug_file , imgaug_fname)
                        sitk.WriteImage(mskaugt_file, mskaug_fname)  
                else:
                        msk_arr = sitk.GetArrayFromImage(msk_file)
                        msk_arr[msk_arr >0.3]= 1 ; msk_arr[msk_arr<=0.3]= 0    
                        mskt_file = sitk.GetImageFromArray(msk_arr) 
                        mskt_file.CopyInformation(msk_file)
                        
                        sitk.WriteImage(img_file, imgaug_fname)
                        sitk.WriteImage(mskt_file, mskaug_fname)       
                #endregion resize and save images in output directory
                                      
        #region get list of renamed and resized images
        imgaug_list = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(output_dir,"brains")), x),os.listdir(os.path.join(output_dir,"brains")))) 
        if labels_available:
                mskaug_list = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(output_dir,"target_labels")), x),os.listdir(os.path.join(output_dir,"target_labels"))))
        else:  mskaug_list = None
        #endregion get list of renamed and resized images
        
        """
        # If an augmentation method is selected, start respective process
        # Images are first rotated. Other augmenation methods are applied 
        # afterwards to the new dataset (including originals and rotated)
        """
        if augtypes[0]: 
                rot_elapsedtime = rotate_images(imgaug_list, mskaug_list)
        
        if augtypes[1]: 
                noise_elapsedtime = rotate_images(imgaug_list, mskaug_list) 
                
         
        #endregion augmentation - individual                
if __name__ == "__main__":
    main()