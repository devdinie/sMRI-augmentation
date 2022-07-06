#!/usr/bin/env python3
from ast import AugAssign
import os
import time
import scipy
import shutil
import random
import torchio
import argparse
import numpy as np
import scipy.ndimage
import multiprocessing
import SimpleITK as sitk

from tqdm import tqdm
from functools import partial

is_overwrite = True # If true, any data previously processed will be lost
is_resize_outputs = True
labels_available  = True
progressbar_enabled = True
mix_augmethods = False
indivaug_done  = False
output_imgsize = (144,144,144)

#region custom errors
class FilenameError(Exception):
        pass
class InputError(Exception):
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
        
        parser.add_argument("-aug_types", default = ['r','n','m','s','b'], nargs='+',
                            help= "Methods to augment input data: first letter of method(s) separated with space.\n"
                                + "Eg: to perform all augmentation methods: augment.py -aug_types r n m s b \n"
                                + " r - rotation\t n - noise injection\n m - adding motion\t s - spiking\n b - blurring") 
        return parser.parse_args()
#endregion parse arguments

#region utilities
def resample_img(output_imgsize, img_file, msk_file):

        
        msk_resampledtoimg = sitk.Resample(msk_file, img_file.GetSize(), sitk.Transform(),sitk.sitkNearestNeighbor,
                                           img_file.GetOrigin(), img_file.GetSpacing(), img_file.GetDirection(), 0,
                                           msk_file.GetPixelID())
        
        #region reference image properties
        ref_physicalsize = np.zeros(img_file.GetDimension())
        ref_physicalsize[:] = [(sz-1)*spc if sz*spc > mx  else mx for sz, spc, mx in zip(img_file.GetSize(), img_file.GetSpacing(), ref_physicalsize)]
        
        reference_spacing   = [ phys_sz/(sz-1) for sz, phys_sz in zip(output_imgsize, ref_physicalsize) ]
        
        reference_img = sitk.GetImageFromArray(np.zeros(output_imgsize))
        reference_img.SetOrigin(msk_resampledtoimg.GetOrigin())
        reference_img.SetDirection(msk_resampledtoimg.GetDirection())
        reference_img.SetSpacing(reference_spacing)
        
        img_resampled = sitk.Resample(img_file,reference_img)
        img_resampled.SetOrigin(reference_img.GetOrigin())
        img_resampled.SetSpacing(reference_img.GetSpacing())
        img_resampled.SetDirection(reference_img.GetDirection())
        
        msk_resampled = sitk.Resample(msk_file,img_resampled)
        msk_resampled.SetOrigin(img_resampled.GetOrigin())
        msk_resampled.SetSpacing(img_resampled.GetSpacing())
        msk_resampled.SetDirection(img_resampled.GetDirection())
        #endregion define reference image properties
        
        return img_resampled, msk_resampled

#endregion utilities

#region augmentation methods
def rotate(img_fname, msk_fname, angle, axes):
        
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)        
        img_arr  = sitk.GetArrayFromImage(img_file)
        
        dir = 'C' if angle >= 0 else 'A'
        rot_str = "r"+dir+str(f'{np.abs(angle):02}')+str(axes[0])+str(axes[1])
        
        imgaug_arr = scipy.ndimage.rotate(img_arr, angle, axes=axes, reshape=False)   
        imgaug_arr = imgaug_arr[:,::-1,::-1]
        imgaug_file = sitk.GetImageFromArray(imgaug_arr)
        imgaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1))
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('r'):fname_arr_last.index('r')+6], rot_str)
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if labels_available and not (msk_fname==None):
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)
                
                msk_arr = sitk.GetArrayFromImage(msk_file)
                msk_arr[msk_arr >0.3]= 1 ; msk_arr[msk_arr<=0.3]= 0
                
                #TODO: add binary fill holes (temporarily removed: output visible with nib but can't Display with minctoolkit)
                
                mskaug_arr = scipy.ndimage.rotate(msk_arr, angle, axes=axes, reshape=False)
                mskaug_arr = mskaug_arr[:,::-1,::-1]
                mskaug_file = sitk.GetImageFromArray(mskaug_arr)
                mskaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1)) 
                            
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
        
                sitk.WriteImage(mskaug_file, mskaug_fname)
        """       
        if not (angle == 0):
                sitk.WriteImage(img_file, imgaug_fname.replace(rot_str,"r"+dir+str(f'{np.abs(angle):02}')+"00"))
                
                if labels_available and not (msk_fname==None):
                        sitk.WriteImage(msk_file,mskaug_fname.replace(rot_str,"r"+dir+str(f'{np.abs(angle):02}')+"00"))
        """
        
def noisify(img_fname, msk_fname, noise_perc):
        
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)                
        img_arr  = sitk.GetArrayFromImage(img_file)
        
        standard_dev = np.max(img_arr)*(noise_perc/100)
        
        gaussian_re = np.random.normal(loc=0, scale=standard_dev, size=tuple(img_arr.shape))
        gaussian_im = np.random.normal(loc=0, scale=standard_dev, size=tuple(img_arr.shape))
        
        imgaug_arr = np.sqrt(np.square(img_arr + gaussian_re) + np.square(gaussian_im))
        
        imgaug_arr = imgaug_arr[:,::-1,::-1]
        imgaug_file = sitk.GetImageFromArray(imgaug_arr)
        imgaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1))
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('n'):fname_arr_last.index('n')+3], "n"+str(f'{noise_perc:02}'))
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if labels_available:
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)                
                msk_arr  = sitk.GetArrayFromImage(msk_file)
                
                mskaug_arr = msk_arr[:,::-1,::-1]
                mskaug_file = sitk.GetImageFromArray(mskaug_arr)
                mskaug_file.SetDirection((-1,0,0,0,-1,0,0,0,-1))
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                
                sitk.WriteImage(mskaug_file, mskaug_fname)

def random_motion(img_fname, msk_fname, motion):
     
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)
        randmotion_tfm = torchio.transforms.RandomMotion(degrees=5, translation=5, num_transforms=2, image_interpolation= 'linear')
        imgaug_file = randmotion_tfm(img_file)
      
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('m'):fname_arr_last.index('m')+3], "m"+str(f'{motion:02}'))
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if labels_available:
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)  
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                
                sitk.WriteImage(msk_file, mskaug_fname)
        
def random_spikes(img_fname, msk_fname, spikes):
        
        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)
        randspike_tfm = torchio.transforms.RandomSpike(num_spikes= 1, intensity=(0, 1))
        imgaug_file = randspike_tfm(img_file)
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('s'):fname_arr_last.index('s')+3], "s"+str(f'{spikes:02}'))
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if labels_available:
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)  
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                
                sitk.WriteImage(msk_file, mskaug_fname)        

def blur(img_fname, msk_fname, blur_in):

        img_file = sitk.ReadImage(img_fname, imageIO=imgio_type)
        blur_tfm = torchio.transforms.RandomBlur(std = (0, 1))
        imgaug_file = blur_tfm(img_file)
        
        img_basename  = os.path.basename(img_fname)
        fname_arr_last= img_basename.split('_')[len(img_basename.split('_'))-1]
        imgaug_basename = img_basename.replace(fname_arr_last[fname_arr_last.index('b'):fname_arr_last.index('b')+3], "b"+str(f'{blur_in:02}'))
        imgaug_fname = os.path.join(os.path.dirname(img_fname), imgaug_basename)
        
        sitk.WriteImage(imgaug_file, imgaug_fname)
        
        if labels_available:
                msk_file = sitk.ReadImage(msk_fname, imageIO=imgio_type)  
                mskaug_fname = os.path.join(os.path.dirname(msk_fname), imgaug_basename.replace("_t1_","_labels_"))
                
                sitk.WriteImage(msk_file, mskaug_fname)     
        
#endregion augmentation methods      

#region augment dataset
def rotate_images(imgaug_list):
        """
        Set min and max rotation angles and list axis of rotation to be used. 
        Rotate each image along each axis and in each of these cases, rotate 
        each image from min to max angle in increments of rot_inc.
                
        Inputs: List of files in output directories (resampled and renamed)
                data/data_aug
                |-brains
                |-target_labels
        """
        
        angle_min = -10 
        angle_max =  10
        rot_inc  = 5
        all_axes = [(1, 0), (1, 2)] #[(1, 0), (1, 2), (0, 2)]
        
        if mix_augmethods and indivaug_done: no_rot_files = int(len(imgaug_list))
        else: no_rot_files = int((len(imgaug_list)*((angle_max-angle_min)/rot_inc)*len(all_axes)))
 
        if progressbar_enabled: pbar = tqdm(total=no_rot_files,desc='Generating rotated images')
        starttime_rot = time.time()
        
        for img_fname in imgaug_list:
        
                if labels_available:
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                 
                if mix_augmethods and indivaug_done:
                        rotate(img_fname, msk_fname, random.choice(range(angle_min, angle_max,rot_inc)), random.choice(all_axes))
                        if progressbar_enabled: pbar.update(1)    
                else:
                        for rot_angle in range(angle_min,angle_max+1,rot_inc):
                                for axes in all_axes:
                                        rotate(img_fname, msk_fname, rot_angle, axes)
                                if progressbar_enabled: pbar.update(2)
        endtime_rot = time.time()
        rot_elapsedtime = round(endtime_rot-starttime_rot,3)
        
        if progressbar_enabled: pbar.close()  
        print("Generating rotated images complete.", no_rot_files,"files generated in",rot_elapsedtime,"seconds.")  
        
        return rot_elapsedtime

def noisify_images(imgaug_list):
        
        noise_perc_min = 2
        noise_perc_max = 4
        noise_perc_inc = 1
        
        if mix_augmethods and indivaug_done: no_noised_files = int(len(imgaug_list))
        else: no_noised_files = int((len(imgaug_list)*(((noise_perc_max-noise_perc_min)/noise_perc_inc)+1)))
        
        if progressbar_enabled: pbar = tqdm(total=no_noised_files,desc='Adding noise to images')
        starttime_noise = time.time()
        
        for img_fname in imgaug_list:
                
                if labels_available:
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                
                if mix_augmethods and indivaug_done:
                        noisify(img_fname, msk_fname, random.choice(range(noise_perc_min, noise_perc_max, noise_perc_inc)))
                        if progressbar_enabled: pbar.update(1)
                else:
                        for noise_perc in range(noise_perc_min, noise_perc_max+1, noise_perc_inc):
                                noisify(img_fname, msk_fname, noise_perc)
                                if progressbar_enabled: pbar.update(1)
                        
        endtime_noise = time.time()
        noise_elapsedtime = round(endtime_noise-starttime_noise,3)
        
        if progressbar_enabled: pbar.close()  
        print("Adding noise to images complete.", no_noised_files,"files generated in",noise_elapsedtime,"seconds.")  			
        
        return noise_elapsedtime

def addmotion_images(imgaug_list):
        
        no_motion_files = int(len(imgaug_list))*5
        
        if progressbar_enabled: pbar = tqdm(total=no_motion_files,desc='Adding random motion to images')
        starttime_motion = time.time()
        
        for img_fname in imgaug_list:
 
                if labels_available:
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None

                for motion in range(1,6):
                        random_motion(img_fname, msk_fname, motion) 
                        if progressbar_enabled: pbar.update(1)
                        
        endtime_motion = time.time()
        motion_elapsedtime = round(endtime_motion-starttime_motion,3)
        
        if progressbar_enabled: pbar.close()  
        print("Adding motion to images complete.", no_motion_files,"files generated in",motion_elapsedtime,"seconds.")  			
        
        return motion_elapsedtime

def spike_images(imgaug_list):
        
        no_spiked_files = int(len(imgaug_list))*5
        
        if progressbar_enabled: pbar = tqdm(total=no_spiked_files,desc='Adding random spikes to images')
        starttime_spikes = time.time()
        
        for img_fname in imgaug_list:
                
                if labels_available:
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                
                for spike in range(1,5):
                        random_spikes(img_fname, msk_fname, spike) 
                        if progressbar_enabled: pbar.update(1)
                        
        endtime_spikes = time.time()
        spiking_elapsedtime = round(endtime_spikes-starttime_spikes,3)
        
        if progressbar_enabled: pbar.close()  
        print("Adding spikes to images complete.", no_spiked_files,"files generated in",spiking_elapsedtime,"seconds.")  

def blur_images(imgaug_list):
        
        no_blurred_files = int(len(imgaug_list))*4
        
        if progressbar_enabled: pbar = tqdm(total=no_blurred_files, desc='Randomly blurring images')
        starttime_blur = time.time()
        
        for img_fname in imgaug_list:
                if labels_available:
                        msk_fname = img_fname.replace("/brains/","/target_labels/").replace("_t1_","_labels_")
                else: msk_fname = None
                
                for b in range(1,4):
                        blur(img_fname, msk_fname, b) 
                        if progressbar_enabled: pbar.update(1)
   
        endtime_blur = time.time()
        blur_elapsedtime = round(endtime_blur-starttime_blur,3)
        
        if progressbar_enabled: pbar.close()  
        print("Randomly blurring images complete.", no_blurred_files,"files generated in",blur_elapsedtime,"seconds.")  
         
#endregion augment dataset
"""
def mixaug_images(aug_idx, filename_suffix, mixaug_list):
        augfuncs_list = [rotate_images, noisify_images, addmotion_images, spike_images, blur_images]
        
        fname_augsuffix = filename_suffix.split('-')[aug_idx]
        [mixaug_list.remove(line) for line in mixaug_list if fname_augsuffix in line.split('_')[len(line.split('_'))-1].split('.')[0]]
        
        
        mixaug_keepline = sorted(random.sample(range(0, len(mixaug_list)-1), int((len(mixaug_list))/4)))
        
        mixaug_list2 = list()
        [mixaug_list2.append(mixaug_list[line_no]) for line_no in range(0, len(mixaug_list)-1) if line_no in mixaug_keepline]
        
        augfuncs_list[aug_idx](mixaug_list2)
"""

def augment_data(data_dir, augtypes_in, output_dir):
        """
        # Input arguments: data directory, output directory, file type (nii or mnc)
        # If overwrite is true, delete any previously created output directores or files
        # Create an empty bool matrix to assign augmentation methods to perform in the
        # order of: rotation (r), noise (n), motion (m), spiking (s), and blurring (b)
        # Rotate all images. Then start remaining augmentation processes in parallel.
        """
        global indivaug_done, imgio_type
         
        #region initialize directories
        if is_overwrite and (os.path.exists(output_dir)): 
                shutil.rmtree(output_dir)  
        
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
        
        filename_suffix = "rC0000-n00-m00-s00-b00"	
        
        if not augtypes_in == None:        
                augtypes = np.zeros(5,dtype=bool)
                for type in augtypes_in: 
                        try:
                                if not type in ['r','n','m','s','b']: raise InputError()
                                else: augtypes[['r','n','m','s','b'].index(type)]=True
                        except InputError as e: print("Invalid input {}. See --help.".format(type))
            
        for fname in files_list:
                #region check file extension  
                try:
                        if not fname.endswith((".nii",".mnc")): 
                                raise FilenameError()
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
        
        if augtypes_in == None:
                return    
                                     
        imgaug_list = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(output_dir,"brains")), x),os.listdir(os.path.join(output_dir,"brains")))) 
        
        #region augmentation - individual
        """
        # If an augmentation method is selected, start respective process
        # Images are first rotated. Other augmenation methods are applied 
        # afterwards to the new dataset (including originals and rotated)
        """
        if augtypes[0]: 
                rot_elapsedtime = rotate_images(imgaug_list)
        
        if augtypes[1]: 
                noise_elapsedtime = noisify_images(imgaug_list) 
                
        if augtypes[2]: 
                motion_elapsedtime = addmotion_images(imgaug_list)
        
        if augtypes[3]: 
                spiking_elapsedtime = spike_images(imgaug_list)        
                
        if augtypes[4]: 
                blur_elapsedtime = blur_images(imgaug_list)       
        
        indivaug_done = True
        #endregion augmentation - individual          
        
        #region augmentation - mixed
        """
        For half of the augmented files, For each augmentation type (aug_idx)      
        If other augmentation types are selected (True),Then mix all selected 
        augmentation types (type_idx), except for origina/unaugmented images                                                                                                                                                  
        
        if mix_augmethods and indivaug_done and (len(np.where(augtypes)[0])>1):
                
                indivaug_list = list(map(lambda x: os.path.join(os.path.abspath(os.path.join(output_dir,"brains")), x),os.listdir(os.path.join(output_dir,"brains")))) 
                
                mixaug_list = indivaug_list
                augidx_list = list(np.where(augtypes)[0])
                
                if augtypes[0] and (0 in augidx_list): augidx_list.remove(0)
                if augtypes[2] and (2 in augidx_list): augidx_list.remove(2) ; augidx_list.insert(0,2)
                        
                [mixaug_list.remove(line) for line in mixaug_list if (filename_suffix.split('-') == line.split('_')[len(line.split('_'))-1].split('.')[0].split('-'))]
                
                mixaug_randidx = random.sample(range(0, len(mixaug_list)), len(mixaug_list))
                mixaug_randlist = list()
                [mixaug_randlist.append(mixaug_list[idx]) for idx in mixaug_randidx]
                
                pool = multiprocessing.Pool(processes=len(augidx_list))
                func_inputs = partial(mixaug_images, filename_suffix=filename_suffix, mixaug_list=mixaug_randlist)
                pool.map(func_inputs, augidx_list[1:len(augidx_list)])
        """                                         
def main():
        global data_dir, output_dir
  
        augment_data(data_dir = get_args().input_dir,
                     augtypes_in = get_args().aug_types,
                     output_dir = get_args().output_dir)
                      
if __name__ == "__main__":
    main()