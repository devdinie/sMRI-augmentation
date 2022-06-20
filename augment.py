import os
import shutil
import argparse
from weakref import ref

import scipy
import scipy.ndimage
import numpy as np
import SimpleITK as sitk

is_overwrite = True # If true, any data previously processed will be lost
is_resize_outputs = True

output_imgsize = (144,144,144)

#region custom errors
class FilenameError(Exception):
        pass
#endregion custom errors

#region parse arguments
def get_args():
        parser = argparse.ArgumentParser(description="sMRI-augmentation", add_help=True,
                                 formatter_class=argparse.RawTextHelpFormatter)
        
        parser.add_argument("-input_dir", default = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/")), 
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
def resample_img(img_file, msk_file, output_imgsize):
        
        ref_file = sitk.GetImageFromArray(np.zeros((output_imgsize)))
        ref_file.SetOrigin(img_file.GetOrigin())
        ref_file.SetDirection(img_file.GetDirection())
        
        ref_physicalsize = np.zeros(img_file.GetDimension())
        ref_physicalsize[:] = [(sz-1)*spc if sz*spc > mx  else mx for sz, spc, mx in zip(img_file.GetSize(), img_file.GetSpacing(), ref_physicalsize)]
        
        ref_spacing   = [ phys_sz/(sz-1) for sz, phys_sz in zip(output_imgsize, ref_physicalsize) ]
        ref_file.SetSpacing(ref_spacing)
        
        img_resampled = sitk.Resample(img_file, ref_file)
        if msk_file:
                msk_resampled = sitk.Resample(msk_file, img_file)
                return img_resampled, msk_resampled
        else:
                return img_resampled, None
#endregion utilities

def rotate(img, msk, angle, axes):
        
        img_arr = sitk.GetArrayFromImage(img)
        msk_arr = sitk.GetArrayFromImage(msk)
        
        if angle < 0 : angle += 360
        dir = 'C' if angle <= 300 else 'A'
        
        imgrot_arr = scipy.ndimage.rotate(img_arr, angle, axes=axes, reshape=True)
        mskrot_arr = scipy.ndimage.rotate(msk_arr, angle, axes=axes, reshape=True)
        
        imgrot_file = sitk.GetImageFromArray(imgrot_arr)
        mskrot_file = sitk.GetImageFromArray(mskrot_arr)
        
        return imgrot_file, mskrot_file, dir
        
def main():
        #region get input arguments and initialize directories
        """
        get path to input data and augmentation methods to be used if no input argument:
        # default path is used and all types of augmentation methods are performed 
        """
        data_dir    = get_args().input_dir
        augtypes_in = get_args().aug_types
        output_dir  = get_args().output_dir
        
        # if overwrite, previous output directories are deleted
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
           
        files_list = os.listdir(os.path.join(data_dir,"brains"))
        no_files   = len(files_list)
        
        filename_suffix = "rC00-n00-d0-sp0000-gh0"	
        
        """
        empty bool matrix to assign augmentation methods to perform in the order:
        rotation, noise, spiking, deformation and ghosting
        if augmentation method is input, corresponding index in bool matrix is set to true
        """
        augtypes = np.zeros(5,dtype=bool)
        for type in augtypes_in: augtypes[['r','n','s','d','g'].index(type)]=True
        #endregion get input arguments and initialize directories
        
        #region augmentation - individual
        """
	perform each augmentation method on each image, only one method at a time. eg. if inputs: r n, 
 	each image will have a set of rotated images with (no other methods applied) and a set of images 
  	with noise added (with no other methods, including rotation)
 	"""
        for img_fname in files_list:
                
                #region get files and file info
                """
                get file info: extension, subject id from filename(s)
                raise error if files are not nii or mnc but keep processing
                load (brain) images and if available, labels
                """
                try:
                        if not img_fname.endswith((".nii",".mnc")): raise FilenameError(img_fname.split(".")[1])
                        else: imgio_type = "NiftiImageIO" if img_fname.split(".")[1] == "nii" else "MINCImageIO"
                except FilenameError as e:  
                        print("Invalid extension. File must be .mnc or .nii. {} not processed".format(img_fname))
                
                subject_id   = img_fname.split("_")[0]  
                imgaug_fname = os.path.abspath(img_fname).replace("t1","t1_"+filename_suffix)
                img_file     = sitk.ReadImage(os.path.join(data_dir,"brains",img_fname), imageIO=imgio_type)
                
                if labels_available:
                        msk_fname   = img_fname.replace("t1","labels")
                        mskaug_fname= imgaug_fname.replace("t1","labels")
                        msk_file    = sitk.ReadImage(os.path.join(data_dir,"target_labels",msk_fname), imageIO=imgio_type)
                
                if is_resize_outputs:
                        img_file, msk_file = resample_img(img_file, msk_file, output_imgsize)
                        
                #endregion get files and file info 
                       
                #region augmentation - individual: rotation
                """
                if rotation is a selected augmentation method, set min and max rotation angles 
                and list axis of rotation to be used. rotate each image along each axis and in 
                each of these cases, rotate the image from min to max angle in steps of rot_inc
                """
                """
                if augtypes[0]:
                        angle_limit_neg = -6 ; angle_limit_pos =  6
                        rot_inc  = 3
                        all_axes = [(1, 0), (1, 2), (0, 2)]
                        
                        for rot_angle in range(angle_limit_neg,angle_limit_pos+1,rot_inc):
                                if not rot_angle == 0:
                                        for axes in all_axes:
                                                imgaug_file, mskaug_file, dir = rotate(img_file, msk_file, rot_angle, axes)
                                                dir_ang = dir+str(f'{np.abs(rot_angle):02}')
                                                                                            
                                imgaug_fname = os.path.join(output_dir,"brains", imgaug_fname.replace("rC00",dir_ang))
                                mskaug_fname = os.path.join(output_dir,"target_labels", imgaug_fname.replace("rC00",dir_ang))
                                                
                                #sitk.WriteImage(imgaug_file, imgaug_fname)
                                #sitk.WriteImage(mskaug_file, mskaug_fname)
                """
        	#endregion augmentation - individual: rotation
        
        #region augmentation - individual: noise
        #endregion augmentation - individual: noise
        
        #region augmentation - individual: spiking
        #endregion augmentation - individual: spiking
        
        #region augmentation - individual: deformation
        #endregion augmentation - individual: deformation
        
        #region augmentation - individual: ghosting
        #endregion augmentation - individual: ghosting
        
        #endregion augmentation - individual
        
if __name__ == "__main__":
    main()
