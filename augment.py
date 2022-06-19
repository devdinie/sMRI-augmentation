from fileinput import filename
import os
import argparse

import numpy as np
import SimpleITK as sitk

def get_args():
        parser = argparse.ArgumentParser(description="sMRI-augmentation", add_help=True,
                                 formatter_class=argparse.RawTextHelpFormatter)
        
        parser.add_argument("-data_dir", default = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/"), 
                            help="Path to input data directory with brains and if available, labels.\n"
                                + "filename format: brains- subjectid_t1.nii | labels- subjectid_labels.nii\n"
                                + "directory structure:\ndata\n |- brains\n |- target_labels")
        
        parser.add_argument("-aug_types", default = ['r','n','s','d','g'], nargs='+',
                            help= "Methods to augment input data: first letter of method(s) separated with space.\n"
                                + "Eg: to perform all augmentation methods: augment.py -aug_types r n s d g\n"
                                + " r - rotation\t n - noise\n s - spiking\t d - deformation \n g - ghosting")
        
        return parser.parse_args()

        
def main():
        #region get augmentation methods
        """
        get path to input data and augmentation methods to be used if no input argument:
        # default path is used and all types of augmentation methods are performed 
        """
        data_dir    = get_args().data_dir
        augtypes_in = get_args().aug_types
        
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
        #endregion get augmentation methods
        
        #region augmentation - individual
        """
	perform each augmentation method on each image, only one method at a time. eg. if inputs: r n, 
 	each image will have a set of rotated images with (no other methods applied) and a set of images 
  	with noise added (with no other methods, including rotation)
 	"""
        
        for imgFile in files_list:
                
                subject_id = os.path.basename(imgFile).split("_")[0]
                imgaugFile = os.path.abspath(imgFile).replace("t1","t1_"+filename_suffix)
                print(imgaugFile)
                
                if os.path.exists(os.path.join(data_dir,"target_labels")):
                        mskaugFile = imgaugFile.replace("t1","labels")
                        print(mskaugFile)
                
                #region augmentation - individual: rotation
                """
                set min and max angles for rotation and list axes to rotate
                rotate each image from angles min to max with steps of rot_inc
                """
                if augtypes[0]:
                        #region rotation - initialization
                        angle_limit_neg = -15 ; angle_limit_pos =  15
                        rot_inc  = 3
                        all_axes = [(1, 0), (1, 2), (0, 2)]
                        #endregion rotation - initialization
                        
                        #region rotation - rotate by angles
                        for rot_angle in range(angle_limit_neg,angle_limit_pos+1,rot_inc):
                                s=1#print(rot_angle)
                                
                        #endregion rotation - rotate by angles
                                
                print("-------")
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
