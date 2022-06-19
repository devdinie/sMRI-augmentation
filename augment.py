import os
import argparse

import numpy as np

def get_args():
        parser = argparse.ArgumentParser(description="sMRI-augmentation", add_help=True,
                                 formatter_class=argparse.RawTextHelpFormatter)
        
        parser.add_argument("-data_dir", default = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/"), 
                            help="Path to input data directory with brains and if available, labels. Structure:\ndata\n |- brains\n |- target_labels")
        
        parser.add_argument("-aug_types", default = "r n s d g", 
                            help= "Methods to augment input data: first letter of method(s) separated with space.\n" \
                                + "Eg: to perform all augmentation methods: augment.py -aug_types r n s d g\n" \
                                + " r - rotation\t n - noise\n s - spiking\t d - deformation \n g - ghosting")
        
        return parser.parse_args()

        
def main():
        data_dir  = get_args().data_dir
        aug_types = get_args().aug_types.split(' ')
        
        augtypes_arr = ['r','n','s','d','g']
        
        for i in augtypes_arr:
                if any(
        #[aug_rotate, aug_noise, aug_spike, aug_deform, aug_ghost] = aug_types
        
        print("data directory", data_dir)
        print("data directory", tt)

if __name__ == "__main__":
    main()
