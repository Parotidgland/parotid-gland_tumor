
import numpy as np
import random
import math
from PIL import Image
from skimage.transform import resize
import skimage
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
import os


file_path = r"D:\concile\data\dataset\new_data"
segfile_path = r"D:\concile\data\dataset\new_seg_data"
proceeded_file_path = r"D:\concile\data\dataset\new_final_data"


for root, dir, files in os.walk(file_path):
    for file in files:
        file_path = root + os.sep + file
        print(file_path)
        seg_file_name_order = file.split('.')[0]
        #print(seg_file_name_order)
        #print(root)
        seg_file_name_name = seg_file_name_order.split('_')[0] + "_" + seg_file_name_order.split('_')[1]
        seg_file_name_order = seg_file_name_order.split('_')[-1]
        #print(seg_file_name_order)
        seg_file_name = segfile_path + os.sep + root.split('\\')[-1] + os.sep + seg_file_name_name + '_label_' + seg_file_name_order + '.nii.gz'
        print(seg_file_name)

        seg_file = nib.load(seg_file_name).get_fdata()
        final_data = nib.load(file_path).get_fdata()

        final_data = seg_file * final_data

        final_nii = nib.Nifti1Image(final_data, np.eye(4))

        final_nii_path = proceeded_file_path + os.sep + root.split('\\')[-1] + os.sep + seg_file_name_name + '_' + seg_file_name_order + '.nii.gz'
        print(final_nii_path)

        nib.save(final_nii, final_nii_path)






