
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

old_data_path = r"D:\concile\data\bingli"
seg_path = r"D:\concile\data\bingli\new_data"
new_data_path = r"D:\concile\data\dataset\new_data"
new_segfile_path = r"D:\concile\data\dataset\new_seg_data"

for root, dir, files in os.walk(seg_path):
    for file in files:

        if file.endswith('.nrrd'):
            file_name = file.split('.')[0]
            #print(root)
            #print(file_name)
            file_name1 = file_name.split('_')[0]
            file_name2 = file_name.split('_')[1]
            pre_name = root.split('\\')[-1]
            old_segfile_name = root + os.sep + file
            #print(old_segfile_name)
            old_file_path = old_data_path + os.sep +pre_name + os.sep + file_name1 + os.sep + file_name2

            #print(old_file_path)

            for file_root, file_dir, file_files in os.walk(old_file_path):
                for file_file in file_files:
                    if file_file.endswith('Recon 2.nrrd'):

                        old_file_name = old_file_path + os.sep + file_file
                        file, opt = nrrd.read(old_file_name)
                        x, y, z = file.shape

                        seg_file, opt1 = nrrd.read(old_segfile_name)

                        x1, y1, z1 = seg_file.shape

                        #print(z)
                        #print(z1)

                        # if z is not z1:
                        #print(old_segfile_name)

                        num = z // 16
                        rest = z - num * 16
                        #random.seed(9001)
                        a = random.randint(0, rest)
                        #print(num)

                        #for j in range(num):
                            # fileq = file[:, :, j + a:z:num]
                            # segq = seg_file[:, :, j + a:z:num]
                            # # x_file, y_file, z_files = fileq.shape
                            # # for z_file in range(z_files):
                            # # fileqs = fileq[:, :, z_file]
                            # # fileq[:, :, z_file] = np.flipud(fileqs)
                            #
                            # print(fileq.shape)
                            # print(segq.shape)
                        for j in range(num + 1):
                            if num == 0:
                                fileq = file[:, :, 0:]
                                segq = seg_file[:, :, 0:]
                            else:
                                #num > 0:
                                if j < num:
                                    fileq = file[:, :, j * 16:16 + 16 * j]
                                    segq = seg_file[:, :, j * 16:16 + j * 16]
                                elif j == num:
                                    fileq = file[:, :, z - 16:z]
                                    segq = seg_file[:, :, z - 16:z]


                        #print(old_file_name)
                            #print(old_segfile_name)
                            # print(fileq.shape)
                            img = nib.Nifti1Image(fileq, np.eye(4))
                            # aa = name.split("\\")
                            # bb = aa[3].split(".")
                            # print(segq.shape)
                            #
                            seg_img = nib.Nifti1Image(segq, np.eye(4))
                            new_file_attribute = old_segfile_name.split('\\')[-2]
                            #print(new_file_attribute)
                            new_file_pre = old_segfile_name.split('\\')[-1]
                            new_file_pre = new_file_pre.split('.')[0]
                            #print(new_file_pre)
                            new_file_name = new_data_path + os.sep + new_file_attribute + os.sep + new_file_pre + "_" + str(j + 1) + ".nii.gz"
                            new_segfile_name = new_segfile_path + os.sep +  new_file_attribute + os.sep +new_file_pre + "_label_" + str(j + 1) + ".nii.gz"
                            print(new_file_name)
                            print(new_segfile_name)
                            nib.save(img, new_file_name)

                            nib.save(seg_img, new_segfile_name)
                        print("finished one split")




