import os
#import general.normalization as reg
from torch.utils.data import Dataset
import pandas as pd
import nibabel as nib
import random

class nii2tensor():
    def __init__(self,root_dir, data_file, transform = None):
        self.root_dir = root_dir
        self.data_file = data_file
        self.transform = transform
        path = self.root_dir + os.sep + "datalist" + os.sep + self.data_file
        #print(path)
        self.images , self.paths , self.labels = _load_files(path )
        #print(self.paths)
        print("loading")
    def __len__(self):
        path = self.root_dir + os.sep + "datalist" + os.sep + self.data_file
        #print(sum(1 for line in open(path)))
        return sum(1 for line in open(path))


    def __getitem__(self,idx):
        #path = self.root_dir + os.sep + "data" + os.sep + "basic_classification" + os.sep + self.data_file

        #df = open(path)
        #lines = df.readlines()
        #lst = lines[idx].split(",")
        #img_name = lst[0]
        #img_label2 = lst[1]
        #img_label7 = lst[2]
        #if img_label2 == '0':
            #label2 = 0
        #elif img_label2 == '1':
            #label2 = 1

        #label7 = int(img_label7)


        #img_path = self.root_dir + os.sep + "data" + os.sep + "dataset" + os.sep + img_name + ".nii.gz"
        #reader = sitk.ImageSeriesReader()
        #img_names = reader.GetGDCMSeriesFileNames(img_path)
        #reader.SetFileNames(img_names)
        #image = reader.Execute()
        #img = sitk.GetArrayFromImage(image)
        #img = reg.norm_img(image_array)
        #img = nib.load(img_path).get_fdata()

        real_img = self.images[idx]
        label2 = self.labels[idx]
        if label2 == '0':
            label2 = 0
        elif label2 == '1':
            label2 = 1
        #print(label2.type())
        if self.transform:
            real_img = self.transform(real_img)
            #label2 = torch.
        return real_img, label2




def _load_files(path):
    # global paths_data
    # global files_data
    # global labels_data
    #mode = "continous"
    #root_dir =
    files_data = []
    labels_data = []
    paths_data = []

    df = open(path)
    lines = df.readlines()
    for line in lines:
        line = line.split(",")

        #img_path = root_dir + os.sep + "data" + os.sep + "dataset" + os.sep + line[0] + ".nii.gz"
        img_path = "/data/xavieryang/classification/medical/new_final_data" + os.sep + line[0] + ".nii.gz"

        img = nib.load(img_path).get_fdata()
        img_label2 = line[1]
        img_label7 = line[2]

        #print(img_label2)
        #print(img_label7)
        #print(img_path)

        files_data.append(img)
        paths_data.append(img_path)
        labels_data.append(img_label2)

        #files_data , paths_data , labels_data = imgsplit(img , img_path , mode , files_data , labels_data , img_label2)

    return files_data , paths_data , labels_data
    #img = nib.load(line)
    #for files in os.listdir(dir):
        #path =


# def imgsplit(img , img_path , mode ,files_data ,labels , paths , img_label2):
#     global fileq
#     if mode == "lianxu":
#         file_w, file_h, file_z = img.shape
#         #file_w, file_h, file_z = file.shape
#         num = file_z // 16
#         rest = file_z - num * 16
#         # random.seed(9001)
#         # a = random.randint(0, rest)
#         # print(type(a))
#         # print(type(file_z))
#         # print(type(num))
#         # print(num)
#         # print(rest)
#         # print(num)
#         for j in range(num + 1):
#             if num == 0:
#                 fileq = img[:, :, 0:]
#                 #segq = seg_file[:, :, 0:]
#             elif num > 0:
#                 if j < num:
#                     fileq = img[:, :, j * 16:16 + 16 * j]
#                     #segq = seg_file[:, :, j * 16:16 + j * 16]
#                 elif j == num:
#                     fileq = img[:, :, file_z - 16:file_z]
#                     #segq = seg_file[:, :, file_z - 16:file_z]
#             files_data.append(fileq)
#             paths.append(img_path)
#             labels.append(img_label2)
#
#
#
#
#
#     elif mode =="lisan":
#         file_w, file_h, file_z = img.shape
#         num = file_z // 16
#         rest = file_z - num * 16
#         random.seed(9001)
#         a = random.randint(0, rest)
#
#     return files_data , paths , labels
