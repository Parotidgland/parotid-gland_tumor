import os
import shutil



path = r"D:\concile\data\bingli"
new_file  = r"D:\concile\data\bingli\new_data"

for root, dir, files in os.walk(path):
    for file in files:
        if file.endswith("label_1.nrrd"):
            print(root)
            old_name = root + os.sep + file
            print(old_name)
            new_file_name = root.split(os.sep)[-2] + '_' + root.split(os.sep)[-1] + ".nrrd"
            new_name = new_file + os.sep + root.split(os.sep)[-3] + os.sep + new_file_name
            print(new_name)
            shutil.copy(old_name, new_name)