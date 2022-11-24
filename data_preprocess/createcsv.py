

import os
import pandas as pd
import csv

path = r"D:\concile\data\dataset\new_final_data"
csv_path = r"D:\concile\myall\9recognization\basic recognition\medical\classification12\classification12\210.csv"
# dat = pd.read_csv(csv_path)
lists = []
for root, dir, files in os.walk(path):
    for file in files:
        if file.endswith(".nii.gz"):
            filesplit = file.split("_")
            filesplit2 = file.split(".")
            if "duoxing" in filesplit:
                list = [filesplit2[0], "1", "4", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "1", "4", "", "", "", ""])

            elif "jidi" in filesplit:
                list = [filesplit2[0], "1", "5", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "1", "5", "", "", "", ""])
            elif "warin" in filesplit:
                list = [filesplit2[0], "1", "6", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "0", "6", "", "", "", ""])
            elif "xianyang" in filesplit:
                list = [filesplit2[0], "0", "0", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "0", "0", "", "", "", ""])
            elif "xianpao" in filesplit:
                list = [filesplit2[0], "0", "1", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "0", "1", "", "", "", ""])
            elif "nianye" in filesplit:
                list = [filesplit2[0], "0", "2", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "0", "2", "", "", "", ""])
            elif "shangpi" in filesplit:
                list = [filesplit2[0], "0", "3", ]
                lists.append(list)
                # with open(csv_path, "w") as csvfile:
                # writer = csv.writer(csvfile)

                # 先写入columns_name
                # writer.writerow(["index", "a_name", "b_name"])
                # 写入多行用writerows
                # writer.writerow([filesplit2[0], "0", "3", "", "", "", ""])

print(lists)
with open(csv_path, "w",newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["patient_name", "2label", "7label", "LR", "MD", "name", "gender", "score", "most_score", ])
    # 写入多行用writerows
    writer.writerows(lists)
