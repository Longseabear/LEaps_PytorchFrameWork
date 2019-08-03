import numpy as np
import os

# arg -> dataset path
dataset_path = "/media/it-315/hard/data_256"
output_path = "../file_list.txt"

def fetch_img_list(dir):
    filenames = os.listdir(dir)
    file_list = []
    for filename in filenames:
        file_path = os.path.join(dir,filename)
        if os.path.isdir(file_path):
            file_list += fetch_img_list(file_path)
        else:
            file_list.append(file_path + "\n")
    return file_list

if __name__ == "__main__":
    file_list = fetch_img_list(dataset_path)
    print("file list {} ".format(file_list), len(file_list))

    with open(output_path,'w') as f:
        f.writelines(file_list)

