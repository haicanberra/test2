import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import dicom_global_params

# Function: Read images
# path: Directory
# size: Size of images, size = (N, M)
# OS: Name of OS
# Return: List of image matrices (list of numpy arrays)
def read_images(path, size = (256, 256), OS = 'Windows'):
    I_plains = os.listdir(path)
    N_files = len(I_plains)
    kI = []
    str_Fnames = []

    for i in range(0, N_files):
        str = I_plains[i]
        str_Fnames.append(str)
        Ip = cv2.imread(path + '\\' + str if OS == 'Windows' else path + '/' + str)
        Ip_resized = cv2.resize(Ip, size, interpolation = cv2.INTER_AREA)
        kI.append(Ip_resized)
    return kI, str_Fnames


# Function: Show images
# kI: List of image matrices
# suptitle: Super title
# str_Fnames: List of image names
# size: Size of a image window
# rows: Number of rows presented
# cols: Number of columns presented
def show_images(kI, suptitle, str_Fnames, size = (10, 10), rows = 3, cols = 3):
    fig = plt.figure(figsize = size)

    for i in range(len(kI)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(kI[i].astype(np.uint16))
        plt.title(str_Fnames[i])
    fig.suptitle(suptitle, size = 16)
    fig.tight_layout(pad=1.0)
    plt.show()


# Function: Split images
def split_images(kI, list_num_bits_pre):
    kI_merge = []
    temp = []
    for i in range(len(kI)):
        if (list_num_bits_pre[i] == dicom_global_params.NB_max):
            temp.append(np.uint8(kI[i]      ))
            temp.append(np.uint8(kI[i] >> 8 ))
            temp.append(np.uint8(kI[i] >> 16))
            merged_image = cv2.merge(temp)
            kI_merge.append(merged_image)
            temp.clear()
        else:
            kI_merge.append(kI[i])
    return kI_merge


# Function: Save image into directory
def save_images(kC, folder_path, str_Fnames, grey_image):
    for i in range(len(kC)):
        if (grey_image[i] == 0):
            cv2.imwrite(os.path.join(folder_path, str_Fnames[i]), cv2.cvtColor(kC[i], cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(folder_path, str_Fnames[i]), kC[i])
