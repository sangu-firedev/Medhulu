import os
from pandas import DataFrame
from src.main import LoadImage
import nibabel as nib

def load_files(folder_path):

    folder = os.listdir(folder_path)
    images_list = []

    for file in folder:
        file_path = os.path.join(folder_path, file)
        if file.endswith(("nii.gz", "mgz", "img", "img.gz")):
            images_list.append(LoadImage(file_path))
    
    return images_list

def save_nifti(img, output_path):
    nib_img = nib.Nifti1Image(img.data, img.affine)
    nib.save(nib_img, output_path)

def multiple_file_handler(folder_path : str) -> DataFrame :

    """

    Process :
        This function takes full path of the meta folder of ecg data, and extract all path, file path and folder name from
        ecg meta data.

    Inputs :
        name    : folder_path
        type    : python str
        content : full path ecg meta data

    Outputs :
        name    : folder_path_df
        type    : pandas dataframe
        content : dataframe have path of the ecg files, folder name, file name and extension of files.

    i.e. : 

    # path of meta data
    path = "abc/xyz/123/456/789/"

    # extract full-path and roots
    fn_return = multiple_file_handler(path)  

    """
    
    # store file info dict 
    folder_path_ls = [ ]

    # extract all paths and folders of given path
    for root, _, files in os.walk(folder_path) :

        # loop all files in folders and sub folders
        for file in files :

            # get full path of file
            full_folder_path = os.path.join(root, file)

            # extract folder name
            path_head_tail = os.path.split(root)
            folder_name = path_head_tail[1]

            # extract file extension
        
            file_extension = file.split(".")[-1]

            # store data into dict
            file_info_dict = {"File_name" : file, 
                              "File_extension" : file_extension, 
                              "Folder_name" : folder_name, 
                              "folder_path" : full_folder_path}

            # store dict into list
            folder_path_ls.append(file_info_dict)

    # convert list of dicts into pandas dataframe
    folder_path_df = DataFrame(folder_path_ls)

    return folder_path_df

def get_file_ext(file_path):
    extensions = []
    while True:
        file_name, file_ext = os.path.splitext(file_path)
        if file_ext:
            extensions.append(file_ext)
            file_path = file_name
        else:
            break
    return ''.join(reversed(extensions))