from joblib import Parallel, delayed
import numpy as np
import ants 
import nibabel as nib
import os
from scipy.ndimage import zoom
from src.main import LoadImage

def register(input_folder, output_folder, fixed_image, filename, transform):
    moving_path = os.path.join(input_folder, filename)
    moving_image = ants.image_read(moving_path)

    reg = ants.registration(fixed_image, moving_image, type_of_transform=transform)

    output_path = os.path.join(output_folder, f"coreg_{filename}")
    ants.image_write(reg['warpedmovout'], output_path)
    print(f"Registered : {filename}")

def register_images(input_folder, output_folder, fixed_image, transform, thread=4):
    fixed = ants.image_read(fixed_image)
    os.makedirs(output_folder, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(("nii.gz", "mgz", "img", "img.gz"))]

    Parallel(n_jobs=4)(delayed(register)(input_folder, output_folder, fixed, f, transform) for f in file_list)

def zscore_normalization(self):
    mri = self.data
    affine = self.affine

    mean_int = np.mean(mri)
    std_int = np.std(mri)

    mri_norm = (mri - mean_int) / std_int
    return LoadImage(mri_norm, affine)

def min_max_normalization(self):
    mri = self.data
    affine = self.affine
    
    mri_min = np.min(mri)
    mri_max = np.max(mri)

    mri_minmax = (mri - mri_min)/(mri_max - mri_min)
    return LoadImage(mri_minmax, affine)
    
def resample(self, shape):

    mri = self.data
    affine = self.affine

    zoom_factors = np.array(shape)/np.array(mri.shape)
    resampled_data = zoom(mri, zoom_factors, order=3)
    return LoadImage(resampled_data, affine)

def N4_bias_field_correction(self):
    ants_img = ants.from_numpy(self.data)
    mri_correction = ants.n4_bias_filed_correction(ants_img)
    img = mri_correction.numpy()
    return LoadImage(img, self.affine)

def extract_gm_wm_csf(input_folder, output_folder, ):
    os.makedirs(output_folder, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(("nii.gz", "mgz", "img", "img.gz"))]

    for file in file_list:
        file_path = os.path.join(input_folder, file)
        img = ants.image_read(file_path)

    fixed = ants.image_read(fixed_image)