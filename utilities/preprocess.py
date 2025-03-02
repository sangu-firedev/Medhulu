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
    return LoadImage(data=mri_norm, affine=affine)

def min_max_normalization(self):
    mri = self.data
    affine = self.affine
    
    mri_min = np.min(mri)
    mri_max = np.max(mri)

    mri_minmax = (mri - mri_min)/(mri_max - mri_min)
    return LoadImage(data=mri_minmax, affine=affine)
    
def resample(self, shape):

    mri = self.data
    affine = self.affine

    zoom_factors = np.array(shape)/np.array(mri.shape)
    resampled_data = zoom(mri, zoom_factors, order=3)
    return LoadImage(data=resampled_data, affine=affine)

def N4_bias_field_correction(self):

    ants_img = ants.from_numpy(self.data)
    mri_correction = ants.n4_bias_field_correction(ants_img)
    img = mri_correction.numpy()
    return LoadImage(data=img, affine=self.affine)

# Not using this function, Cause of quality issues
def extract_gm_wm_csf(self):

    img = ants.from_numpy(self.data)
    mask = ants.get_mask(img)
    extraction = ants.atropos(a=img, m='[0.2,1x1x1]', c='[2,0]', i='kmeans[3]', x=mask) 
    csf_img = LoadImage(data=extraction['probabilityimages'][0].numpy(), affine=self.affine)
    gm_img = LoadImage(data=extraction['probabilityimages'][1].numpy(), affine=self.affine)
    wm_img = LoadImage(data=extraction['probabilityimages'][2].numpy(), affine=self.affine)
    return csf_img, gm_img, wm_img 

def transform()
