from joblib import Parallel, delayed
import numpy as np
import ants 
import nibabel as nib
import os
from scipy.ndimage import zoom
#from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from src.main import LoadImage
from utilities.load_files import save_nifti 

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

def extract_gm_wm_csf(self):

    img = ants.from_numpy(self.data)
    mask = ants.get_mask(img)
    extraction = ants.atropos(a=img, m='[0.2,1x1x1]', c='[2,0]', i='kmeans[3]', x=mask) 
    csf_img = LoadImage(data=extraction['probabilityimages'][0].numpy(), affine=self.affine)
    gm_img = LoadImage(data=extraction['probabilityimages'][1].numpy(), affine=self.affine)
    wm_img = LoadImage(data=extraction['probabilityimages'][2].numpy(), affine=self.affine)
    return csf_img, gm_img, wm_img 

def get_labelled_mask(csf_img, gm_img, wm_img, affine):

    # Intensity Normalization
    csf_n = (csf_img.data > 0.5).astype(np.uint8)
    gm_n = (gm_img.data > 0.5).astype(np.uint8)
    wm_n = (wm_img.data > 0.5).astype(np.uint8)

    # Labelling different masks with integer values
    masked_label = csf_n * 1 + gm_n * 2 + wm_n * 3

    return LoadImage(data=masked_label, affine=affine)

def preprocess_file(img, output_path):
    img_norm = min_max_normalization(img)
    n4_img_norm = N4_bias_field_correction(img_norm)
    csf_img, gm_img, wm_img = extract_gm_wm_csf(n4_img_norm)
    label_mask = get_labelled_mask(csf_img, gm_img, wm_img, img.affine)

    file_name = f"{img.name}.label"
    output_path = f"{output_path}/{file_name}.nii.gz"

    save_nifti(label_mask, output_path) 

def preproces_pipeline(input_path, output_path, workers, batch_size):

    os.makedirs(output_path, exist_ok=True)

    all_files = list_files(input_path)

    print(f"Preprocessing {len(all_files)} Images with {workers} Workers")

    for batch_num, file_batch in enumerate(batch_iterator(all_files, batch_size), start=1):
        print(f"\n🔹 Processing Batch {batch_num}")

        # **Step 3: Load only batch-size images into memory**
        images = load_files(file_batch)  # Now only loading `batch_size` images

        partial_preprocess_func = partial(preprocess_file, output_path=output_path)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(partial_preprocess_func, img): img for img in images}

            for _ in tqdm(as_completed(futures), total=len(images), desc=f"Batch {batch_num} Progress"):
                pass  # Just iterating to track completion

    #partial_preprocess_func = partial(preprocess_file, output_path=output_path)

    #with ThreadPoolExecutor(max_workers=threads) as executor:
    #    tqdm(executor.map(partial_preprocess_func, images), total=len(images), desc="Processing Images")
    #with ProcessPoolExecutor(max_workers=workers) as executor:
    #    futures = {executor.submit(partial_preprocess_func, img): img for img in images}

    #    for _ in tqdm(as_completed(futures), total=len(images), desc="Processing Images"):
    #        pass
    
    print("Preprocessing completed!")

def load_files(file_batch):
    """Load only the given batch of files to avoid memory overuse."""
    return [LoadImage(file_path) for file_path in file_batch]  # Load batch-size files at a time

def list_files(folder_path):
    """List only valid files from the directory (no loading yet)."""
    return sorted([
        os.path.join(folder_path, file) 
        for file in os.listdir(folder_path) 
        if file.endswith(("nii.gz", "mgz", "img", "img.gz"))
    ])

def batch_iterator(file_list, batch_size):
    """Yield successive batch-sized chunks from the list of files."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]