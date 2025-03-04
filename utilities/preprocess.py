from joblib import Parallel, delayed
import numpy as np
import ants 
import nibabel as nib
import os
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm

from src.main import LoadImage
from utilities.load_files import save_nifti 


#########################################
def register(input_folder, output_folder, fixed_image, filename, transform):
    """
    Process :
        Co-registering all the MRI images from the input_folder to common reference fixed_image

    Inputs :
        input_folder : str
        output_folder : str
        fixed_image: ANTsImage 
        filename: str
        transform: str

    Outputs :
        None : saves the Registered image to the output folder 

    """
    moving_path = os.path.join(input_folder, filename)
    moving_image = ants.image_read(moving_path)

    reg = ants.registration(fixed_image, moving_image, type_of_transform=transform)

    output_path = os.path.join(output_folder, f"coreg_{filename}")
    ants.image_write(reg['warpedmovout'], output_path)
    print(f"Registered : {filename}")

#########################################
def register_images(input_folder, output_folder, fixed_image, transform, thread=4):
    fixed = ants.image_read(fixed_image)
    os.makedirs(output_folder, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(("nii.gz", "mgz", "img", "img.gz"))]

    Parallel(n_jobs=thread)(delayed(register)(input_folder, output_folder, fixed, f, transform) for f in file_list)

#########################################
def zscore_normalization(self):
    """ return Z-Score normalized image (LoadImage) """
    mri = self.data
    affine = self.affine

    mean_int = np.mean(mri)
    std_int = np.std(mri)

    mri_norm = (mri - mean_int) / std_int
    return LoadImage(data=mri_norm, affine=affine)

#########################################
def min_max_normalization(self):
    """ return Min-Max normalized image (LoadImage) """
    mri = self.data
    affine = self.affine
    
    mri_min = np.min(mri)
    mri_max = np.max(mri)

    mri_minmax = (mri - mri_min)/(mri_max - mri_min)
    return LoadImage(data=mri_minmax, affine=affine)
    
#########################################
def resample(self, shape):
    """ return Resampled image (LoadImage) to desired shape """

    mri = self.data
    affine = self.affine

    zoom_factors = np.array(shape)/np.array(mri.shape)
    resampled_data = zoom(mri, zoom_factors, order=3)
    return LoadImage(data=resampled_data, affine=affine)

#########################################
def N4_bias_field_correction(self):
    """ 
    return image (LoadImage) with N4 Bias Field Corrected, Maintains uniform intensity, helps to get good extraction in 
    extract_gm_wm_csf() function
    """

    ants_img = ants.from_numpy(self.data)
    mri_correction = ants.n4_bias_field_correction(ants_img)
    img = mri_correction.numpy()
    return LoadImage(data=img, affine=self.affine)

#########################################
def extract_gm_wm_csf(self):
    """
    Process :
        extracting the CSF, Grey Matter & White Matter from the Raw MRI image using ANTS.antropos as returning them seperately 

    Inputs :
        self : LoadImage

    Outputs :
        (csf_img, gm_img, wm_img) :  Tuple

    """

    img = ants.from_numpy(self.data)
    mask = ants.get_mask(img)
    extraction = ants.atropos(a=img, m='[0.2,1x1x1]', c='[2,0]', i='kmeans[3]', x=mask) 
    csf_img = LoadImage(data=extraction['probabilityimages'][0].numpy(), affine=self.affine)
    gm_img = LoadImage(data=extraction['probabilityimages'][1].numpy(), affine=self.affine)
    wm_img = LoadImage(data=extraction['probabilityimages'][2].numpy(), affine=self.affine)
    return csf_img, gm_img, wm_img 

#########################################
def get_labelled_mask(csf_img, gm_img, wm_img, affine):
    """Labelling different mask images with appropriate values and return the combination of all three as LoadImage image"""

    # Intensity Normalization
    csf_n = (csf_img.data > 0.5).astype(np.uint8)
    gm_n = (gm_img.data > 0.5).astype(np.uint8)
    wm_n = (wm_img.data > 0.5).astype(np.uint8)

    # Labelling different masks with integer values
    masked_label = csf_n * 1 + gm_n * 2 + wm_n * 3

    return LoadImage(data=masked_label, affine=affine)

#########################################
def preprocess_file(img, output_path):
    """ 
    Run min_max_normatlization() -> N4_bias_field_correction() -> extract_gm_wm_csf() -> get_labelled_mask() preprocessing
    operations on the input image

    Note : Can include skull stripping if you can import the set a pytorch model using SynthStrip weights and run inference through it
    """

    img_norm = min_max_normalization(img)
    n4_img_norm = N4_bias_field_correction(img_norm)
    csf_img, gm_img, wm_img = extract_gm_wm_csf(n4_img_norm)
    label_mask = get_labelled_mask(csf_img, gm_img, wm_img, img.affine)

    file_name = f"{img.name}.label"
    output_path = f"{output_path}/{file_name}.nii.gz"

    save_nifti(label_mask, output_path) 

#########################################
def preproces_pipeline(input_path, output_path, workers, batch_size):
    """
    Process :
        Running the process_file() function on all the files present in the input_path directory and saving them in 
        the output_path folder

    Inputs :
        input_path: str 
        output_path: str 
        workers : int 
        batch_size : int 

    Outputs :
        None : Saves the processed images to the output_path directory 

    Note : Adjust workers and batch_size according to your hardware capacity
    """

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
    
    print("Preprocessing completed!")

#########################################
def load_files(file_batch):
    """Load only the given batch of files to avoid memory overuse."""
    return [LoadImage(file_path) for file_path in file_batch]  # Load batch-size files at a time

#########################################
def list_files(folder_path):
    """List only valid files from the directory (no loading yet)."""
    return sorted([
        os.path.join(folder_path, file) 
        for file in os.listdir(folder_path) 
        if file.endswith(("nii.gz", "mgz", "img", "img.gz"))
    ])

#########################################
def batch_iterator(file_list, batch_size):
    """Yield successive batch-sized chunks from the list of files."""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]