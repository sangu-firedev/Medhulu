from joblib import Parallel, delayed
import ants 
import os

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

