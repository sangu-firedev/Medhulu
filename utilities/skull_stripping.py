import subprocess
import os
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor
from functools import partial


def _skull_strip(self, output_path, use_gpu=False):
    input_path = self.path
    from utilities.load_files import get_file_ext
    
    output_path = os.path.abspath(output_path)

    os.makedirs(output_path, exist_ok=True)

    # Have the output file with the same name as input with ss appended
    output_file_name= f'{self.name}_ss{get_file_ext(input_path)}'

    if use_gpu:
        bash = [
            "docker", "run", "--rm", "--gpus all"
            "-v", f"{os.path.dirname(input_path)}:/input",
            "-v", f"{output_path}:/output",
            "freesurfer/synthstrip",
            "-i", f"/input/{os.path.basename(input_path)}",
            "-o", f"/output/{output_file_name}", "-g"
        ]
    else:
        bash = [
            "docker", "run", "--rm",
            "-v", f"{os.path.dirname(input_path)}:/input",
            "-v", f"{output_path}:/output",
            "freesurfer/synthstrip",
            "-i", f"/input/{os.path.basename(input_path)}",
            "-o", f"/output/{output_file_name}",
        ]

    try:
        subprocess.run(bash, check=True)
        print(f"Skull-stripped image saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Synthstrip: {e}")

def _skull_strip_files(input_path, output_path, use_gpu=False, threads=1):
    from utilities.load_files import load_files

    images_list = load_files(input_path)

    _skull_strip_func = partial(_skull_strip, output_path=output_path)

    with ThreadPoolExecutor(max_workers=threads) as exe:

        exe.map(_skull_strip_func, images_list)