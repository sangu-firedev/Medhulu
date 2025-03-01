import subprocess
import os
import nibabel as nib

def _skull_strip(input_path, output_path):
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)

    os.makedirs(output_dir, exist_ok=True)

    bash = [
        "docker", "run", "--rm",
        "-v", f"{os.path.dirname(input_path)}:/input",
        "-v", f"{output_dir}:/output",
        "freesurfer/synthstrip",
        "-i", f"/input/{os.path.basename(input_path)}",
        "-o", f"/output/{os.path.basename(output_path)}",
    ]

    try:
        subprocess.run(bash, check=True)
        print(f"Skull-stripped image saved at: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Synthstrip: {e}")
    