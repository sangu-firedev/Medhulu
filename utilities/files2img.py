import os
import argparse
import nibabel as nib




def file2img(input_dir, output_dir):
    os.listdir(input_dir)


img = nib.load(path)
data = img.get_fdata()