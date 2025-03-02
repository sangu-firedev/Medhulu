import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import gzip
import shutil
import tempfile

from utilities.skull_stripping import _skull_strip 
from utilities.plotting import _plot

class LoadImage():
    
    def __init__(self, path=None, data=None, affine=None):

        if path is not None:
            self.path = path
            self.name = os.path.basename(path).split(".")[0]
            self.data = self.get_data()
            self.header = self.get_img().header
            self.affine = self.get_img().affine
        if data is not None and affine is not None:
            self.path = None
            self.name = None
            self.data = data
            self.header = None
            self.affine = affine
    
    def get_img(self):

        if os.path.basename(self.path).endswith("img.gz"):
            with tempfile.TemporaryDirectory() as tmp_dir: 
                decompressed_img_path = tmp_dir + f'/{self.name}.img'
                
                with gzip.open(self.path, 'rb') as f_in:
                    with open(decompressed_img_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                hdr_path = self.path.replace('.img.gz', '.hdr')
                shutil.copy(hdr_path, decompressed_img_path.replace('.img', '.hdr'))
            
                img = nib.load(decompressed_img_path)
                img = nib.as_closest_canonical(img)
                return img

        img = nib.load(self.path)
        img = nib.as_closest_canonical(img)
        return img
    
    def get_data(self):
        return self.get_img().get_fdata()

    def plot(self, slice=None):
        _plot(self, slice=slice), 

    def skull_strip(self, output_dir, load_nib=True):
        from utilities.load_files import get_file_ext

        _skull_strip(self, output_dir)

        if load_nib == True:
            return LoadImage(f'{output_dir}/{self.name}_ss{get_file_ext(self.path)}')

