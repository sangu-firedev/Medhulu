import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from utilities.skull_stripping import _skull_strip 

class LoadImage():
    
    def __init__(self, path):
        self.path = path
        self.data = self.get_data()
    
    def get_img(self):
        img = nib.load(self.path)
        return img
    
    def get_data(self):
        return self.get_img().get_fdata()

    def plot(self):
        data = np.rot90(self.data, k=1)
        axial_index = data.shape[0] // 2
        coronal_index = data.shape[1] // 2
        saggital_index = data.shape[2] // 2

        axial_mid = data[axial_index, :, :]
        coronal_mid = data[:, coronal_index, :]
        saggital_index_mid = data[:, :,saggital_index] 

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(axial_mid, cmap='gray')
        axs[1].imshow(coronal_mid, cmap='gray')
        axs[2].imshow(saggital_index_mid, cmap='gray')

        plt.show()

    def skull_strip(self, output_dir, load_nib=True):

        _skull_strip(self.path, output_dir)

        if load_nib == True:
            return LoadImage(output_dir)

