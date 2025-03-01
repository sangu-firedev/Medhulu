import matplotlib.pyplot as plt
import numpy as np

def _plot(self, slice='mid'):
    data = self.data

    if slice=='mid':
        axial_index = data.shape[2] // 2
        coronal_index = data.shape[1] // 2
        saggital_index = data.shape[0] // 2
    else:
        axial_index = slice[2]
        coronal_index = slice[1]
        saggital_index = slice[0]

    axial_mid = data[axial_index, :, :]
    coronal_mid = data[:, coronal_index, :]
    saggital_mid = data[:, :,saggital_index] 

    saggital_mid = np.rot90(saggital_mid, k=1)
    coronal_mid = np.rot90(coronal_mid, k=1)
    axial_mid = np.rot90(axial_mid, k=1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(axial_mid, cmap='gray', aspect='auto')

    axs[1].imshow(coronal_mid, cmap='gray', aspect='auto')

    axs[2].imshow(saggital_mid, cmap='gray', aspect='auto')
    fig.tight_layout(pad=2.5)
    fig.suptitle(f"{self.name}")
    plt.show()

def plot_files(folder_path):
    from utilities.load_files import load_files

    images = load_files(folder_path)

    for image in images:
        _plot(image)