import matplotlib.pyplot as plt
import numpy as np

def _plot(self, slice='mid', mask_list=None):
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

def plot_with_mask(self, masks):
    _plot(self)

    # Raw MRI 
    data = self.data

    axial_index = data.shape[2] // 2
    coronal_index = data.shape[1] // 2
    saggital_index = data.shape[0] // 2

    axial_mid = data[axial_index, :, :]
    coronal_mid = data[:, coronal_index, :]
    saggital_mid = data[:, :,saggital_index] 

    saggital_mid = np.rot90(saggital_mid, k=1)
    coronal_mid = np.rot90(coronal_mid, k=1)
    axial_mid = np.rot90(axial_mid, k=1)

    # Mask MRI
    csf = masks[0].data
    gm = masks[1].data
    wm = masks[2].data

    saggital_index = csf.shape[0] // 2
    coronal_index = csf.shape[1] // 2
    axial_index = csf.shape[2] // 2

    saggital_mid_csf = csf[saggital_index, :, :] 
    coronal_mid_csf = csf[:, coronal_index, :]
    axial_mid_csf = csf[:, :, axial_index]

    saggital_mid_gm = gm[saggital_index, :, :] 
    coronal_mid_gm = gm[:, coronal_index, :]
    axial_mid_gm = gm[:, :, axial_index]

    saggital_mid_wm = wm[saggital_index, :, :] 
    coronal_mid_wm = wm[:, coronal_index, :]
    axial_mid_wm = wm[:, :, axial_index]

    saggital_mid_csf = np.rot90(saggital_mid_csf, k=1)
    coronal_mid_csf = np.rot90(coronal_mid_csf, k=1)
    axial_mid_csf = np.rot90(axial_mid_csf, k=1)

    saggital_mid_gm = np.rot90(saggital_mid_gm, k=1)
    coronal_mid_gm = np.rot90(coronal_mid_gm, k=1)
    axial_mid_gm = np.rot90(axial_mid_gm, k=1)

    saggital_mid_wm = np.rot90(saggital_mid_wm, k=1)
    coronal_mid_wm = np.rot90(coronal_mid_wm, k=1)
    axial_mid_wm = np.rot90(axial_mid_wm, k=1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    # Saggital View
    axs[0].imshow(axial_mid, cmap='gray', aspect='auto')
    axs[0].imshow(saggital_mid_csf, cmap='jet', alpha=0.1, aspect='auto')
    axs[0].imshow(saggital_mid_gm, cmap='magma', alpha=0.1, aspect='auto')
    axs[0].imshow(saggital_mid_wm, cmap='viridis', alpha=0.1, aspect='auto')
    
    # Coronal View
    axs[1].imshow(coronal_mid, cmap='gray', aspect='auto')
    axs[1].imshow(coronal_mid_csf, cmap='jet', alpha=0.1, aspect='auto')
    axs[1].imshow(coronal_mid_gm, cmap='magma', alpha=0.1, aspect='auto')
    axs[1].imshow(coronal_mid_wm, cmap='viridis', alpha=0.1, aspect='auto')
    
    # Axial View
    axs[2].imshow(saggital_mid, cmap='gray', aspect='auto')
    axs[2].imshow(axial_mid_csf, cmap='jet', alpha=0.1, aspect='auto')
    axs[2].imshow(axial_mid_gm, cmap='magma', alpha=0.1, aspect='auto')
    axs[2].imshow(axial_mid_wm, cmap='viridis', alpha=0.1, aspect='auto')
    
    # Adjust layout and title
    fig.tight_layout(pad=2.5)
    fig.suptitle(f"{self.name}")
    
    plt.show()


def plot_files(folder_path):
    from utilities.load_files import load_files

    images = load_files(folder_path)

    for image in images:
        _plot(image)