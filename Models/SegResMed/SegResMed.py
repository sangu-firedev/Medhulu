import torch
import nibabel as nib
import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, CropForegroundd, ResizeWithPadOrCropd, ScaleIntensityd, ToTensord
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference(model_path, image_path, output_path):
    # Define the model architecture
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=4,
        dropout_prob=0.2,
    )
    model.to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    reg_size = (197, 233, 189)
    transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        CropForegroundd(keys=["image"], source_key="image"),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=reg_size),
        ScaleIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ])
    
    data = transforms({"image": image_path})
    image = data["image"].unsqueeze(0).to(device)
    
    # Save the preprocessed image
    raw_image = torch.softmax(image, dim=1)
    raw_image = torch.argmax(raw_image, dim=1)[0].cpu().numpy()
    
    with torch.no_grad():
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(image, roi_size, sw_batch_size, model)
        outputs = torch.softmax(outputs, dim=1)
        outputs = torch.argmax(outputs, dim=1)[0].cpu().numpy()
    
    original_img = nib.load(image_path)
    
    output_path_image = os.path.join(output_path, "image.nii.gz")
    output_path_seg = os.path.join(output_path, "seg.nii.gz")
    
    # Save images
    result_img = nib.Nifti1Image(raw_image.astype(np.uint8), original_img.affine, original_img.header)
    nib.save(result_img, output_path_image)
    
    result_img = nib.Nifti1Image(outputs.astype(np.uint8), original_img.affine, original_img.header)
    nib.save(result_img, output_path_seg)
    
    print(f"Segmentation saved to {output_path}")