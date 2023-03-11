import os

from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchmetrics import JaccardIndex

import pandas as pd

import warnings


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from data.custom_dataset_data_loader import CustomTestDataLoader, sample_data
from utils.saving_utils import load_checkpoint_mgpu
from options.base_options import TestParser

from networks import U2NET

torch.cuda.empty_cache()
device = "cuda"

checkpoint_path = os.path.join("trained_checkpoint", "cloth_segm_u2net_latest.pth")
do_palette = True
opt = TestParser()

def get_palette(num_cls: int):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette


net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()

palette = get_palette(4)
jaccard = JaccardIndex(num_classes=4, task="multiclass")

# images_list = sorted(os.listdir(image_dir))
# pbar = tqdm(total=len(images_list))
custom_dataloader = CustomTestDataLoader()
custom_dataloader.initialize(opt)
loader = custom_dataloader.get_loader()

# pbar = range(opt.iter)
n_iter = range(custom_dataloader.dataset.dataset_size)
get_data = sample_data(loader)
metrics = list()
pbar = tqdm(total=custom_dataloader.dataset.dataset_size)
for itr in n_iter:
    data_batch = next(get_data)
    image_tensor, label_tensor, image_name = data_batch
    # print(f"Image tensor shape: {image_tensor.shape}, label tensor shape: {label_tensor.shape}, image name: {image_name}")
    output_image_name = image_name[0][:-3] + "png"
    
    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    # print(f"Output tensor shape: {output_tensor.shape}")
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = output_tensor.cpu() # shape [768, 768]
    # print(f"Output tensor after postprocessing shape: {output_tensor.shape}")
    label_tensor = torch.squeeze(label_tensor, dim=0) # shape [768, 768]
    # print(f"Postprocessed label tensor shape: {label_tensor.shape}")
    I_o_U = float(jaccard(label_tensor, output_tensor))
    print(f"IoU metric for {output_image_name}: {I_o_U}")
    entry = (output_image_name, I_o_U)
    metrics.append(entry)
    output_merged_tensor = torch.cat((label_tensor, output_tensor), -1)
    output_arr = output_merged_tensor.numpy()
    compressed_arr = output_arr.astype("uint8")
    

    output_img = Image.fromarray(output_arr.astype("uint8"), mode="L")
    if do_palette:
        output_img.putpalette(palette)
    output_img.save(os.path.join(opt.output_images_dir, output_image_name))
    pbar.update(1)

pbar.close()
metrics_df = pd.DataFrame(metrics, columns=["Img_name", "IoU"])
metrics_df.to_csv(opt.output_metrics_file, index=False)