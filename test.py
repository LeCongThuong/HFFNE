
import argparse
import os, glob
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms
from models import *
from torchvision.utils import save_image
from pathlib import Path
import pandas as pd
import numpy as np 

if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="./ckpts/RefinerModel.pt")
    parser.add_argument("--ckpt_s1", type=str, default="./ckpts/ExemplarModel.pkl")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--exp_name", type=str, default="1_rec_cm")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--lossflag", type=int, default=0)
    parser.add_argument("--refNorm", type=float, default=1)
    parser.add_argument("--refVGG", type=int, default=1)
    parser.add_argument("--refDN", type=float, default=1)
    parser.add_argument("--gpuID", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--dest_dir", type=str, default="")
    parser.add_argument("--test_csv_file", type=str, default="")


    args = parser.parse_args()
    device = "cuda:" + str(args.gpuID)

    NormalEncoder = NormalEncoder(norm_dim=512).to(device)
    NormalDecoder = NormalDecoder(bilinear=False).to(device)
    PreTrainModel = UNet(channel=1).to(device)

    ckptModel = torch.load(args.ckpt_s1, map_location=lambda storage, loc: storage)
    PreTrainModel.load_state_dict(ckptModel)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        NormalEncoder.load_state_dict(ckpt["NormalEncoder"])
        NormalDecoder.load_state_dict(ckpt["NormalDecoder"])


    PreTrainModel.eval()
    NormalEncoder.eval()
    NormalDecoder.eval()
  
    Image2Tensor = transforms.Compose([
    transforms.ToTensor(),
    ])
    sampleList = pd.read_csv(args.test_csv_file, header=None)
    num_samples = len(sampleList)
    with torch.no_grad():
        for i in range(num_samples):
            img_path = os.path.join(args.data_dir, sampleList.iloc[i, 0])
            PilImg = Image.open(str(img_path)).convert('L')
            TensorImg = Image2Tensor(PilImg).unsqueeze(0).to(device)

            coNorm = F.normalize(PreTrainModel(TensorImg))      
            normal_feat = NormalEncoder(coNorm)
            fine_normal = F.normalize(NormalDecoder((TensorImg, normal_feat)))
            
            # fine_normal = (fine_normal * 128 + 128).clamp(0, 255) / 255
            model_output_perm = fine_normal.permute(0, 2, 3, 1)
            model_output_np = model_output_perm.cpu().numpy()[0]
            dest_path = os.path.join(args.dest_dir, str(sampleList.iloc[i, 0]).replace("crop.jpg", "predict.npy"))
            Path(dest_path).parent.mkdir(exist_ok=True, parents=True)
            np.save(dest_path, model_output_np)