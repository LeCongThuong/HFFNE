import matplotlib.pyplot as plt 
import numpy as np
import os
import cv2



gt_dir = "/mnt/hmi/thuong/Photoface_dist/PhotofaceDBLib/"
pred_dir = "/mnt/hmi/thuong/HFFNE/sample_results/"
pred_path = "1034/2009-08-03_12-39-50/im0_predict.npy"
gt_path = pred_path.replace("_predict.npy", "_sn.npy")
img = gt_path.replace("_sn.npy", "_crop.jpg")
mask = gt_path.replace("_sn.npy", "_mask.png")

np_gt = np.load(str(os.path.join(gt_dir, gt_path)))
np_img = cv2.imread(str(os.path.join(gt_dir, img)))
np_pred = np.load(str(os.path.join(pred_dir, pred_path)))
np_mask = cv2.imread(str(os.path.join(gt_dir, mask)))
bool_mask = np_mask > 0


fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,9))
axes[0].imshow(np_img)
axes[1].imshow(((np_gt+1)*127.5).astype(np.uint8))
axes[2].imshow(((np_pred+1)*127.5).astype(np.uint8))
axes[3].imshow(np_mask)

plt.show()