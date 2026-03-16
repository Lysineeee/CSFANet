import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from index2one_hot import get_one_hot
from dataload.GZCDDdataset import GZCDDataset
from models.CSFANet import MY_NET


def visualize_cd(gt, pred):
    """
    gt, pred: [H, W], values in {0,1}
    return: [H, W, 3] uint8 RGB image
    """
    h, w = gt.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # 正确未变化 TN -> 黑
    vis[(gt == 0) & (pred == 0)] = [0, 0, 0]

    # 正确变化 TP -> 白
    vis[(gt == 1) & (pred == 1)] = [255, 255, 255]

    # 漏报 FN -> 绿
    vis[(gt == 1) & (pred == 0)] = [0, 255, 0]

    # 误报 FP -> 红
    vis[(gt == 0) & (pred == 1)] = [255, 0, 0]

    return vis


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ================== 路径配置 ==================
    weights_path = "./checkpoints/GZ_CD/1st_embedding/Last_Epoch.pth"
    save_root = "./checkpoints/GZ_CD/1st_embedding/vis"
    os.makedirs(save_root, exist_ok=True)

    # ================== 数据 ==================
    test_dataset = GZCDDataset(mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ================== 模型 ==================
    net = MY_NET(2)
    net.load_state_dict(torch.load(weights_path, map_location=device))
    net.to(device)
    net.eval()

    idx = 0

    with torch.no_grad():
        for before, after, change in tqdm(test_loader, ncols=100):
            before = before.to(device)
            after = after.to(device)
            change = change.squeeze(1).long()

            pred, aux1, aux2, aux3 = net(before, after)
            label_pred = F.softmax(pred, dim=1).argmax(dim=1).cpu().numpy()
            label_true = change.cpu().numpy()

            for i in range(label_pred.shape[0]):
                vis_img = visualize_cd(label_true[i], label_pred[i])
                save_path = os.path.join(save_root, f"{idx:05d}.png")
                plt.imsave(save_path, vis_img)
                idx += 1


if __name__ == "__main__":
    main()
