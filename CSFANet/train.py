import os
# import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from assess import hist_sum, compute_metrics
# from pytorchtools import EarlyStopping

from index2one_hot import get_one_hot 
from poly import adjust_learning_rate_poly

from models.CSFANet import MY_NET


from dataload.LEVIRdataset import LEVIRDataset
from dataload.GZCDDdataset import GZCDDataset


import warnings
warnings.filterwarnings('ignore')


train_data = GZCDDataset(mode='train')
# train_data = LEVIRDataset(mode='train')
data_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

batch = next(iter(data_loader))
before, after, change = batch
before_img = before[0]   # [3, H, W]
after_img  = after[0]    # [3, H, W]
change_img = change[0]   # [1, H, W]
before_img = before_img.permute(1, 2, 0).cpu().numpy()
after_img  = after_img.permute(1, 2, 0).cpu().numpy()
change_img = change_img.squeeze(0).cpu().numpy()
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(before_img)
plt.title("T1")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(after_img)
plt.title("T2")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(change_img, cmap="gray")
plt.title("Change Label")
plt.axis("off")
plt.savefig("fig.png")
plt.show()


val_data = GZCDDataset(mode='val')
# val_data = LEVIRDataset(mode='val')
val_data_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

Epoch = 200
lr = 0.0001
n_class = 2
F1_max = 0.00
device = "cuda"

config = {
    "learning_rate": 0.0001,
    "epochs": 200,
    "batch_size": 8,
    "dataset": "LEVIRDataset",
    "model_name": "LysineNet",
    "optimizer": "Adam",
    "criterion": "BCEWithLogitsLoss"
}

# wandb.init(
#     project="LysineNet",
#     config=config
# )

root = r'./checkpoints/GZ_CD/1st'
os.makedirs(root, exist_ok=True)

net = MY_NET(2).to(device)

criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5)


with open(root +'/train.txt', 'a') as f:
    for epoch in range(Epoch):
        # print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        torch.cuda.empty_cache()

        new_lr = adjust_learning_rate_poly(optimizer, epoch, Epoch, lr, 0.9)
        print('lr:', new_lr)

        _train_loss = 0

        _hist = np.zeros((n_class, n_class))

        net.train()
        for before, after, change in tqdm(data_loader, desc='epoch{}'.format(epoch), ncols=100):
            before = before.to(device)
            after = after.to(device)

            # ed_change = change.cuda()
            # ed_change = edge(ed_change)
            # lbl = torch.where(ed_change > 0.1, 1, 0)
            # plt.figure()
            # plt.imshow(lbl.data.cpu().numpy()[0][0], cmap='gray')
            # plt.show()
            # lbl = lbl.squeeze(dim=1).long().cpu()
            # lbl_one_hot = get_one_hot(lbl, 2).permute(0, 3, 1, 2).contiguous().cuda()

            change = change.squeeze(dim=1).long()
            change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().to(device)

            optimizer.zero_grad()

            # print(before.shape)
            pred, aux1, aux2, aux3 = net(before, after)
            loss_pred = criterion(pred, change_one_hot)
            loss_aux_1 = criterion(aux1, change_one_hot)
            loss_aux_2 = criterion(aux2, change_one_hot)
            loss_aux_3 = criterion(aux3, change_one_hot)
            loss = loss_pred + loss_aux_1 + loss_aux_2 + loss_aux_3


            loss.backward()
            optimizer.step()
            _train_loss += loss.item()

            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
            label_true = change.data.cpu().numpy()

            hist = hist_sum(label_true, label_pred, 2)

            _hist += hist

        # scheduler.step()

        miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

        trainloss = _train_loss / len(data_loader)

        print('Epoch:', epoch, ' |train loss:', trainloss, ' |train oa:', oa,  ' |train iou:', iou, ' |train F1:', F1)
        f.write('Epoch:%d|train loss:%0.04f|train miou:%0.04f|train oa:%0.04f|train kappa:%0.04f|train precision:%0.04f|train recall:%0.04f|train iou:%0.04f|train F1:%0.04f' % (
                epoch, trainloss, miou, oa, kappa, precision, recall, iou, F1))
        f.write('\n')
        f.flush()

        with torch.no_grad():
            with open(root + '/val.txt', 'a') as f1:

                torch.cuda.empty_cache()

                _val_loss = 0

                _hist = np.zeros((n_class, n_class))

                k = 0

                net.eval()
                for before, after, change in tqdm(val_data_loader, desc='epoch{}'.format(epoch), ncols=100):
                    before = before.cuda()
                    after = after.cuda()
                    change = change.squeeze(dim=1).long()
                    change_one_hot = get_one_hot(change, 2).permute(0, 3, 1, 2).contiguous().to(device)

                    pred, aux1, aux2, aux3 = net(before, after)

                    loss = criterion(pred, change_one_hot)

                    # loss_aux_1 = criterion(aux1, change_one_hot)
                    # loss_aux_2 = criterion(aux2, change_one_hot)
                    # loss_aux_3 = criterion(aux3, change_one_hot)
                    # loss = loss_pred + 0.4 * loss_aux_1 + 0.4 * loss_aux_2 + 0.4 * loss_aux_3

                    _val_loss += loss.item()

                    label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
                    label_true = change.data.cpu().numpy()

                    hist = hist_sum(label_true, label_pred, 2)

                    _hist += hist

                miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

                valloss = _val_loss / len(val_data_loader)

                # wandb.log({
                #         "Val/Loss": valloss,
                #         "Val/mIoU": miou,
                #         "Val/F1_Score": F1,
                #         "Val/IoU": iou,
                #         "Val/Overall_Accuracy": oa,
                #     })

                print('Epoch:', epoch, ' |val loss:', valloss, ' |val oa:', oa,  ' |val iou:', iou, ' |val F1:', F1)
                f1.write('Epoch:%d|val loss:%0.04f|val miou:%0.04f|val oa:%0.04f|val kappa:%0.04f|val precision:%0.04f|val recall:%0.04f|val iou:%0.04f|val F1:%0.04f' % (
                    epoch, valloss, miou, oa, kappa, precision, recall, iou, F1))
                f1.write('\n')
                f1.flush()
        # scheduler.step(valloss)


                if F1 > F1_max:
                    # save_path = args.summary_path+args.dir_name+'/checkpoints/'+'miou_{:.6f}.pth'.format(miou)
                    # torch.save(model.state_dict(), save_path)

                    save_path = root + '/F1_{:.4f}_iou_{:.4f}_epoch_{}.pth'.format(F1, iou, epoch)
                    torch.save(net.state_dict(), save_path)

                    # torch.save(net, root + 'F1_{:.4f}_epoch_{}.pth'.format(F1, epoch))
                    F1_max = F1

# wandb.finish()

save_path = root + '/Last_Epoch_F1_{:.4f}_iou{:.4f}_epoch_{}.pth'.format(F1, iou, Epoch)
torch.save(net.state_dict(), save_path)
