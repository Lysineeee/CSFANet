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

# from models.ISDANet import MY_NET
from models.LysineNet import MY_NET


from dataload.LEVIRdataset import LEVIRDataset
from dataload.SYSUCDdataset import SYSUCDDataset
from dataload.GZCDDdataset import GZCDDataset



import warnings
warnings.filterwarnings('ignore')


# test_data = SYSUCDDataset(mode='test')
test_data = GZCDDataset(mode='test')
# test_data = LEVIRDataset(mode='test')
test_data_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4)

lr = 0.0001
n_class = 2
F1_max = 0.85
device = "cuda"

root = '/home/data3/liuyansong/Lysinear/ISDANet/checkpoints/GZ_CD/1st_embedding/'
weights_path = "/home/data3/liuyansong/Lysinear/ISDANet/checkpoints/GZ_CD/1st_embedding/Last_Epoch_F1_0.8859_iou0.7952_epoch_200.pth"

net = MY_NET(2)
net.load_state_dict(torch.load(weights_path, map_location=device))
net.to(device)


criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=5)


with torch.no_grad():
    with open(root + '/test.txt', 'a') as f1:
        torch.cuda.empty_cache()

        _test_loss = 0

        _hist = np.zeros((n_class, n_class))

        k = 0

        net.eval()
        for before, after, change in tqdm(test_data_loader, ncols=100):
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

            _test_loss += loss.item()

            label_pred = F.softmax(pred, dim=1).max(dim=1)[1].data.cpu().numpy()
            label_true = change.data.cpu().numpy()

            hist = hist_sum(label_true, label_pred, 2)

            _hist += hist

        miou, oa, kappa, precision, recall, iou, F1 = compute_metrics(_hist)

        testloss = _test_loss / len(test_data_loader)

        print( '|test loss:', testloss, ' |test oa:', oa,  ' |test iou:', iou, ' |test F1:', F1)
        f1.write(f"weights path: {weights_path} \n")
        f1.write('|test loss:%0.04f|test miou:%0.04f|test oa:%0.04f|test kappa:%0.04f|test precision:%0.04f|test recall:%0.04f|test iou:%0.04f|test F1:%0.04f' % (
            testloss, miou, oa, kappa, precision, recall, iou, F1))
        f1.write('\n')
        f1.flush()
# scheduler.step(testloss)
