from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import random
import torchvision.transforms.functional as tf
import re
from pathlib import Path

root_path = './Datasets/CD_Data_GZ_256_split'
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', str(s))]

def read_images(root):
    root = Path(root)
    
    def get_sorted_files(pattern):
        return sorted(root.glob(pattern), key=natural_sort_key)
    
    train_T1 = get_sorted_files('train/T1/*.tif')
    train_T2 = get_sorted_files('train/T2/*.tif')
    train_label = get_sorted_files('train/labels_change/*.png')
    
    val_T1 = get_sorted_files('val/T1/*.tif')
    val_T2 = get_sorted_files('val/T2/*.tif')
    val_label = get_sorted_files('val/labels_change/*.png')
    
    test_T1 = get_sorted_files('test/T1/*.tif')
    test_T2 = get_sorted_files('test/T2/*.tif')
    test_label = get_sorted_files('test/labels_change/*.png')
    
    assert len(train_T1) == len(train_T2) == len(train_label), \
        f"训练集文件数量不匹配: T1={len(train_T1)}, T2={len(train_T2)}, Label={len(train_label)}"
    assert len(val_T1) == len(val_T2) == len(val_label), \
        f"验证集文件数量不匹配: T1={len(val_T1)}, T2={len(val_T2)}, Label={len(val_label)}"
    assert len(test_T1) == len(test_T2) == len(test_label), \
        f"测试集文件数量不匹配: T1={len(test_T1)}, T2={len(test_T2)}, Label={len(test_label)}"
    
    return (
        [str(p) for p in train_T1], [str(p) for p in train_T2], [str(p) for p in train_label],
        [str(p) for p in val_T1],   [str(p) for p in val_T2],   [str(p) for p in val_label],
        [str(p) for p in test_T1],  [str(p) for p in test_T2],  [str(p) for p in test_label]
    )

def train_transforms(before, after, change):
    if random.random() > 0.7:
        before = tf.hflip(before)
        after = tf.hflip(after)
        change = tf.hflip(change)
    
    if random.random() < 0.3:
        before = tf.vflip(before)
        after = tf.vflip(after)
        change = tf.vflip(change)
    
    angle = transforms.RandomRotation.get_params([-180, 180])
    before = tf.rotate(before, angle)
    after = tf.rotate(after, angle)
    change = tf.rotate(change, angle)
    
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    
    return before, after, change

def val_test_transforms(before, after, change):
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

class GZCDDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val' or 'test'"
        self.mode = mode
        
        (self.train_before, self.train_after, self.train_change,
         self.val_before,   self.val_after,   self.val_change,
         self.test_before,  self.test_after,  self.test_change) = read_images(root_path)
        
        if mode == 'train':
            print(f'训练集加载了 {len(self.train_before)} 张图片')
        elif mode == 'val':
            print(f'验证集加载了 {len(self.val_before)} 张图片')
        elif mode == 'test':
            print(f'测试集加载了 {len(self.test_before)} 张图片')
    
    def __getitem__(self, item):
        if self.mode == 'train':
            before_path = self.train_before[item]
            after_path = self.train_after[item]
            change_path = self.train_change[item]
        elif self.mode == 'val':
            before_path = self.val_before[item]
            after_path = self.val_after[item]
            change_path = self.val_change[item]
        else:  # test
            before_path = self.test_before[item]
            after_path = self.test_after[item]
            change_path = self.test_change[item]
        
        before = Image.open(before_path).convert('RGB')
        after = Image.open(after_path).convert('RGB')
        change = Image.open(change_path).convert('L')
        
        if self.mode == 'train':
            before, after, change = train_transforms(before, after, change)
        else:
            before, after, change = val_test_transforms(before, after, change)
        
        return before, after, change
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_before)
        elif self.mode == 'val':
            return len(self.val_before)
        else:
            return len(self.test_before)

if __name__ == '__main__':
    dataset = GZCDDataset(mode='train')
    
    print("\n开始可视化检查（指定索引），仅显示图像内容...")
    
    i = 25
    
    before, after, change = dataset[i]
    
    before_np = before.permute(1, 2, 0).cpu().numpy()
    after_np = after.permute(1, 2, 0).cpu().numpy()
    change_np = change.squeeze(0).cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    axes[0].imshow(before_np)
    axes[0].set_title("Before (T1)", fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(after_np)
    axes[1].set_title("After (T2)", fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(change_np, cmap='gray')
    axes[2].set_title("Change Label", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('temp.png')
    plt.show()
