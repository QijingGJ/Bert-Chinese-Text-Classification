# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:37:15 2024

@author: QiJing
"""

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
from bert_get_data import BertClassifier, MyDataset, GenerateData

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))

# 训练超参数
epoch = 5
batch_size = 64
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 20240121
save_path = './bert_checkpoint'
setup_seed(random_seed)

# 定义模型
model = BertClassifier()
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 构建数据集
train_dataset = GenerateData(mode='train')
dev_dataset = GenerateData(mode='val')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader):
        input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
        masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
        labels = labels.to(device)
        output = model(input_ids, masks)

        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    
    with torch.no_grad():
        for inputs, labels in dev_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device) # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device) # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()
        
        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')
        
        # 保存最优的模型
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model('best.pt')
        
    model.train()

# 保存最后的模型
save_model('last.pt')

