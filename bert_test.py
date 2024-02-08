# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:05:55 2024

@author: QiJing
"""
import os
import torch
from bert_get_data import BertClassifier, GenerateData
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = './bert_checkpoint'
model = BertClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
model = model.to(device)
model.eval()

def evaluate(model, dataset):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=128)
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_loader:
            input_id = test_input['input_ids'].squeeze(1).to(device)
            mask = test_input['attention_mask'].to(device)
            test_label = test_label.to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc   
    print(f'Test Accuracy: {total_acc_test / len(dataset): .3f}')
    
test_dataset = GenerateData(mode="test")
evaluate(model, test_dataset)
