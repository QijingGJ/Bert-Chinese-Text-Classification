# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:05:26 2024

@author: QiJing
"""
import os
from transformers import BertTokenizer
import torch
from bert_get_data import BertClassifier

bert_name = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

save_path = './bert_checkpoint'
model = BertClassifier()
model.load_state_dict(torch.load(os.path.join(save_path, 'best.pt')))
model = model.to(device)
model.eval()

real_labels = []
with open('./THUCNews/data/class.txt', 'r') as f:
    for row in f.readlines():
        real_labels.append(row.strip())

while True:
    text = input('请输入新闻标题：')
    bert_input = tokenizer(text, padding='max_length', 
                           max_length = 35, 
                           truncation=True,
                           return_tensors="pt")
    input_ids = bert_input['input_ids'].to(device)
    masks = bert_input['attention_mask'].unsqueeze(1).to(device)
    output = model(input_ids, masks)
    pred = output.argmax(dim=1)
    print(real_labels[pred])
