"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.18.3
"""
import pandas as pd
import torch
from keras.utils import pad_sequences
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from typing import Dict


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self,labels,texts,tokenizer,max_length):
        self.labels = labels
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return(len(self.texts))
    
    def __getitem__(self,idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = self.tokenizer(text,padding = "max_length",max_length = self.max_length, truncation = True, return_tensors = "pt")
        
        input_ids = pad_sequences(tokenized_text['input_ids'],maxlen = self.max_length,dtype = torch.Tensor, truncating = "post",padding = "post")
        input_ids = input_ids.astype(dtype='int64')
        input_ids = torch.tensor(input_ids)
        
        
        attention_mask = pad_sequences(tokenized_text['attention_mask'],maxlen = self.max_length,dtype = torch.Tensor, truncating = "post",padding = "post")
        attention_mask = attention_mask.astype(dtype='int64')
        attention_mask = torch.tensor(attention_mask)
                                      
        return {'labels':torch.tensor(label,dtype = torch.long),
                'text':text,
                'input_ids':input_ids,
                'attention_mask':attention_mask.flatten()}

def label_mapping(df):
    id2label = {v:k for k,v in enumerate(df['category'].unique())}
    df['category'] = df['category'].map(id2label)
    return df

def create_dataset(train_data,test_data):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = Dataset(texts = train_data['text'].to_numpy(),
    labels = train_data['category'].to_numpy(),
    tokenizer = tokenizer,
    max_length= 512)

    test_dataset = Dataset(texts = test_data['text'].to_numpy(),
    labels = test_data['category'].to_numpy(),
    tokenizer = tokenizer,
    max_length= 512)

    return train_dataset,test_dataset

def split_df(df):
    train_data, test_data = train_test_split(df,test_size = 0.2,random_state = 42)
    return train_data,test_data

def create_data_loader(train_dataset,test_dataset,batch_size = 2, shuffle = True):
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = batch_size, shuffle = shuffle)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size = batch_size, shuffle = shuffle)
    return train_dataloader,test_dataloader

