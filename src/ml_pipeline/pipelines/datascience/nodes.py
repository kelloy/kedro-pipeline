"""
This is a boilerplate pipeline 'datascience'
generated using Kedro 0.18.3
"""
from torch import nn
from transformers import BertModel
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from collections import defaultdict
from typing import Dict
from transformers import get_scheduler
import plotly_express as px

class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768,5)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids = input_ids, attention_mask = attention_mask, return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        
        return final_layer

def train_epoch(model, train_dataloader,train_dataset,device,parameters:Dict):
    model = model.train()
    metric = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = float(parameters['LR']))
    lr_scheduler = get_scheduler(name = "linear", optimizer = optimizer,num_warmup_steps = 0, num_training_steps = parameters['EPOCHS']*len(train_dataset))
    
    losses = []
    acc_scores = 0
    progress_bar = tqdm(range(len(train_dataset)))
    
    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].squeeze(1).to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask)
        loss = metric(outputs, labels.long())
        predictions = torch.argmax(outputs,dim = -1)
        labels = labels.cpu().detach().numpy()
        predictions = predictions.cpu().detach().numpy()
        acc = (predictions == labels).sum().item()
        acc_scores += acc
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    return acc_scores/len(train_dataset), np.mean(losses)
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def eval_model(model, test_dataloader,test_dataset,device):
    model.eval()
    metric = nn.CrossEntropyLoss()

    validation_losses = []
    validation_acc_scores = 0
    
    for batch in test_dataloader:
                    
        test_label = batch['labels'].to(device)
        test_attention_mask = batch["attention_mask"].to(device)
        test_input_ids = batch["input_ids"].squeeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(test_input_ids,test_attention_mask)
            loss = metric(outputs, test_label.long())
            predictions = torch.argmax(outputs,dim = -1)
            test_label = test_label.cpu().detach().numpy()
            predictions = predictions.cpu().detach().numpy()
            
            acc = (predictions == test_label).sum().item()
            
            validation_acc_scores += acc
            validation_losses.append(loss.item())
            
    return validation_acc_scores/len(test_dataset), np.mean(validation_losses)

def model_fine_tune(train_dataset,train_dataloader,test_dataset,test_dataloader, parameters: Dict):
    EPOCHS = parameters['EPOCHS']
    best_acc_score = 0
    history = defaultdict(list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertClassifier().to(device)
    early_stopper = EarlyStopper(patience= parameters['patience'], min_delta=parameters['min_delta'])

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-'*10)
        train_acc_score, train_loss = train_epoch(model,train_dataloader, train_dataset,device,parameters)
        print(f"Train loss: {train_loss},Train Acc Score: {train_acc_score}")

        val_acc_score, val_loss = eval_model(
        model,test_dataloader,test_dataset,device)
        print(f"Test loss: {val_loss},Test Acc Score: {val_acc_score}")

        history['train_acc_score'].append(train_acc_score)
        history['train_loss'].append(train_loss)
        history['val_acc_score'].append(val_acc_score)
        history['val_loss'].append(val_loss)
        if val_acc_score > best_acc_score:
            torch.save(model.state_dict(),'./bert_best_model')
            best_acc_score = val_acc_score
            
        if early_stopper.early_stop(val_loss):
            break

        best_state_dict = torch.load("./bert_best_model")

    return best_state_dict, history

def plot_metrics(history):
    fig = px.line(history, title = "Metrics over Epochs")
    return fig

