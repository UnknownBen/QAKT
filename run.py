
import os
import numpy as np
import torch
from sklearn import metrics
from utils import model_isPid_type
from torch.utils.data import DataLoader

device = torch.device(os.environ['DEVICE_STR'])
use_multi_gpu = int(os.environ['USE_MULTI_GPU']) != 0
transpose_data_model = {'qakt'}

def display(device_str, seq_num, count, true_el, q, label):
    content = ['===== 设备: ' + str(device_str) + ' =====',
                '训练总段数: ' + str(seq_num),
                "实际用段数: " + str(count), 
                "实际元素数: " + str(true_el),
                "最后一次所用q.shpe: " + str(q.shape),
                '===== 数据来自' + str(label) + '=====']
    content = '\n'.join(content)
    print(content)

def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False
def compute_auc(all_target, all_pred):
    return metrics.roc_auc_score(all_target, all_pred)
def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)
def train(net, params,  optimizer,  train_dataloader, label):
    net.train()
    pid_flag, model_type = model_isPid_type(params.model)
    count = 0
    true_el = 0

    seq_num = len(train_dataloader)
    for idx, (q, qa, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if model_type not in transpose_data_model:
            q = q.T
            qa = qa.T
            target = target.T
        q = q.long().to(device)
        qa = qa.long().to(device)
        target = target.float().to(device)
        loss, pred, true_ct = net(q, qa, target)
        target = target.cpu()
        loss.backward()
        true_el += true_ct.cpu().numpy()
        count += target.shape[0]
        if params.maxgradnorm > 0.:
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), max_norm=params.maxgradnorm)
        optimizer.step()
    display(device, seq_num, count, true_el, q, label)
    return True
def evalm(net, params, optimizer, data, label):
    pid_flag, model_type = model_isPid_type(params.model)
    net.eval()
    seq_num = len(data)
    pred_list = []
    target_list = []
    count = 0
    true_el = 0
    dataloader = DataLoader(data, batch_size=params.batch_size, shuffle=True)
    for idx, (q, qa, target) in enumerate(dataloader):
        if model_type not in transpose_data_model:
            q = q.T
            qa = qa.T
            target = target.T
        q = q.long().to(device)
        qa = qa.long().to(device)
        target = target.float().to(device)
        with torch.no_grad():
            loss, pred, ct = net(q, qa, target)
        pred = pred.cpu().numpy()  
        target = target.cpu()
        true_el += ct.cpu().numpy()
        count += target.shape[0]
        target = target.reshape((-1,))
        nopadding_index = np.flatnonzero(target > -1)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)
    assert count == seq_num, "Seq not matching"
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)
    display(device, seq_num, count, true_el, q, label)

    return loss, accuracy, auc
