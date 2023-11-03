import sys
import os
import json
import os.path
import glob
import logging
import argparse
import numpy as np
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def read_config():
    local_rank = os.environ.get("LOCAL_RANK")
    device_str = "cpu"
    use_multi_gpu = False
    if torch.cuda.is_available():
        if local_rank is None:
            device_str = "cuda"
        else:
            device_str = "cuda:" + str(local_rank)
            local_rank = int(local_rank)
            use_multi_gpu = True
    return use_multi_gpu, local_rank, device_str

use_multi_gpu, local_rank, device_str = read_config()
os.environ["DEVICE_STR"] = device_str

os.environ["USE_MULTI_GPU"] = '1' if use_multi_gpu else "0"

from load_data import PID_DATA
from run import train, evalm
from utils import try_makedirs, load_model, get_file_name_identifier

device = torch.device(device_str)

if use_multi_gpu:
    dist.init_process_group(backend='nccl')
print("本次执行可见", torch.cuda.device_count(), "块GPU")
print("使用: ", str(device))
def binarize_qm(raw, thr):
    m = raw.max(-1).values
    mask = raw >= thr * m.unsqueeze(-1)
    res = mask.cpu().float().detach().numpy()
    return res
def train_one_dataset(fold_id, params, file_name, train_data, valid_data):
    model = load_model(params)
    global use_multi_gpu
    if use_multi_gpu:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.lr, betas=(0.9, 0.999), eps=1e-8)

    print("\n")
    all_train_loss = {}
    all_train_accuracy = {}
    all_train_auc = {}
    all_valid_loss = {}
    all_valid_accuracy = {}
    all_valid_auc = {}
    best_valid_auc = 0
    qf_name = None
    ts = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    writer = SummaryWriter("logs/logs_" + "fold" + str(fold_id) + "_qf" + str(int(params.qf is not None)) + "_" + ts)
    dis_sampler = DistributedSampler(train_data) if use_multi_gpu else None
    train_dataloader = DataLoader(train_data, batch_size=params.batch_size, 
                    shuffle=(dis_sampler is None), sampler=dis_sampler)
    for idx in range(params.max_iter):
        if use_multi_gpu:
            dis_sampler.set_epoch(idx)
        train(model, params, optimizer, train_dataloader, label='Training')

        if use_multi_gpu and local_rank != 0:
            continue
        train_loss, train_accuracy, train_auc = evalm(
            model,  params, optimizer, train_data, label='Train_eval')
        valid_loss, valid_accuracy, valid_auc = evalm(
            model,  params, optimizer, valid_data, label='Valid_eval')

        print('epoch', idx + 1)
        print("valid_auc\t", valid_auc, "\ttrain_auc\t", train_auc)
        print("valid_accuracy\t", valid_accuracy,
              "\ttrain_accuracy\t", train_accuracy)
        print("valid_loss\t", valid_loss, "\ttrain_loss\t", train_loss)

        writer.add_scalars("loss", {"train": train_loss, "valid": valid_loss}, idx)
        writer.add_scalars("accuracy", {"train": train_accuracy, "valid": valid_accuracy}, idx)
        writer.add_scalars("auc", {"train": train_auc, "valid": valid_auc}, idx)

        try_makedirs('model')
        try_makedirs(os.path.join('model', params.model))
        try_makedirs(os.path.join('model', params.model, params.save))

        all_valid_auc[idx + 1] = valid_auc
        all_train_auc[idx + 1] = train_auc
        all_valid_loss[idx + 1] = valid_loss
        all_train_loss[idx + 1] = train_loss
        all_valid_accuracy[idx + 1] = valid_accuracy
        all_train_accuracy[idx + 1] = train_accuracy
        if valid_auc > best_valid_auc:
            path = os.path.join('model', params.model,
                                params.save,  file_name) + '_*'
            for i in glob.glob(path):
                os.remove(i)
            best_valid_auc = valid_auc
            best_epoch = idx + 1
            if use_multi_gpu:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            if(params.qf is None):
                qm_binarized = binarize_qm(F.sigmoid(model_state_dict['p_embed.0.weight']), params.qm_bin_thr)
                qf_name = "qm_" + "fold" + str(fold_id) + '_trained_' + str(params.qm_bin_thr) + ".json"
                with open(qf_name, 'w', encoding='utf-8') as foo:
                    json.dump(qm_binarized.tolist(), foo)
            torch.save({'epoch': idx,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        },
                       os.path.join('model', params.model, params.save,
                                    file_name) + '_' + str(idx + 1)
                       )
        if idx-best_epoch > 40:
            break   
    if use_multi_gpu and local_rank != 0:
        return -1
    try_makedirs('result')
    try_makedirs(os.path.join('result', params.model))
    try_makedirs(os.path.join('result', params.model, params.save))
    f_save_log = open(os.path.join(
        'result', params.model, params.save, file_name), 'w')
    f_save_log.write("valid_auc:\n" + str(all_valid_auc) + "\n\n")
    f_save_log.write("train_auc:\n" + str(all_train_auc) + "\n\n")
    f_save_log.write("valid_loss:\n" + str(all_valid_loss) + "\n\n")
    f_save_log.write("train_loss:\n" + str(all_train_loss) + "\n\n")
    f_save_log.write("valid_accuracy:\n" + str(all_valid_accuracy) + "\n\n")
    f_save_log.write("train_accuracy:\n" + str(all_train_accuracy) + "\n\n")
    f_save_log.close()

    return best_epoch, qf_name
def test_one_dataset(params, file_name, test_data, best_epoch):

    print("\n\nStart testing ......................\n Best epoch:", best_epoch)
    model = load_model(params)
    checkpoint = torch.load(os.path.join(
        'model', params.model, params.save, file_name) + '_' + str(best_epoch))
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_accuracy, test_auc = evalm(
        model, params, None, test_data, label='Test')
    print("\ntest_auc\t", test_auc)
    print("test_accuracy\t", test_accuracy)
    print("test_loss\t", test_loss)
    return test_loss, test_accuracy, test_auc
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test KT')
    parser.add_argument('--max_iter', type=int, default=300,
                        help='number of iterations')
    parser.add_argument('--train_set', type=int, default=1)
    parser.add_argument('--qm_bin_thr', type=float, default=0.99, help='Q-matrix binarizing threshold')
    parser.add_argument('--seed', type=int, default=224, help='default seed')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Default Optimizer')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='the batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--maxgradnorm', type=float,
                        default=-1, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=int, default=512,
                        help='hidden state dim for final fc layer')
    parser.add_argument('--d_model', type=int, default=256,
                        help='Transformer d_model shape')
    parser.add_argument('--d_ff', type=int, default=1024,
                        help='Transformer d_ff shape')
    parser.add_argument('--dropout', type=float,
                        default=0.05, help='Dropout rate')
    parser.add_argument('--n_block', type=int, default=1,
                        help='number of blocks')
    parser.add_argument('--n_head', type=int, default=8,
                        help='number of heads in multihead attention')
    parser.add_argument('--kq_same', type=int, default=1)
    parser.add_argument('--l2', type=float,
                        default=1e-5, help='l2 penalty for difficulty')
    parser.add_argument('--q_embed_dim', type=int, default=50,
                        help='question embedding dimensions')
    parser.add_argument('--qa_embed_dim', type=int, default=256,
                        help='answer and question embedding dimensions')
    parser.add_argument('--memory_size', type=int,
                        default=50, help='memory size')
    parser.add_argument('--init_std', type=float, default=0.1,
                        help='weight initialization std')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lamda_r', type=float, default=0.1)
    parser.add_argument('--lamda_w1', type=float, default=0.1)
    parser.add_argument('--lamda_w2', type=float, default=0.1)
    parser.add_argument('--model', type=str, default='qakt_pid',
                        help="combination of qakt/sakt/dkvmn/dkt (mandatory), pid/cid (mandatory) separated by underscore '_'. For example tf_pid")
    parser.add_argument('--dataset', type=str, default="assist2009_pid")
    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009_pid"}:
        params.n_question = 122
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_pid = 17751
    if dataset in {"assist2012_pid"}:
        params.n_question = 198
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_pid = 50988
    if dataset in {"assist2017_pid"}:
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_question = 101
        params.n_pid = 3162
    if dataset in {"assist2015"}:
        params.n_question = 100
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    if dataset in {"statics_pid"}:
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/'+dataset
        params.data_name = dataset
        params.n_question = 84
        params.n_pid = 633
    if dataset in {"statics"}:
        params.n_question = 1223
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
    params.save = params.data_name
    params.load = params.data_name
    dat = PID_DATA(n_pid=params.n_pid, seqlen=params.seqlen, separate_char=',')
    file_name_identifier = get_file_name_identifier(params)
    d = vars(params)
    for key in d:
        print('\t', key, '\t', d[key])
    file_name = ''
    for item_ in file_name_identifier:
        file_name = file_name + item_[0] + str(item_[1])
    if(params.train_set == -1):
        fold_pool = range(1, 6)
    else:
        fold_pool = [params.train_set]

    test_aucs = []
    for fold_id in fold_pool:
        print("=====第", fold_id, "次实验=====")

        seedNum = params.seed
        np.random.seed(seedNum)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seedNum)
        np.random.seed(seedNum)

        params.qf = None
        base_file_name = "fold" + str(fold_id) + "_qf" + str(int(params.qf is not None)) + file_name
        train_data_path = params.data_dir + "/" + \
            params.data_name + "_train" + str(fold_id) + ".csv"
        valid_data_path = params.data_dir + "/" + \
            params.data_name + "_valid" + str(fold_id) + ".csv"
        train_data = dat.load_data(train_data_path)
        valid_data = dat.load_data(valid_data_path)

        print("\n")
        print("train_data length: ", len(train_data))
        print("valid_data length: ", len(valid_data))  
        print("\n")
        best_epoch, qf_name = train_one_dataset(fold_id, 
            params, base_file_name, train_data, valid_data)
        if best_epoch != -1:
            test_data_path = params.data_dir + "/" + \
                params.data_name + "_test" + str(fold_id) + ".csv"
            test_data = dat.load_data(test_data_path)

            test_one_dataset(params, base_file_name, test_data, best_epoch)
        params.qf = qf_name
        base_file_name = "fold" + str(fold_id) + "_qf" + str(int(params.qf is not None)) + file_name
        best_epoch, _ = train_one_dataset(fold_id, 
            params, base_file_name, train_data, valid_data)
        if best_epoch != -1:
            test_loss, test_accuracy, test_auc = test_one_dataset(params, base_file_name, test_data, best_epoch)
            test_aucs.append(test_auc)
    mean_auc = sum(test_aucs) / len(test_aucs)
    print("本次执行平均值auc： ", mean_auc)
