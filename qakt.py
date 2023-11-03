import os
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np

device = torch.device(os.environ['DEVICE_STR'])
use_multi_gpu = int(os.environ['USE_MULTI_GPU']) != 0
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class QAKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, n_blocks,
                 kq_same, dropout, model_type, qm=None, 
                 final_fc_dim=512, n_heads=8, d_ff=2048,  l2=1e-5):
        super().__init__()
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.qm = qm
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid+1, 1)
            if(self.qm is None):
                self.p_embed = nn.Sequential(
                        nn.Embedding(self.n_pid+1, self.n_question+1), 
                        nn.Sigmoid()
                )
            else:
                self.p_embed = nn.Sequential(
                        nn.Embedding.from_pretrained(self.qm, freeze=True), 
                )
        self.q_embed = nn.Sequential(
            nn.Linear(self.n_question+1, embed_l),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                    d_model=d_model, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(2*d_model)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model + embed_l),
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.LayerNorm(final_fc_dim),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )
        if(self.qm is None):
            self.reset()
            assert(self.p_embed[0].weight.sum() == 0, "ERROR: 未传入qf参数，且Q-matrix未初始化为0")
        else:
            self.reset_without_qm()
            assert(self.p_embed[0].weight.sum() != 0, "ERROR: 已传入qf参数，但Q-matrix又被初始化为0，会导致传入的Q-matrix被覆盖")

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)
    def reset_without_qm(self):
        for name, para in self.named_parameters():
            if para.size(0) == self.n_pid+1 and self.n_pid > 0  \
                and 'p_embed.0.weight' not in name:
                torch.nn.init.constant_(para, 0.)

    def normalize(self, raw, eps=1e-32):
        tmp = raw - raw.min(dim=-1, keepdim=True).values
        res = tmp / (tmp.max(dim=-1, keepdim=True).values + eps)
        return res

    def binary_mt(self, raw, theta=0.8):
        sr = F.softmax(raw, dim=-1)
        threshold = sr.max(-1).values
        mask = sr > theta * threshold.unsqueeze(-1)
        return mask.float()
    def forward(self, p_data, pa_data, target):
        q_data = self.p_embed(p_data)
        q_embed_data = self.q_embed(q_data)  / q_data.sum(-1).unsqueeze(-1)
        pa_pos_data = torch.where(pa_data > self.n_pid, pa_data - self.n_pid, 0)
        qa_pos_data = self.p_embed(pa_pos_data)
        pa_neg_data = torch.where(pa_data <= self.n_pid, pa_data, 0)
        qa_neg_data = self.p_embed(pa_neg_data)
        qa_embed_data = torch.cat([
                self.q_embed(qa_pos_data) / qa_pos_data.sum(-1).unsqueeze(-1)
,
                self.q_embed(qa_neg_data) / qa_neg_data.sum(-1).unsqueeze(-1)

            ], dim=-1)
        if self.n_pid > 0:
            pid_embed_data = self.difficult_param(p_data)  
            q_embed_data = q_embed_data + pid_embed_data
            qa_embed_data = qa_embed_data + pid_embed_data
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.
        if(self.qm is None):
            sparse_loss = (0.5 - (q_data - 0.5).abs()).sum()
        else:
            sparse_loss = 0
        q_embed_data = self.layernorm1(q_embed_data)
        qa_embed_data = self.layernorm2(qa_embed_data)
        d_output = self.model(q_embed_data, qa_embed_data)  
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)
        labels = target.reshape(-1)
        m = nn.Sigmoid()
        preds = (output.reshape(-1))  
        mask = labels > -1
        masked_labels = labels[mask].float()
        masked_preds = preds[mask]
        loss = nn.BCEWithLogitsLoss(reduction='none')
        output = loss(masked_preds, masked_labels)
        return output.sum()+c_reg_loss+sparse_loss, m(preds), mask.sum()
class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, 
                 d_ff, n_heads, dropout, kq_same, model_type):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.model_type = model_type

        if model_type in {'qakt'}:
            self.blocks_for_y = nn.ModuleList([
                TransformerLayer(d_model=2*d_model, v_len=2*d_model,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_for_x = nn.ModuleList([
                TransformerLayer(d_model=d_model, v_len=d_model,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_for_h = nn.ModuleList([
                    TransformerLayer(d_model=d_model, v_len=2*d_model,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            ])

    def forward(self, q_embed_data, qa_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_data = qa_embed_data
        q_pos_embed = q_embed_data
        y = qa_pos_data
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed
        for block in self.blocks_for_y:  
            y = block(mask=1, query=y, key=y, values=y)
        for idx in range(self.n_blocks):
            x = self.blocks_for_x[idx](mask=1, query=x, key=x,
                      values=x, apply_pos=False)
            x = self.blocks_for_h[idx](mask=0, query=x, key=x, values=y, apply_pos=True)
        return x
class TransformerLayer(nn.Module):
    def __init__(self, d_model, v_len,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.v_len = v_len
        self.masked_attn_head = MultiHeadAttention(
            d_model, v_len, n_heads, dropout, kq_same=kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:  
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)
        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, v_len, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // n_heads
        self.v_len = v_len
        self.d_v = v_len // n_heads
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(self.v_len, self.v_len, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(v_len, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_v)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        #
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, gammas)
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.v_len)
        output = self.out_proj(concat)

        return output
def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)  
        scores_ = scores_ * mask.float().to(device)
        distcum_scores = torch.cumsum(scores_, dim=-1)  
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True)  
        position_effect = torch.abs(
            x1-x2)[None, None, :, :].type(torch.FloatTensor).to(device)  
        dist_scores = torch.clamp(
            (disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  
    total_effect = torch.clamp(torch.clamp(
        (dist_scores*gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  
class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  
