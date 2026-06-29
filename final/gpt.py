import math

import torch.nn.functional as F

import torch

def scaled_dot_product_attention(q, k, v, mask=None):

    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2,-1)/math.sqrt(d_k))

    attn_weights = F.softmax(scores, dim=-1)

    context = torch.matmul(attn_weights, v)

    return context, attn_weights

