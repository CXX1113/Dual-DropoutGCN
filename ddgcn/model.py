from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, init, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.use_bias = use_bias
        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == 'Xavier':
            fan_in, fan_out = self.weight.shape
            init_range = np.sqrt(6.0 / (fan_in + fan_out))
            self.weight.data.uniform_(-init_range, init_range)

            if self.use_bias:
                torch.nn.init.constant_(self.bias, 0.)

        elif init == 'Kaiming':
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

            if self.use_bias:
                fan_in, _ = self.weight.shape
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.use_bias:
                self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        if inputs.is_sparse:
            support = torch.sparse.mm(inputs, self.weight)
        else:
            support = torch.mm(inputs, self.weight)
        outputs = torch.sparse.mm(adj, support)
        if self.use_bias:
            return outputs + self.bias
        else:
            return outputs

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNEncoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2):
        """
        :param nfeat:
        :param nhid1: Node embedding dim in first GCN layer
        :param nhid2: Node embedding dim in second GCN layer
        :param dropout:
        :param init:
        :param use_bias:
        :param is_sparse_feat1:
        :param is_sparse_feat2:
        """
        super(GCNEncoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1, init, use_bias)
        self.gc2 = GraphConvolution(nhid1, nhid2, init, use_bias)
        self.dropout = dropout
        self.is_sparse_feat1 = is_sparse_feat1
        self.is_sparse_feat2 = is_sparse_feat2

    def forward(self, x1, x2, adj):
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.dropout(x2, self.dropout, training=self.training)
        if self.is_sparse_feat1:
            x1 = x1.to_sparse()
        if self.is_sparse_feat2:
            x2 = x2.to_sparse()
        x1 = F.relu(self.gc1(x1, adj))
        x2 = F.relu(self.gc1(x2, adj))
        if self.training:
            mask = torch.FloatTensor(torch.bernoulli(x1.data.new(x1.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout))
            x1 = x1 * mask
            x2 = x2 * mask
        x1 = self.gc2(x1, adj)
        x2 = self.gc2(x2, adj)
        return x1, x2


class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, inputs1, inputs2):
        if self.training:
            mask = torch.FloatTensor(
                torch.bernoulli(inputs1.data.new(inputs1.data.size()).fill_(1 - self.dropout)) / (1 - self.dropout))
            inputs1 = inputs1 * mask
            inputs2 = inputs2 * mask
        outputs1 = torch.mm(inputs1, inputs1.t())
        outputs2 = torch.mm(inputs2, inputs2.t())
        return outputs1, outputs2


class GraphAutoEncoder(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GCNEncoder(nfeat, nhid1, nhid2, dropout, init, use_bias, is_sparse_feat1, is_sparse_feat2)
        self.decoder = InnerProductDecoder(dropout)

    def forward(self, x1, x2, adj):
        node_embed1, node_embed2 = self.encoder(x1, x2, adj)
        reconstruct_adj_logit1, reconstruct_adj_logit2 = self.decoder(node_embed1, node_embed2)

        return reconstruct_adj_logit1, reconstruct_adj_logit2


