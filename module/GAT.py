
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from module.GATStackLayer import MultiHeadSGATLayer, MultiHeadLayer
from module.GATLayer import PositionwiseFeedForward, WSGATLayer, SWGATLayer , WPGATLayer , PSGATLayer

######################################### SubModule #########################################
class WSWGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WSGATLayer)
        elif layerType == "S2W":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=SWGATLayer)
        elif layerType == "S2S":
            self.layer = MultiHeadSGATLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out)
        else:
            raise NotImplementedError("GAT Layer has not been implemented!")

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, s):
        if self.layerType == "W2S":
            origin, neighbor = s, w
        elif self.layerType == "S2W":
            origin, neighbor = w, s
        elif self.layerType == "S2S":
            assert torch.equal(w, s)
            origin, neighbor = w, s
        else:
            origin, neighbor = None, None

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

class WPWGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "W2P":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=WPGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, w, p):
        if self.layerType == "W2P":
            origin, neighbor = p, w  #from word to passage 

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h

class SPSGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, ffn_inner_hidden_size, ffn_drop_out, feat_embed_size, layerType):
        super().__init__()
        self.layerType = layerType
        if layerType == "P2S":
            self.layer = MultiHeadLayer(in_dim, int(out_dim / num_heads), num_heads, attn_drop_out, feat_embed_size, layer=PSGATLayer)

        self.ffn = PositionwiseFeedForward(out_dim, ffn_inner_hidden_size, ffn_drop_out)

    def forward(self, g, p, s):
        if self.layerType == "P2S":
            origin, neighbor = s, p  #from passage to sentence

        h = F.elu(self.layer(g, neighbor))
        h = h + origin
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h