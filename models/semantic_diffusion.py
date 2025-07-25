from cv2 import norm
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
# import clip

import math
import time
import os
import pickle
from torch.nn.utils import weight_norm
import wandb
import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import copy
from .motion_encoder import * 


    


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=6000):
        super(PeriodicPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.period = period
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1 # 1 if max_seq_len % period == 0 else 0
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)

        
    # def create_positional_encoding(self):

    #     pe = torch.zeros(self.period, self.d_model)
    #     position = torch.arange(0, self.period, dtype=torch.float).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     pe = pe.unsqueeze(0) # (1, period, d_model)
    #     repeat_num = (self.max_seq_len // self.period) + 1
    #     pe = pe.repeat(1, repeat_num, 1)
    #     return pe
    

    def forward(self, x, dropout=False):
        # # 动态调整 self.pe 的大小以匹配输入 x 的长度
        # if self.pe.size(1) < x.size(1):
        #     self.max_seq_len = x.size(1)
        #     self.pe = self.create_positional_encoding().cuda()

        x = x + self.pe[:, :x.size(1), :]
        
        if dropout:
            x = self.dropout(x)
        return x


# class PeriodicPositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
#         super(PeriodicPositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.dropout = nn.Dropout(p=dropout)
#         self.period = period
#         self.max_seq_len = max_seq_len
#         self.pe = self.create_positional_encoding()

#     def create_positional_encoding(self):
#         pe = torch.zeros(self.period, self.d_model)
#         position = torch.arange(0, self.period, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0) # (1, period, d_model)
#         repeat_num = (self.max_seq_len // self.period) + 1
#         pe = pe.repeat(1, repeat_num, 1)
#         return pe

#     def forward(self, x, dropout=False):
#         # 动态调整 self.pe 的大小以匹配输入 x 的长度
#         if self.pe.size(1) < x.size(1):
#             self.max_seq_len = x.size(1)
#             self.pe = self.create_positional_encoding()

#         x = x + self.pe[:, :x.size(1), :]
        
#         if dropout:
#             x = self.dropout(x)
#         return x





def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module




class SelfAttention(nn.Module):

    def __init__(self, latent_dim, num_head, dropout):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
    
    def forward(self, x):
        B, T, D = x.shape
        H = self.num_head

        # Normalize and linear transform for queries, keys, values
        query = self.query(self.norm(x)).view(B, T, H, D // H)
        key = self.key(self.norm(x)).view(B, T, H, D // H)
        value = self.value(self.norm(x)).view(B, T, H, D // H)

        # Compute attention scores
        scores = torch.einsum('bthd,bshd->bths', query, key) / (D // H) ** 0.5
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attention_output = torch.einsum('bths,bshd->bthd', attention_weights, value)
        attention_output = attention_output.reshape(B, T, D)

        # Add and project
        y = x + self.proj_out(attention_output)
        return y



class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super(TransformerTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x + self.pos_embedding[:, :seq_len, :]  # Add positional encoding
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_dim)
        x = self.norm(x)
        return x



class CrossAttention(nn.Module):

    def __init__(self, latent_dim, aud_latent_dim, num_head, dropout):
        super().__init__()
        self.audio_input_dim = 768
        self.text_feature_dim = 256
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(latent_dim)
        self.semantic_norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.audio_proj = nn.Linear(self.audio_input_dim, self.text_feature_dim)  # Project xf to text_feature_dim
        self.audio_text_proj = nn.Linear(self.text_feature_dim, latent_dim)  # Project concatenated features to latent_dim
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
    
    def forward(self, x, xf, xw, xs):
        """
        x: B, T, D - Noisy latent sequence
        xf: B, 768 - Audio features
        xw: B, N, D - Text features (transcripts)
        xs: B, S, D - Semantic features
        """
        B, T, D = x.shape
        N = xw.shape[1]
        S = xs.shape[1]
        H = self.num_head

        # Project audio features to match text feature dimension
        xf_proj = self.audio_proj(xf)  # B, text_feature_dim

        # Expand xf_proj to have the same sequence length as xw
        xf_proj = xf_proj.unsqueeze(1).expand(-1, N, -1)  # B, N, text_feature_dim

        # Concatenate audio and text features
        xw_xf = torch.cat((xw, xf_proj), dim=1)  # B, N+N, text_feature_dim
        xw_xf = self.audio_text_proj(xw_xf)  # B, N+N, latent_dim

        # Normalize and project the inputs
        query = self.query(self.norm(x))  # B, T, D
        key = self.key(self.text_norm(xw_xf))  # B, N+N, D
        semantic_key = self.key(self.semantic_norm(xs))  # B, S, D

        # Reshape and apply softmax
        query = query.view(B, T, H, -1)  # B, T, H, D/H
        key = key.view(B, 2 * N, H, -1)  # B, N+N, H, D/H
        semantic_key = semantic_key.view(B, S, H, -1)  # B, S, H, D/H
        
        query = F.softmax(query, dim=-1)  # Softmax over the last dimension
        key = F.softmax(key, dim=1)  # Softmax over the second dimension (N+N)
        semantic_key = F.softmax(semantic_key, dim=1)  # Softmax over the second dimension (S)
        
        # Compute values
        value = self.value(self.text_norm(xw_xf)).view(B, 2 * N, H, -1)  # B, N+N, H, D/H
        semantic_value = self.value(self.semantic_norm(xs)).view(B, S, H, -1)  # B, S, H, D/H

        # Compute attention
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)  # B, H, D/H, D/H
        semantic_attention = torch.einsum('bshd,bshl->bhsl', semantic_key, semantic_value)  # B, H, S, D/H

        # Apply attention
        y = torch.einsum('bthd,bhdl->bthl', query, attention).reshape(B, T, D)  # B, T, D
        semantic_y = torch.einsum('bthd,bhsl->bthl', query, semantic_attention).reshape(B, T, D)  # B, T, D

        # Combine outputs
        y = y + semantic_y

        # Apply final projection and residual connection
        y = x + self.proj_out(y)
        return y



class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(latent_dim)
        self.proj_out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, x):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = self.norm(y)
        y = self.proj_out(y)
        y = x + y
        return y

class AddNorm(nn.Module):
      def __init__(self, normalized_shape, dropout, **kwargs):
          super(AddNorm, self).__init__(**kwargs)
          self.dropout = nn.Dropout(dropout)
          self.ln = nn.LayerNorm(normalized_shape)
 
      def forward(self, X, Y):
          return self.ln(self.dropout(Y) + X)


class SemanticAttentionTransformerLayer(nn.Module):
    def __init__(self,
                 opt,
                 seq_len=60,
                 latent_dim=32,
                 aud_latent_dim=512,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1,
                 cond_proj=True):
        super(SemanticAttentionTransformerLayer, self).__init__()
        self.opt = opt
        pre_proj_dim = latent_dim
        if cond_proj:
            if "linear" in self.opt.cond_projection:
                self.feat_proj = nn.Linear(pre_proj_dim, latent_dim)
            elif "mlp" in self.opt.cond_projection:
                self.feat_proj = nn.Sequential(
                    nn.LayerNorm(pre_proj_dim),
                    nn.Linear(pre_proj_dim, latent_dim*2),
                    nn.SiLU(),
                    nn.Linear(latent_dim*2, latent_dim),
                )

        self.sa_block = SelfAttention(latent_dim, num_head, dropout)
        self.ca_block = CrossAttention(latent_dim, aud_latent_dim, num_head, dropout)
        self.addnorm = AddNorm(latent_dim, dropout)
        self.ffn = FFN(latent_dim, ffn_dim, dropout)

    def forward(self, x, xf, xw, xs, add_cond={}, null_cond_emb=None):
        x_ori = x.detach().clone() if x.requires_grad else x.clone()
        # if self.opt.classifier_free and xf is not None:
        #     if self.training:
        #         mask = (torch.rand(x.shape[0]) < self.opt.null_cond_prob).to(x.device)
        #         x = torch.where(mask.unsqueeze(1).unsqueeze(2), null_cond_emb.repeat(x.shape[1], 1).unsqueeze(0), x)
        #     elif self.opt.cond_scale != 1:
        #         mask = (torch.rand(x.shape[0]) < 0.5).to(x.device)
        #         x = torch.where(mask.unsqueeze(1).unsqueeze(2), null_cond_emb.repeat(x.shape[1], 1).unsqueeze(0), x)

        x = self.feat_proj(x)
        y = x + x_ori

        y_1 = self.addnorm(y,self.sa_block(y))

   
        y_2 = self.addnorm(y_1,self.ca_block(y_1,xf, xw, xs))

        y_3 = self.addnorm(y_2,self.ffn(y_2))
        return y_3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create a long enough PE matrix with shape [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model] for broadcasting over batch size
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x should have dimensions [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x.permute(1, 0, 2)


class SemanticAnotationEncoder(nn.Module):
    def __init__(self, annotation_emb_dim, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(SemanticAnotationEncoder, self).__init__()
        self.linear = nn.Linear(annotation_emb_dim, 256)
        self.pos_encoder = PositionalEncoding(256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    
    def forward(self, src):
        input_dim = src.size(-1)
        if self.linear.in_features != input_dim:
            self.linear = nn.Linear(input_dim, 256).cuda()
        src = self.linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)



class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



class TextEncoderTCN(nn.Module):
    """ based on https://github.com/locuslab/TCN/blob/master/TCN/word_cnn/model.py """
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None,
                 kernel_size=2, dropout=0.3, emb_dropout=0.1):
        super(TextEncoderTCN, self).__init__()
        #print(pre_trained_embedding.shape[0])

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            #print(pre_trained_embedding.shape)
            assert pre_trained_embedding.shape[0] == n_words
            assert pre_trained_embedding.shape[1] == embed_size
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),
                                                          freeze=args.freeze_wordembed)
        else:
            self.embedding = nn.Embedding(n_words, embed_size)

        num_channels = [args.hidden_size] * args.n_layer
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.init_weights()

    def init_weights(self):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous(), 0



class MotionTransformer(nn.Module):
    def __init__(self,
                 opt,
                 input_feats,
                 audio_dim=128,
                 style_dim=4,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu",
                 num_text_layers=4,
                 aud_latent_dim=256,
                 annotation_emb_dim=2048,
                 text_num_heads=4,
                 no_clip=False,
                 pe_type='learnable',
                 block=None,
                 **kargs):
        super().__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.annotation_emb_dim = annotation_emb_dim


        self.audio_dim = audio_dim

        self.opt = opt

        # self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, 512))

        if pe_type == 'learnable':
            self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
        elif pe_type == 'ppe_sinu':
            self.PPE = PeriodicPositionalEncoding(latent_dim, period=25)
        elif pe_type == 'pe_sinu':
            self.PE = PeriodicPositionalEncoding(latent_dim, period=600)
        elif pe_type == 'ppe_sinu_dropout':
            self.PPE_drop = PeriodicPositionalEncoding(latent_dim, period=25)
        elif pe_type == 'pe_sinu_repeat':
            self.PE = PeriodicPositionalEncoding(latent_dim, period=200)
        
        self.pre_proj_dim = latent_dim
        if self.opt.cond_projection in ["linear_excludeX", "mlp_excludeX"]:
            self.pre_proj_dim = 0

        self.pre_proj_dim = self.pre_proj_dim + opt.pose_latent_dim

        self.null_cond_emb = None
        if self.opt.classifier_free:
            self.null_cond_emb = nn.Parameter(torch.randn(1, self.pre_proj_dim))
                
        self.joint_embed = nn.Linear(self.input_feats, latent_dim)
        self.audio_proj = nn.Linear(audio_dim, aud_latent_dim)

        with open(opt.data_path_1 +  "/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_encoder = TextEncoderTCN(opt, opt.word_index_num, opt.word_dims, pre_trained_embedding=pre_trained_embedding,
                                           dropout=opt.dropout_prob)
        self.semanticanotation_encoder = SemanticAnotationEncoder(annotation_emb_dim=34, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1)
        if self.opt.encode_hubert:
            self.hubert_encoder = nn.Sequential(*[
                nn.Conv1d(1024, 128, 3, 1, 1, bias=False),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, 3, 1, 1, bias=False)
            ])
        elif self.opt.encode_wav2vec2:
            self.hubert_encoder = nn.Linear(768, 256)


        self.temporal_decoder_blocks = nn.ModuleList()

        for i in range(num_layers):
            self.temporal_decoder_blocks.append(
                SemanticAttentionTransformerLayer(
                    opt=opt,
                    seq_len=num_frames,
                    latent_dim=latent_dim,
                    aud_latent_dim=aud_latent_dim,
                    ffn_dim=ff_size,
                    num_head=num_heads,
                    dropout=dropout
                )
            )
        
        self.out = nn.Linear(latent_dim, self.input_feats)
        # self._reset_parameters()
        
        
    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.opt.pose_latent_dim ** -0.5)
    def generate_src_mask(self, T, length, causal=False):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        
        if causal:
            for i in range(B):
                for j in range(i + 1, length[i]):
                    src_mask[i, j] = 0

        return src_mask
    

    def forward(self, x, audio_emb, word_emb, sem_emb,pe_type="learnable", block=None): #x, audio_emb, word_emb, sem_emb, pe_type, block="gesture"
        add_cond = {}
        add_cond_new = add_cond.copy()

        word_feat_seq, _ = self.text_encoder(word_emb)

        sem_feat_seq = self.semanticanotation_encoder(sem_emb)

    

        B, T = x.shape[0], x.shape[1]
        length = torch.LongTensor([T for ii in range(B)]).to(x.device)
        # masked_embeddings = self.mask_embeddings.expand_as(x)
        # masked_motion = torch.where(mask == 1, masked_embeddings, x) # bs, t, 256 
        src_mask = self.generate_src_mask(T, length).to(x.device).unsqueeze(-1)

        x = x * src_mask


        
        h = self.joint_embed(x)

        if pe_type == 'learnable':
            h = h + self.sequence_embedding.unsqueeze(0)[:, :T, :]
        elif pe_type == 'ppe_sinu':
            h = self.PPE(h)
        elif pe_type == 'pe_sinu' or pe_type == 'pe_sinu_repeat':
            h = self.PE(h)
        elif pe_type == 'ppe_sinu_dropout':
            h = self.PPE_drop(h, dropout=True)
        



        for module in self.temporal_decoder_blocks:
            h = module(h, audio_emb, word_feat_seq, sem_feat_seq, add_cond=add_cond_new, null_cond_emb=self.null_cond_emb)
        output = self.out(h).view(B, T, -1).contiguous()

        if self.opt.classifier_free and not self.training and self.opt.cond_scale != 1:
            output = output[:output.shape[0] // 2] + self.opt.cond_scale * (output[output.shape[0] // 2:] - output[:output.shape[0] // 2])
        return output


class SemanticTransformer(nn.Module):
    def __init__(self,
                 opt,
                 input_feats,
                 audio_dim = 128,
                 style_dim = 4,
                 num_frames=240,
                 latent_dim=512,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_text_layers=4,
                 aud_latent_dim=256,
                 annotation_emb_dim=2048,
                 text_num_heads=4,
                 no_clip=False,
                 pe_type='learnable',
                 **kargs):
        super(SemanticTransformer, self).__init__()
        
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats

        self.opt = opt

        
        
        self.opt.expCondition_gesture_only = 'pred' 
        self.semanticlayer = MotionTransformer(
                        opt=opt,
                        input_feats=opt.pose_latent_dim,
                        audio_dim=audio_dim*2,
                        style_dim=style_dim,
                        num_frames=num_frames,
                        num_layers=num_layers,
                        ff_size=ff_size,
                        latent_dim=latent_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        activation=activation, 
                        num_text_layers=num_text_layers,
                        aud_latent_dim=aud_latent_dim,
                        annotation_emb_dim=annotation_emb_dim,
                        text_num_heads=text_num_heads,
                        no_clip=no_clip,
                        pe_type=pe_type,
                        block="gesture"
                        )
    
    
    def forward(self, x, word_emb, sem_emb, audio_emb, pe_type="learnable"): #x_start, text_emb, sem_emb, audio_emb, self.opt.PE

        ges = self.semanticlayer(x, audio_emb, word_emb, sem_emb, pe_type, block="gesture")

        
        return ges
    




class SemanticsLatentDiffusion(nn.Module):

    def __init__(self, args, eval_model=None):
        super(SemanticsLatentDiffusion, self).__init__()
        self.opt = args
        self.epoch = 0
        self.eval_model = eval_model
        if eval_model is not None and 'test' not in self.opt.mode:
            self.load_fid_net(args.e_path)
            self.eval_model.eval()
            
        self.diffusion_steps = args.diffusion_steps

        self.SemanticTransformerLayer = SemanticTransformer(
            opt=args,
            input_feats=args.motion_f,
            audio_dim=args.audio_f,
            num_frames=args.max_frame,
            latent_dim=args.motion_f,
            ff_size=args.ff_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            activation=args.activation,
            num_text_layers=args.num_text_layers,
            aud_latent_dim=args.aud_latent_dim,
            text_num_heads=args.text_num_heads,
            pe_type=args.PE
        )
        self.index_map = MLP(args.index_hidden_dim, args.hidden_size, args.vae_codebook_size)


        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        # self.encoder = VQEncoderV5(args)
        # self.quantizer = Quantizer(args.vae_codebook_size, args.vae_length, args.vae_quantizer_lambda)
        # self.decoder = VQDecoderV5(args)
        # self.fc = nn.Linear(in_features=768, out_features=17408, bias=True)

    def forward(self, batch_data):
        motions_hands = batch_data["in_pre_pose_hand"]
        motions_body = batch_data["in_pre_pose_body"]

        audio_emb = batch_data["in_audio"]
        text_emb = batch_data["in_word"]
        sem_emb = batch_data["in_sem"]

        self.audio_emb = audio_emb
        self.motions_hands = motions_hands
        self.motions_body = motions_body
        self.text_emb = text_emb
        self.sem_emb = sem_emb

        x_start_hands = motions_hands
        x_start_body = motions_body

        output_hands = self.SemanticTransformerLayer(x_start_hands, text_emb, sem_emb, audio_emb, self.opt.PE)
        output_body = self.SemanticTransformerLayer(x_start_body, text_emb, sem_emb, audio_emb,self.opt.PE)
        index_hands = self.index_map(output_hands)
        index_body = self.index_map(output_body)


        return {
            "poses_feat_hands":output_hands,
            "poses_feat_body":output_body,
            "index_hands":index_hands,
            "index_body":index_body,

            }
    




