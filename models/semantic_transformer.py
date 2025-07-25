import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Literal
import pytorch_lightning as pl
import numpy as np
from einops.layers.torch import Rearrange
import pickle
from torch.nn.utils import weight_norm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Norm(nn.Module):
    """ Norm Layer """

    def __init__(self, fn, size):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-5)
        self.fn = fn

    def forward(self, x_data):
        x, word_emb = x_data
        x_norm, _ = self.fn((self.norm(x), word_emb))
        return (x_norm, word_emb)


class Residual(nn.Module):
    """ Residual Layer """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x_data):
        x, word_emb = x_data
        x_resid, _ = self.fn(x_data)
        return (x_resid + x, word_emb)


class MLP(nn.Module):
    """ MLP Layer """

    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_data):
        x, word_emb = x_data
        out = self.l2(self.activation(self.l1(x)))
        return (out, word_emb)


class Attention(nn.Module):
    def __init__(self, in_dim, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(in_dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.rearrange_qkv = Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")

    def forward(self, x_data):
        x, word_emb = x_data
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return (out, word_emb)


class CrossModalAttention(nn.Module):
    """ Cross Modal Attention Layer """

    def __init__(self, in_dim, dim, heads=8, in_dim2=None):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        if in_dim2 is not None:
            self.to_kv = nn.Linear(in_dim2, dim * 2, bias=False)
        else:
            self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.to_q = nn.Linear(in_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.rearrange_qkv = Rearrange(
            "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")

    def forward(self, x_data):
        x_a, x_b = x_data
        kv = self.to_kv(x_b)
        q = self.to_q(x_a)

        qkv = torch.cat((q, kv), dim=-1)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return (out, x_b)


class Transformer(nn.Module):
    def __init__(self, in_size=50, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072):
        super().__init__()
        blocks = []
        for i in range(num_hidden_layers):
            blocks.extend([
                Residual(Norm(Attention(in_size, hidden_size, heads=num_attention_heads), hidden_size)),
                Residual(Norm(CrossModalAttention(in_size, hidden_size, heads=num_attention_heads, in_dim2=hidden_size), hidden_size)),
                Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size), hidden_size))
            ])
        self.net = torch.nn.Sequential(*blocks)

    def forward(self, x_data):
        x, word_emb = x_data
        x, _ = self.net((x, word_emb))
        return x


class LinearEmbedding(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.net = nn.Linear(size, dim)

    def forward(self, x):
        return self.net(x)


class TransformerPredictor(pl.LightningModule):
    def __init__(self, latent_dim: int, num_layers: int, num_heads: int, intermediate_size: int, audio_dim: int, one_hot_dim: int):
        super().__init__()
        self.audio_feature_map = nn.Linear(768 * 2, latent_dim)
        self.style_embedding = nn.Linear(one_hot_dim, latent_dim, bias=False)
        self.squasher = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(0.2, True),
            nn.InstanceNorm1d(latent_dim, affine=False)
        )
        self.encoder_transformer = Transformer(
            in_size=latent_dim, hidden_size=latent_dim, num_hidden_layers=num_layers, num_attention_heads=num_heads, intermediate_size=intermediate_size)
        self.encoder_pos_embedding = PositionalEncoding(latent_dim, batch_first=True)
        self.encoder_linear_embedding = LinearEmbedding(latent_dim, latent_dim)
        self.proj = nn.Linear(512, 256)

    def forward(self, audio, id_feature, word_feat_seq):
        inputs = self.audio_feature_map(audio)
        # print('4:',id_feature.shape)
        id_feature = id_feature.repeat(1, 34, 1)
        # print('5:',inputs.shape)
        # print('6:',id_feature.shape)
        inputs = torch.cat((inputs, id_feature), 2)
        inputs = self.proj(inputs)
        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, word_feat_seq))
        return encoder_features



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
        # print(pre_trained_embedding.shape[0])

        if pre_trained_embedding is not None:  # use pre-trained embedding (fasttext)
            print(pre_trained_embedding.shape)
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


class SemanticsLatentTransformer(nn.Module):
    def __init__(self, args):
        super(SemanticsLatentTransformer, self).__init__()
        self.SemanticTransformerLayer = TransformerPredictor(
            latent_dim=args.latent_dim, num_layers=args.num_layers, num_heads=args.num_heads,
            intermediate_size=args.intermediate_size, audio_dim=args.audio_dim, one_hot_dim=args.one_hot_dim
        )
        self.index_map = nn.Linear(args.index_hidden_dim, args.vae_codebook_size)
        self.proj = nn.Linear(256, 512)
        if args.dataset_name == "beat":
            self.spearker_encoder_body = nn.Sequential(
                    nn.Embedding(34, 256),
                    nn.Linear(256, 256), 
                    nn.LeakyReLU(0.1, True)
                )
            self.spearker_encoder_hand = nn.Sequential(
                    nn.Embedding(34, 256),
                    nn.Linear(256, 256), 
                    nn.LeakyReLU(0.1, True)
                )
        else:
            self.spearker_encoder_body = nn.Sequential(
                    nn.Embedding(2048, 256),
                    nn.Linear(256, 256), 
                    nn.LeakyReLU(0.1, True)
                )
            self.spearker_encoder_hand = nn.Sequential(
                    nn.Embedding(2048, 256),
                    nn.Linear(256, 256), 
                    nn.LeakyReLU(0.1, True)
                )
        if args.dataset_name == "beat":
            with open(args.data_path_1 +  "/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                pre_trained_embedding = self.lang_model.word_embedding_weights
        else: 
            with open(args.cache_path +  "/vocab_cache.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_encoder = TextEncoderTCN(args, args.word_index_num, args.word_dims, pre_trained_embedding=pre_trained_embedding,
                                           dropout=args.dropout_prob)
        self.dataset_name = args.dataset_name

    def forward(self, batch_data):
        in_id = batch_data["tar_id"]
        audio_emb = batch_data["in_audio"]
        # emotion = batch_data["in_emo"]
        in_word = batch_data["in_word"]
        word_feat_seq, _ = self.text_encoder(in_word)
        # print('1:', in_id.shape)
        # Encode speaker embeddings
        speaker_embedding_hand = self.spearker_encoder_hand(in_id)

        speaker_embedding_body = self.spearker_encoder_body(in_id)
        if self.dataset_name == "beat":
            speaker_embedding_hand = speaker_embedding_hand.squeeze(1)
            speaker_embedding_body = speaker_embedding_body.squeeze(1)

        # Feature Fusion and Output Generation
        output_hands = self.SemanticTransformerLayer(audio_emb, speaker_embedding_hand, word_feat_seq)
        output_hands = self.proj(output_hands)
        
        output_body = self.SemanticTransformerLayer(audio_emb, speaker_embedding_body, word_feat_seq)
        output_body = self.proj(output_body)
        word_feat_seq = self.proj(word_feat_seq)

        return {
            "poses_feat_hands": output_hands,
            "poses_feat_body": output_body,
            "word_feat_seq": word_feat_seq
        }
