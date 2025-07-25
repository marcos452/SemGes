import logging

logger = logging.getLogger(__name__)

from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from ..base import OrderedNamespace
from ..module import ClipModel, MLPLayers
from ..module.cif import CIF
from ..module.kw_modules import TransformerModels
from ..module.speechclip_c_modules import vector_quantizers
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm, Kw_BatchNorm_dynamic
from ..util import get_keypadding_mask


class GeneralBranch(nn.Module):
    def __init__(self, config: OrderedNamespace, audio_dim: int, text_dim: int) -> None:
        super().__init__()
        logger.info(f"Using {type(self).__name__}")
        self.config = config
        self.audio_dim = audio_dim
        self.text_dim = text_dim

    def _create_self_attn_layer(self, branch_config: OrderedNamespace):
        """create self-attention layer"""
        transformer_args = branch_config.transformer_args
        transformer_type = (
            transformer_args.type
            if hasattr(transformer_args, "type")
            else branch_config.transformer_type
        )
        logger.info(f"Using {transformer_type} as {type(self).__name__}")
        self.self_att = getattr(TransformerModels, transformer_type)(
            **branch_config.transformer_args
        )

    def _create_kw_proj_layer(self):
        """create projection layer of keyword embeddings (before BatchNorm Layer)"""
        self.kw_projection_config = (
            self.config.model_settings.cascaded_branch.keyword.get(
                "kw_projection", None
            )
        )
        if self.kw_projection_config is None:
            logger.info(
                "kw_projection not specified, using single linear layer as default"
            )
            self.linear_proj = nn.Linear(
                self.config.model_settings.cascaded_branch.transformer_args.d_model,
                self.text_dim,
            )
        else:
            logger.info(
                f"kw_projection dims:{self.kw_projection_config.dimensions} droupout:{self.kw_projection_config.dropout}"
            )
            assert (
                self.kw_projection_config.dimensions[0]
                == self.config.model_settings.cascaded_branch.transformer_args.d_model
            ), f"first dim({self.kw_projection_config.dimensions[0]}) should match the audio encoder dim({self.config.model_settings.cascaded_branch.transformer_args.d_model})"
            assert (
                self.kw_projection_config.dimensions[-1] == self.text_dim
            ), f"last dim({self.kw_projection_config.dimensions[-1]}) should match the text encoder dim({self.text_dim})"
            self.linear_proj = MLPLayers(
                units=self.kw_projection_config.dimensions,
                dropout=self.kw_projection_config.dropout,
            )

    def _create_vector_quantizer(self):
        """create vector quantizer for the codebook selection"""
        self.vector_quantizer = None
        self.vq_type = self.config.model_settings.cascaded_branch.vq.type

        if not hasattr(
            vector_quantizers, self.config.model_settings.cascaded_branch.vq.type
        ):
            raise NotImplementedError(
                "Vq ({}) not implemented".format(
                    self.config.model_settings.cascaded_branch.vq.type
                )
            )

        self.vector_quantizer = getattr(vector_quantizers, self.vq_type)(
            **self.config.model_settings.cascaded_branch.vq.args
        )

    def _create_kw_batchnorm(self):
        """create batchnorm layer for the keyword"""
        self.bn_layer = Kw_BatchNorm(
            kw_num=self.keyword_num,
            kw_dim=self.text_dim,
            batchnorm_type=self.config.model_settings.cascaded_branch.keyword.batchnorms.type,
            init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
            init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
            std_scale=self.config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
            learnable=(
                self.config.model_settings.cascaded_branch.keyword.batchnorms.learnable
                if hasattr(
                    self.config.model_settings.cascaded_branch.keyword.batchnorms,
                    "learnable",
                )
                else True
            ),
            parallel=(
                self.config.model_settings.cascaded_branch.keyword.batchnorms.parallel
                if hasattr(
                    self.config.model_settings.cascaded_branch.keyword.batchnorms,
                    "parallel",
                )
                else False
            ),
        )

    def _create_cls(self, length: int, cls_dim: int) -> nn.Parameter:
        """create CLS token

        Args:
            length (int): length of the CLS token
            cls_dim (int): embedding dimension of the CLS token

        Returns:
            nn.Parameter: CLS token with the shape [1, lenght, cls_dim]
        """
        cls = torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    length,
                    cls_dim,
                ]
            )
        )
        logger.info("Start init [CLS] {}".format(cls.shape))

        return cls

    def project_feats_to_CLIPspace(self, features: torch.Tensor) -> torch.Tensor:
        """Batchnormalized the input features to fit CLIP's embedding space

        Args:
            features (torch.Tensor): input features

        Returns:
            Batch normalized features
        """
        features = self.linear_proj(features)
        if hasattr(self, "bn_layer"):
            features = self.bn_layer(features)

        return features

    def get_keyword_cosine_score(self, keywords: torch.Tensor) -> torch.Tensor:
        """Compute cosine scores between the keyword embeddings and the CLIP tokens' embeddings

        Args:
            keywords (torch.Tensor): keyword embeddings

        Returns:
            cos_score (torch.Tensor): cosine scores between the keyword embeddings and the CLIP tokens' embeddings
        """
        B, N = keywords.shape[:2]
        cos_score = []
        for i in range(N):
            cos_score.append(
                F.cosine_similarity(
                    keywords[:, i, :].view(B, self.text_dim, 1),
                    self.clip.model.token_embedding.weight.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            )
        cos_score = torch.stack(cos_score, dim=1).to(keywords.device)

        return cos_score

    def vq_audio_features(self, audio_feat: torch.Tensor) -> Tuple[dict, torch.Tensor]:
        """extract CLIP's text tokens (keywords) by performing vector-quantization to the input audio features

        Args:
            audio_feat (torch.Tensor): input audio features

        Returns:
            Tuple[dict, torch.Tensor]: vq_results, keywords
        """

        audio_feat = self.project_feats_to_CLIPspace(audio_feat)
        cos_score = self.get_keyword_cosine_score(audio_feat)
        vq_results = self.vector_quantizer(x=cos_score)
        assert self.clip.model.token_embedding.weight.requires_grad == False
        keywords = vq_results["subword_prob"] @ self.clip.model.token_embedding.weight

        return vq_results, keywords


class KW_ParallelBranch(GeneralBranch):
    """KW_ParallelBranch

    The parallel branch of SpeechCLIP

    """

    def __init__(self, config: OrderedNamespace, audio_dim: int, text_dim: int) -> None:
        super().__init__(config, audio_dim, text_dim)
        # select the transformer structure for main architecture for parallel branch'
        self.self_att = self._create_self_attn_layer(
            self.config.model_settings.parallel_branch
        )
        self.cls = self._create_cls(
            length=1,
            cls_dim=self.config.model_settings.parallel_branch.transformer_args.d_model,
        )
        self.need_projection = self.config.model_settings.parallel_branch.get(
            "need_projection", True
        )
        if self.need_projection:
            self.linear_proj = nn.Linear(self.audio_dim, self.text_dim)

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + 1
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + 1
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        # exclude CLS position
        hidden_states = [x[:, 1:, ...] for x in hidden_states]

        return tuple(hidden_states)

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_len: torch.Tensor,
        otherInputs: dict = None,
    ) -> dict:
        """forward

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):
            otherInputs (dict)
        Returns:
            dict
        """
        output = defaultdict(lambda: None)
        bsz, audio_max_len = audio_feat.shape[:2]
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(
            max_length=audio_max_len + 1,
            data_lens=audio_len + 1,
        )
        out = self.self_att(src=src, key_padding_mask=key_padding_mask)
        out = out[:, :1].reshape(-1, self.audio_dim)

        if hasattr(self, "linear_proj"):
            out = self.linear_proj(out)

        output["parallel_audio_feat"] = out

        return output


class KW_CascadedBranch(GeneralBranch):
    """KW_CascadedBranch

    Cascaded Branch for SpeechCLIP

    """

    def __init__(self, config, audio_dim: int, text_dim: int, clip: ClipModel) -> None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            clip (ClipModel): the CLIP model

        """
        super().__init__(config=config, audio_dim=audio_dim, text_dim=text_dim)
        self.clip = clip
        self.keyword_num = getattr(
            config.model_settings.cascaded_branch.keyword, "number", 8
        )
        self.cls = self._create_cls(
            length=self.keyword_num,
            cls_dim=self.config.model_settings.cascaded_branch.transformer_args.d_model,
        )
        self._create_self_attn_layer(
            self.config.model_settings.cascaded_branch
        )  # select the main structure for transformer encoder layer
        self._create_kw_proj_layer()  # projection network (before BatchNorm Layer)
        self._create_vector_quantizer()  # codebook selection

        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self._create_kw_batchnorm()  # batchnorms

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extracting hidden representation of each layers

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: tuples of hiddenstates
        """
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )[-1]
        # exclude the cls positions
        hidden_states = [x[:, self.keyword_num :, ...] for x in hidden_states]

        return hidden_states

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.Tensor,
        otherInputs: dict = None,
    ) -> dict:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_len (torch.Tensor)
            otherInputs (dict)
        Returns:
            dict
        """
        output = defaultdict(lambda: None)
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_feat_len + self.keyword_num
        )
        audio_feat = self.self_att(src=src, key_padding_mask=key_padding_mask)
        audio_feat = audio_feat[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )
        vq_results, keywords = self.vq_audio_features(audio_feat)
        output["vq_results"] = vq_results
        output["keywords"] = keywords
        output["cascaded_audio_feat"] = self.clip.encode_keywords(
            keywords, self.keyword_num
        )  # Feed keywords into clip text encoder

        return output

    def getAttentionMap(self, audio_feat: torch.Tensor, audio_len: torch.Tensor):
        """getAttentionMap

        return attention maps for visualization

        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: cls_weights, topk_kw, None
        """
        # Use multi-head attention layer to find keywords(cls)
        bsz, total_max_len = audio_feat.size(0), audio_feat.size(1) + self.keyword_num
        cls = torch.cat([self.cls] * bsz, dim=0)
        src = torch.cat([cls, audio_feat], dim=1)

        key_padding_mask = get_keypadding_mask(
            max_length=total_max_len, data_lens=audio_len + self.keyword_num
        )

        _, attn_output_weights = self.self_att.extract_attention_map(
            src=src, key_padding_mask=key_padding_mask
        )

        cls_weights = []
        for i in range(attn_output_weights.shape[0]):
            cls_weights.append(
                attn_output_weights[
                    i, :, : self.keyword_num, : audio_len[i] + self.keyword_num
                ]
            )

        audio_feat = self.self_att(src=src, key_padding_mask=key_padding_mask)
        audio_feat = audio_feat[:, : self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )
        audio_feat = self.project_feats_to_CLIPspace(audio_feat)
        cos_score = self.get_keyword_cosine_score(audio_feat)

        # disallow special tokens
        cos_score[..., 0] -= 100
        cos_score[..., 2] -= 100
        cos_score[..., 3] -= 100

        assert cos_score.shape == (
            bsz,
            self.keyword_num,
            self.clip.model.token_embedding.num_embeddings,
        ), f"{cos_score.shape}, {( bsz, self.keyword_num, self.clip.model.token_embedding.num_embeddings)}"

        topk_kw = [[[] for _ in range(self.keyword_num)] for _ in range(bsz)]
        _, topk_kw_ids = torch.topk(cos_score, dim=-1, k=10)
        for bsz_i in range(bsz):
            for kw_i in range(self.keyword_num):
                topk_kw[bsz_i][kw_i] = [
                    self.clip.tokenizer.decoder[
                        self.clip.reducedl2Original[x.item()]
                        # top1_kw_id[bsz_i, kw_i].item()
                    ].replace("</w>", "")
                    for x in topk_kw_ids[bsz_i, kw_i]
                ]

        return cls_weights, topk_kw, None


class KW_HybridBranch(GeneralBranch):
    """KW_CascadedBranch

    Hybrid Branch (parallel + cascaded) for SpeechCLIP

    """

    def __init__(
        self, config, audio_dim: int, text_dim: int, out_dim: int, clip: ClipModel
    ) -> None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            clip (ClipModel): the CLIP model

        """
        super().__init__(config=config, audio_dim=audio_dim, text_dim=text_dim)
        self.clip = clip
        self.out_dim = out_dim
        self.pbranch_config = config.model_settings.parallel_branch
        self.cbranch_config = config.model_settings.cascaded_branch
        self.keyword_num = getattr(
            config.model_settings.cascaded_branch.keyword, "number", 8
        )
        self.parallel_cls = self._create_cls(
            length=1, cls_dim=self.pbranch_config.transformer_args.d_model
        )
        self.cascaded_cls = self._create_cls(
            length=self.keyword_num,
            cls_dim=self.cbranch_config.transformer_args.d_model,
        )
        self._create_self_attn_layer(
            self.config.model_settings.cascaded_branch
        )  # select the main structure for transformer encoder layer
        self._create_kw_proj_layer()  # projection network (before BatchNorm Layer)
        self._create_vector_quantizer()  # codebook selection
        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self._create_kw_batchnorm()  # batchnorms

        if getattr(self.pbranch_config, "need_projection", True):
            if hasattr(self.pbranch_config, "projection_config"):
                logger.info(
                    f"parallel projection dims:{self.pbranch_config.projection_config.dimensions} droupout:{self.pbranch_config.projection_config.dropout}"
                )
                self.parallel_proj = MLPLayers(
                    units=self.pbranch_config.projection_config.dimensions,
                    dropout=self.pbranch_config.projection_config.dropout,
                )
            else:
                logger.info(
                    "parallel projection not specified, using single linear layer as default"
                )
                self.parallel_proj = nn.Linear(self.audio_dim, self.out_dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.Tensor,
        otherInputs: dict = None,
    ) -> dict:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_len (torch.Tensor)
            otherInputs (dict)
        Returns:
            dict
        """
        output = defaultdict(lambda: None)
        bsz, max_audio_length = audio_feat.shape[:2]
        parallel_cls = torch.cat([self.parallel_cls] * bsz, dim=0)
        cascaded_cls = torch.cat([self.cascaded_cls] * bsz, dim=0)
        cls = torch.cat([parallel_cls, cascaded_cls], dim=1)
        src = torch.cat([cls, audio_feat], dim=1)
        attn_out = self.self_att(
            src=src,
            key_padding_mask=get_keypadding_mask(
                max_audio_length + cls.shape[1], audio_feat_len + cls.shape[1]
            ),
        )
        output["parallel_audio_feat"] = self.parallel_proj(
            attn_out[:, :1].reshape(-1, self.audio_dim)
        )
        post_audio_feat = attn_out[:, 1 : 1 + self.keyword_num].reshape(
            -1, self.keyword_num, self.audio_dim
        )  # audio features after self-attention layer
        vq_results, keywords = self.vq_audio_features(post_audio_feat)
        output["vq_results"] = vq_results
        output["keywords"] = keywords
        output["cascaded_audio_feat"] = self.clip.encode_keywords(
            keywords, self.keyword_num
        )  # Feed keyword into clip text encoder

        return output

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, max_audio_length = audio_feat.shape[:2]
        parallel_cls = torch.cat([self.parallel_cls] * bsz, dim=0)
        cascaded_cls = torch.cat([self.cascaded_cls] * bsz, dim=0)
        cls = torch.cat([parallel_cls, cascaded_cls], dim=1)
        src = torch.cat([cls, audio_feat], dim=1)
        key_padding_mask = get_keypadding_mask(
            max_length=max_audio_length + cls.shape[1],
            data_lens=audio_len + cls.shape[1],
        )

        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        # exclude CLS position
        hidden_states = [x[:, cls.shape[1] :, ...] for x in hidden_states]

        return tuple(hidden_states)


class KW_CascadedBranchPlus(GeneralBranch):
    """KW_CascadedBranchPlus

    Cascaded Branch plus for SpeechCLIP+
    """

    def __init__(
        self, config: OrderedNamespace, audio_dim: int, text_dim: int, clip: ClipModel
    ) -> None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            clip (ClipModel): the CLIP model

        """
        super().__init__(config, audio_dim, text_dim)
        self.clip = clip
        logger.info("Using self-attention before downsampling")
        self._create_self_attn_layer(
            self.config.model_settings.cascaded_branch
        )  # select the main structure for transformer encoder layer
        self._create_kw_proj_layer()  # projection network (before BatchNorm Layer)
        self._create_vector_quantizer()  # codebook selection

        if hasattr(config.model_settings.cascaded_branch.keyword, "batchnorms"):
            self._create_kw_batchnorm()

        # downsampling
        assert hasattr(config.model_settings.cascaded_branch.downsampling, "type")
        self.downsampling_type = config.model_settings.cascaded_branch.downsampling.type
        if self.downsampling_type == "cif":
            self.using_gt_len = config.model_settings.cascaded_branch.downsampling.get(
                "using_gt_len", False
            )
            if self.using_gt_len:
                logger.info("Using ground truth text length target")
            self.downsampling = CIF(
                **config.model_settings.cascaded_branch.downsampling.cif.__dict__
            )
        else:
            raise NotImplementedError(
                "Unknown type:{}".format(config.downsampling.type)
            )
        logger.info("Using {} downsampling method".format(self.downsampling_type))

    def _create_kw_batchnorm(self):
        self.bn_layer = Kw_BatchNorm_dynamic(
            kw_dim=self.text_dim,
            init_bias=torch.mean(self.clip.model.token_embedding.weight, dim=0),
            init_scale=torch.std(self.clip.model.token_embedding.weight, dim=0),
            std_scale=self.config.model_settings.cascaded_branch.keyword.batchnorms.std_scale,
            learnable=(
                self.config.model_settings.cascaded_branch.keyword.batchnorms.learnable
                if hasattr(
                    self.config.model_settings.cascaded_branch.keyword.batchnorms,
                    "learnable",
                )
                else True
            ),
        )

    def downsampling_audio_feat(
        self,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.LongTensor,
        audio_feat_pad_mask: torch.Tensor,
        global_step: int = 0,
        target_len: torch.Tensor = None,
    ) -> dict:
        """Downsample the input audio features

        Args:
            audio_feat (torch.Tensor): audio features that will be downsampled
            audio_feat_len (torch.LongTensor): lengths of audio features
            audio_feat_pad_mask (torch.Tensor): key padding mask of audio features, 1 (True) indicates the padding part
            global_step (int, optional): The current training step for the decision of applying scaling strategy. Defaults to 0.
            target_len (torch.Tensor, optional): The ground truth lengths of the corresponding text captions. Defaults to None.

        Returns:
            dict: {
                "dsample_feats_pad_mask" (torch.Tensor): key padding mask of downsampled features
                "dsample_feats" (torch.Tensor): downsampled features,
                "dsample_feats_length" (torch.Tensor): lengths of downsampled features,
                "alpha" (torch.Tensor): alpha weights in the CIF module,
                "fired_marks" (torch.Tensor): during the accumulation of alpha, which indices fire (aggregate) features. 1 (True) indicates firing.
                "target_len" (torch.Tensor): the ground truth lengths of the corresponding text captions.
                "dsample_len_diff" (torch.LongTensor): length difference between ground truth and the prediction
            }
        """
        inputDict = {
            "audio_feat": audio_feat,
            "audio_feat_len": audio_feat_len,
            "audio_feat_pad_mask": audio_feat_pad_mask,
            "global_step": global_step,
        }

        if not self.training:
            input_target_len = (
                None  # We don't provide the downsampling target length during inference
            )
        else:
            if target_len is None:
                input_target_len = (audio_feat_len / 20).round().long()
            else:
                input_target_len = target_len

        dsample_results = self.downsampling(inputDict, input_target_len)
        if target_len is not None:
            dsample_results["target_len"] = target_len
            dsample_results["dsample_len_diff"] = (
                (dsample_results["dsample_feats_length"] - target_len)
                .abs()
                .float()
                .mean()
            )

        return dsample_results

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.Tensor,
        otherInputs: dict = {},
    ) -> dict:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_feat_len (torch.Tensor)
            otherInputs (dict, optional)

        Returns:
            dict
        """
        output = defaultdict(lambda: None)
        bsz, audio_max_len = audio_feat.shape[:2]
        device = audio_feat.device
        audio_feat_pad_mask = get_keypadding_mask(audio_max_len, audio_feat_len).to(
            device
        )

        # Extracting audio features from self-attention layers
        if hasattr(self, "self_att"):
            audio_feat = self.self_att(
                src=audio_feat, key_padding_mask=audio_feat_pad_mask
            )

        # Downsampling audio features by CIF
        dsample_results = self.downsampling_audio_feat(
            audio_feat=audio_feat,
            audio_feat_len=audio_feat_len,
            audio_feat_pad_mask=audio_feat_pad_mask,
            target_len=(
                otherInputs["target_len"] if "target_len" in otherInputs else None
            ),
            global_step=(
                otherInputs["global_step"] if "global_step" in otherInputs else 0
            ),
        )
        output["dsample_results"] = dsample_results
        audio_feat = dsample_results["dsample_feats"]

        # Feed keyword into clip text encoder
        vq_results, keywords = self.vq_audio_features(audio_feat)
        output["vq_results"] = vq_results
        output["keywords"] = keywords
        output["cascaded_audio_feat"] = self.clip.encode_keywords(
            keywords, dsample_results["dsample_feats_length"]
        )

        return output

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, max_audio_length = audio_feat.shape[:2]
        key_padding_mask = get_keypadding_mask(
            max_length=max_audio_length, data_lens=audio_len
        )
        if hasattr(self, "self_att"):
            hidden_states = self.self_att.extract_hidden_states(
                src=audio_feat, key_padding_mask=key_padding_mask
            )
            return tuple(hidden_states)
        else:
            return ()


class KW_HybridBranchPlus(KW_CascadedBranchPlus):
    """KW_HybridBranchPlus
    Hybrid Branch plus for SpeechCLIP+
    """

    def __init__(
        self, config, audio_dim: int, text_dim: int, out_dim: int, clip: ClipModel
    ) -> None:
        """init

        Args:
            config (OrderedNamespace): config of the model
            audio_dim (int): dimension for audio features
            text_dim (int): dimension for subword embeddings
            out_dim (int): dimension for the projected parallel CLS
            clip (ClipModel): the CLIP model
        """
        super().__init__(config, audio_dim, text_dim, clip)
        self.out_dim = out_dim
        self.cls = self._create_cls(
            length=1,
            cls_dim=self.config.model_settings.cascaded_branch.transformer_args.d_model,
        )  # Creating CLS token for the parallel barnch
        self._create_self_attn_layer(
            self.config.model_settings.cascaded_branch
        )  # select the main structure for transformer encoder layer
        self.parallel_proj = nn.Linear(self.audio_dim, self.out_dim)

    def forward(
        self,
        audio_feat: torch.Tensor,
        audio_feat_len: torch.Tensor,
        otherInputs: dict = {},
    ) -> dict:
        """forward

        Args:
            audio_feat (torch.Tensor)
            audio_feat_len (torch.Tensor)
            otherInputs (dict, optional)

        Returns:
            dict
        """
        output = defaultdict(lambda: None)
        bsz, audio_max_len = audio_feat.shape[:2]
        audio_feat_pad_mask = get_keypadding_mask(
            audio_max_len + 1, audio_feat_len + 1
        ).to(audio_feat.device)

        # Extract features from the self-attention layer by CLS token
        cls = torch.cat([self.cls] * bsz, dim=0)
        audio_feat = torch.cat([cls, audio_feat], dim=1)
        post_audio_feat = self.self_att(
            src=audio_feat, key_padding_mask=audio_feat_pad_mask
        )
        output["parallel_audio_feat"] = self.parallel_proj(
            post_audio_feat[:, :1].reshape(-1, self.audio_dim)
        )
        post_audio_feat = post_audio_feat[:, 1:].reshape(
            -1, audio_max_len, self.audio_dim
        )

        # Downsampling audio features by CIF
        dsample_results = self.downsampling_audio_feat(
            audio_feat=post_audio_feat,
            audio_feat_len=audio_feat_len,
            audio_feat_pad_mask=audio_feat_pad_mask[:, 1:],
            target_len=(
                otherInputs["target_len"] if "target_len" in otherInputs else None
            ),
            global_step=(
                otherInputs["global_step"] if "global_step" in otherInputs else 0
            ),
        )
        output["dsample_results"] = dsample_results
        dsample_feats = dsample_results["dsample_feats"]

        # Feed keyword into clip text encoder
        vq_results, keywords = self.vq_audio_features(dsample_feats)
        output["vq_results"] = vq_results
        output["keywords"] = keywords
        output["cascaded_audio_feat"] = self.clip.encode_keywords(
            keywords, dsample_results["dsample_feats_length"]
        )

        return output

    def extract_hidden_states(
        self, audio_feat: torch.Tensor, audio_len: torch.Tensor
    ) -> Tuple:
        """extract_hidden_states
        Extract hiddenstates of parallel branch
        Args:
            audio_feat (torch.Tensor):
            audio_len (torch.Tensor):

        Returns:
            Tuple: hidden representation of each layers
        """
        bsz, max_audio_length = audio_feat.shape[:2]
        cls = torch.cat([self.cls] * bsz, dim=0)
        key_padding_mask = get_keypadding_mask(
            max_length=max_audio_length + 1, data_lens=audio_len + 1
        )
        src = torch.cat([cls, audio_feat], dim=1)
        hidden_states = self.self_att.extract_hidden_states(
            src=src, key_padding_mask=key_padding_mask
        )
        hidden_states = [x[:, 1:, ...] for x in hidden_states]

        return tuple(hidden_states)
