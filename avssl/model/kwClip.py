import json
import logging

logger = logging.getLogger(__name__)

import os
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F

from ..base import OrderedNamespace
from ..module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    MLPLayers,
    S3prlSpeechEncoderPlus,
    losses,
    mutualRetrieval,
)
from ..optim import get_scheduler
from ..util import (
    draw_embedding_space_PCA,
    extract_dynamic_keyword_neighbors,
    extract_fixed_keyword_neighbors,
)
from .base_model import BaseLightningModel
from .kw_branches import *

__all__ = [
    "KWClip_GeneralTransformer",
]

"""METRIC_REDUCEFN_MAPPING
define the reduction function for each data type when reducing from multiple GPUs

"""
METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}


class KWClipBase(BaseLightningModel):
    """Base Class for SpeechCLIP"""

    def __init__(self, config: OrderedNamespace):
        super().__init__(config)

        # select audio_encoder type
        self.audio_encoder_type = config.audio_encoder.type
        if self.audio_encoder_type == "s3prl":
            raise DeprecationWarning("Please use s3prl_plus")
            self.audio_encoder = S3prlSpeechEncoder(**config.audio_encoder)
        elif self.audio_encoder_type == "s3prl_plus":
            self.audio_encoder = S3prlSpeechEncoderPlus(**config.audio_encoder)
        elif self.audio_encoder_type == "FairseqHubert":
            self.audio_encoder = FairseqSpeechEncoder_Hubert(**config.audio_encoder)
        else:
            logger.warning("No audio encoder loaded")

        # define ClipModel
        self.clip = ClipModel(
            **config.clip,
        )

        if hasattr(self, "audio_encoder"):
            self.audio_embd_dim = self.audio_encoder.out_dim
        # dimension of the CLIP Text Encoder's subword embedding
        self.subword_embd_dim = self.clip.model.token_embedding.weight.size(-1)

        # the recall to calculate
        self.recall_at = config.retrieval.recall_at

        # define loss function
        self.criterion = getattr(losses, config.cl_loss.type)(**config.cl_loss.args)

        # whether or not to log detokenize subwords of keywords
        self.log_detokenize_results = config.log_setting.get(
            "log_detokenize_results", True
        )

        self.keyword_num = None

    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        return_hidden_states: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        """Get the representations of audio wav files after passing through the audio encoder

        Args:
            wav (Union[torch.Tensor, list]): wav files
            wav_len (Union[torch.Tensor, list], optional): lengths of each wavform. Defaults to [].
            return_hidden_states (bool, optional): return the hidden representations in the audio encoder. Defaults to False.

        Raises:
            NotImplementedError: if the audio encoder is not implemented in the code

        Returns:
            Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]: return the representations of waveforms (and also the hidden_states)
        """
        if self.audio_encoder_type in [
            "s3prl_plus",
            "FairseqHubert",
        ]:
            return self.audio_encoder(
                wav, wav_len, return_hidden_states=return_hidden_states
            )
        else:
            raise NotImplementedError("Unknown type:{}".format(self.audio_encoder_type))

    def forward(self, batch: dict) -> tuple:
        """the main forward function for our model (should be implemented in child class)

        Args:
            batch (dict): the input data in a batch

        Returns:
            tuple: return model output : (losses, log_metric, other_feats)
                losses: features required for calculating loss (pass into comput_loss)
                        if loss is calulated on each GPU individually, "loss" should exist in lossess
                log_metric: the calculated metric to log
                other_feats: other features required for validation
        """
        raise NotImplementedError()

    def compute_loss(self, input_feats):
        """compute the loss here

        Args:
            input_feats (Any): the feats required for computing loss (gathered from model forward output)
        """
        raise NotImplementedError()

    def training_step(self, batch: dict) -> dict:
        losses, log_metrics = self.forward(batch)[:2]
        return {"loss_feats": losses, "log_metrics": log_metrics}

    def training_step_end(self, outputs: dict) -> dict:
        """training_step_end

        Collect results from all GPUs

        Args:
            outputs (dict): output from trainin_step

        Raises:
            NotImplementedError: if the outputs' format collected from GPU(s) is not correct

        Returns:
            dict: loss (return to pytorch lightning for updating params)
        """
        if isinstance(outputs, dict):
            if "loss" in outputs:
                # training_step has already calculated the loss
                # we simply just average the loss on GPU(s)
                return {"loss": torch.mean(outputs["loss"])}
            elif "loss_feats" in outputs and "log_metrics" in outputs:
                losses = self.compute_loss(outputs["loss_feats"])
                log_metrics = outputs["log_metrics"]
                result = {
                    **{f"train_{k}": losses[k] for k in losses},
                    **{
                        f"train_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                            log_metrics[k]
                        )
                        for k in log_metrics
                    },
                }
                # log training loss(es) and metrics
                self.log_dict(
                    result,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )
                return {"loss": losses["loss"]}
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """validation_step

        Args:
            batch (dict): input data

        Returns:
            dict: output features
        """
        losses, log_metrics, others = self.forward(batch)

        audio_feat = (
            others["cascaded_audio_feat"]
            if self.config.retrieval.audio_feat_src == "cascaded"
            else others["parallel_audio_feat"]
        )

        image_feat = others.get("image_feat", None)
        text_feat = others.get("text_feat", None)
        id = others["id"]

        returnDict = {
            "id": id,
            "audio_feat": audio_feat,
        }

        if image_feat is not None:
            returnDict["image_feat"] = image_feat
        if text_feat is not None:
            returnDict["text_feat"] = text_feat

        if "keywords" in others and others["keywords"] is not None:
            kwDict = {"gold_text": batch["text"]}
            keywords = others["keywords"]
            if self.keyword_num is not None:
                kwDict["keywords"] = keywords
            else:
                # Dynamic number of keywords
                kwDict.update(
                    {
                        "keywords": keywords.view(-1, keywords.shape[-1]),
                        "keywords_bsz": keywords.shape[0],
                        "max_kw_num": keywords.shape[1],
                    }
                )

            returnDict.update(kwDict)

        if others["keywords_len"] is not None:
            returnDict["keywords_len"] = others["keywords_len"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": returnDict}

    def validation_step_end(self, outputs: dict) -> dict:
        """validation_step_end

        Collect features from all GPU(s) and calculate loss

        Args:
            outputs (dict): output from GPU(s)

        Returns:
            dict: features required for validation
        """

        assert isinstance(outputs, dict)
        losses = self.compute_loss(outputs["loss_feats"])

        log_metrics = outputs["log_metrics"]
        result = {
            **{f"val_{k}": losses[k] for k in losses},
            **{
                f"val_{k}": METRIC_REDUCEFN_MAPPING[type(log_metrics[k])](
                    log_metrics[k]
                )
                for k in log_metrics
            },
        }
        self.log_dict(
            result,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        for k in outputs["others"]:
            if isinstance(outputs["others"][k], torch.Tensor):
                outputs["others"][k] = outputs["others"][k].detach().cpu()
        return outputs["others"]

    def validation_epoch_end(self, outputs: dict) -> dict:
        """validation_epoch_end

        Args:
            outputs (list): list of aggregated results
        """

        RootDir = self.config.trainer.default_root_dir
        if "keywords" in outputs[0].keys():
            if not os.path.exists(os.path.join(RootDir, "retokenizeText")):
                os.makedirs(os.path.join(RootDir, "retokenizeText"), exist_ok=True)
            if not os.path.exists(os.path.join(RootDir, "visualization")):
                os.makedirs(
                    os.path.join(RootDir, "visualization"),
                    exist_ok=True,
                )

            if (
                hasattr(self, "log_detokenize_results_every_n_epoch")
                and self.current_epoch % self.log_detokenize_results_every_n_epoch == 0
            ):

                all_keyword_embeddings = torch.cat(
                    [x["keywords"] for x in outputs], dim=0
                )
                embeddings_stat_dict = defaultdict(dict)
                if self.keyword_num is not None:
                    all_keyword_embeddings = all_keyword_embeddings.view(
                        all_keyword_embeddings.shape[0],
                        self.keyword_num,
                        all_keyword_embeddings.shape[-1],
                    )
                    kwNumRange = self.keyword_num
                else:
                    all_keyword_embeddings = all_keyword_embeddings.unsqueeze(1)
                    kwNumRange = 1

                for i in range(kwNumRange):
                    kw_index = "kw" if self.keyword_num is None else f"kw_{i}"
                    embeddings_stat_dict["mean"][kw_index] = (
                        all_keyword_embeddings[:, i, :].mean(0).mean()
                    )
                    embeddings_stat_dict["std"][kw_index] = (
                        all_keyword_embeddings[:, i, :].std(0).mean()
                    )
                    embeddings_stat_dict["norm"][kw_index] = (
                        all_keyword_embeddings[:, i, :].norm(p=2, dim=-1).mean()
                    )

                if self.keyword_num is None:
                    all_keyword_embeddings.squeeze(1)

                tokenEmbeddings = self.clip.model.token_embedding.weight.detach().cpu()
                self.log(
                    "kw_mean_mse",
                    torch.norm(
                        all_keyword_embeddings.view(-1, self.subword_embd_dim).mean(0),
                        tokenEmbeddings.mean(0),
                        p=2,
                    ),
                    sync_dist=True,
                )
                self.log(
                    "kw_std_mse",
                    torch.std(
                        torch.norm(
                            all_keyword_embeddings.view(-1, self.subword_embd_dim).std(
                                0
                            ),
                            tokenEmbeddings.std(0),
                            p=2,
                        )
                    ),
                )

                # Drawing PCA
                self.log_draw_pca_every_n_epoch = getattr(
                    self.config.log_setting, "log_draw_pca_every_n_epoch", 0
                )
                if self.log_draw_pca_every_n_epoch > 0 and (
                    self.current_epoch % self.log_draw_pca_every_n_epoch == 0
                ):
                    draw_embedding_space_PCA(
                        kw_embs=all_keyword_embeddings,
                        gold_embs=tokenEmbeddings,
                        output_path=os.path.join(
                            RootDir,
                            "visualization/",
                            "pca_ep{}.pdf".format(self.current_epoch),
                        ),
                    )

                # collect glden texts
                gold_texts = []
                keywordEmbeddings_list = []
                kwEmbedLengths = []
                for x in outputs:
                    for sent in x["gold_text"]:
                        gold_texts.append(
                            self.clip.tokenizer.decode(sent.squeeze().tolist())
                        )
                    if self.keyword_num is None:
                        kwEmbedLengths += x["keywords_len"].tolist()
                        embdList = torch.split(
                            x["keywords"],
                            (x["keywords_bsz"] * x["max_kw_num"]).tolist(),
                            dim=0,
                        )
                        keywordEmbeddings_list.append(
                            [
                                embd.view(bsz, knum, -1)
                                for bsz, knum, embd in zip(
                                    x["keywords_bsz"], x["max_kw_num"], embdList
                                )
                            ]
                        )

                # get retrieval method : either via cosine or pseudo inverse (default cosine similarity)
                retrieve_method = getattr(
                    self.config.model_settings.cascaded_branch.keyword,
                    "retrieve_method",
                    "cosine",
                )
                if retrieve_method not in ["cosine", "pseudo_inverse"]:
                    raise NotImplementedError(retrieve_method)

                # list the semantically related subwords to the selected keyword
                K = self.config.model_settings.cascaded_branch.keyword.get(
                    "detokenized_K_neighbors", 10
                )
                print("Detokenizing K={}".format((K)))
                TextDir = os.path.join(RootDir, "retokenizeText/")
                if self.keyword_num is not None:
                    all_retok_outputs = extract_fixed_keyword_neighbors(
                        model=self,
                        K=K,
                        retrieve_method=retrieve_method,
                        tokenEmbeddings=tokenEmbeddings,
                        all_keyword_embeddings=all_keyword_embeddings,
                        gold_texts=gold_texts,
                    )
                else:
                    all_retok_outputs = extract_dynamic_keyword_neighbors(
                        model=self,
                        K=K,
                        retrieve_method=retrieve_method,
                        outputs=outputs,
                        tokenEmbeddings=tokenEmbeddings,
                        keywordEmbeddings_list=keywordEmbeddings_list,
                        gold_texts=gold_texts,
                        kwEmbedLengths=kwEmbedLengths,
                    )

                with open(
                    os.path.join(TextDir, f"keywords_ep{self.current_epoch}.json"), "w"
                ) as f_kw:
                    json.dump(all_retok_outputs, f_kw, indent=4)

                del all_retok_outputs

        # Retreiving images and audios
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        all_imgs = torch.cat([x["image_feat"] for x in outputs], dim=0)
        id_img_pairs = {_id.item(): _img for _id, _img in zip(all_ids, all_imgs)}

        del all_imgs

        all_audo_feats = torch.cat([x["audio_feat"] for x in outputs], dim=0)
        all_audo_feats_id = all_ids

        all_img_feats = torch.stack([x for _, x in id_img_pairs.items()], dim=0)
        all_img_feats_id = torch.LongTensor(list(id_img_pairs.keys()))

        print(
            "Total #{} images, #{} audio".format(
                len(all_img_feats), len(all_audo_feats)
            )
        )

        # calculate dot product
        score_per_audio = torch.matmul(
            all_audo_feats.float().to(self.device),
            all_img_feats.float().T.to(self.device),
        )
        score_per_image = score_per_audio.T

        # AI : Audio -> Image, IA: Image -> Audio
        AI_answers = all_audo_feats_id
        IA_answers = all_img_feats_id

        self.reportRetrieval(
            score_per_A=score_per_audio,
            score_per_B=score_per_image,
            AB_answers=AI_answers,
            BA_answers=IA_answers,
        )

    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        """forward_image

        Args:
            images (Union[list, torch.Tensor]): image input

        Raises:
            ValueError: image tensor shape error
            TypeError: image type should be either list or torch.Tensor

        Returns:
            torch.Tensor: image representations
        """
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        image_feat = self.clip.encode_image(image_tensor)
        return image_feat

    def forward_text(self, sents: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(sents, list):
            text_tensor = self.clip.prep_text(sents).to(self.device)
        elif isinstance(sents, torch.Tensor):
            if sents.dim() != 2:
                raise ValueError(f"Incorrect text tensor shape {sents.shape}")
            text_tensor = sents
        else:
            raise TypeError(f"Unknown text type {type(sents)}")
        if hasattr(self.clip, "original2Reduced"):
            # if reduced embedding is used, we need to convert original ids to reduced ids
            for i in range(text_tensor.shape[0]):
                for j in range(text_tensor.shape[1]):
                    text_tensor[i, j] = self.clip.original2Reduced[
                        text_tensor[i, j].item()
                    ]

        text_feat = self.clip.encode_text(text_tensor)
        return text_feat

    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        """reportRetrieval

        Args:
            score_per_A (torch.Tensor): the similarity score per modality A sample
            score_per_B (torch.Tensor): the similarity score per modality B sample
            AB_answers (torch.Tensor): the golden answer (pair ID) for each audio sample
            BA_answers (torch.Tensor): the golden answer (pair ID) for each image sample
            metadata (dict): metadata should include modality the title for A, B and the abbreviation for A and B
        """

        # metadata should include modality the title for A, B and the abbreviation for A and B
        assert "modality_A_title" in metadata
        assert "modality_B_title" in metadata
        assert "modality_A_logAbbr" in metadata
        assert "modality_B_logAbbr" in metadata

        recall_results_AB, recall_results_BA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            recall_at=self.recall_at,
            modality_A_title=metadata["modality_A_title"],
            modality_B_title=metadata["modality_B_title"],
        )

        log_AB_abbr = "{}{}".format(
            metadata["modality_A_logAbbr"], metadata["modality_B_logAbbr"]
        )
        log_BA_abbr = "{}{}".format(
            metadata["modality_B_logAbbr"], metadata["modality_A_logAbbr"]
        )

        print(f"val_recall_{log_AB_abbr}", recall_results_AB)
        print(f"val_recall_{log_BA_abbr}", recall_results_BA)
        print("val_recall_mean", recall_results_mean)

        if isinstance(self.logger, WandbLogger):
            # when using wandb
            self.log(f"val_recall_{log_AB_abbr}", recall_results_AB, sync_dist=True)
            self.log(f"val_recall_{log_BA_abbr}", recall_results_BA, sync_dist=True)
            self.log("val_recall_mean", recall_results_mean, sync_dist=True)
        elif isinstance(self.logger, TensorBoardLogger):
            # when using tensorboard
            self.logger.experiment.add_scalars(
                f"val_recall_{log_AB_abbr}", recall_results_AB, self.global_step
            )
            self.logger.experiment.add_scalars(
                f"val_recall_{log_BA_abbr}", recall_results_BA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_mean", recall_results_mean, self.global_step
            )
        if self.logger is not None:
            self.log(
                "val_recall_mean_10", recall_results_mean["recall@10"], sync_dist=True
            )

    def processWavs(
        self, wav: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """processWavs

        Args:
            wav (torch.LongTensor): wav input

        Returns:
            Tuple[torch.Tensor,torch.LongTensor]: wavs, wav_lens
        """

        wav_len = [len(x) for x in wav]
        if isinstance(wav, torch.Tensor):
            #wav_len = torch.LongTensor(wav_len, device=wav.device)
            wav_len = torch.LongTensor(wav_len).to(wav.device)
        return wav, wav_len

    def feature_extractor_s3prl(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ) -> torch.Tensor:
        """feature_extractor_s3prl
        Implement for s3prl to get feature
        Args:
            wav ():
        """
        raise NotImplementedError()

    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []

        if hasattr(self, "audio_encoder"):
            my_params += self.audio_encoder.trainable_params()
            my_params += list(self.criterion.parameters())

        my_params += self.clip.trainable_params()

        return my_params

    def configure_optimizers(self) -> Tuple[list, list]:
        """configure_optimizers

        Returns:
            Tuple[list,list]: (optimizer_list,scheduler_list)
        """
        optimizers = []
        schedulers = []

        my_params = self.getTrainableParams()

        audio_optimizer = getattr(torch.optim, self.config.audio_encoder.optim.name)(
            my_params,
            **self.config.audio_encoder.optim.args,
        )
        audio_scheduler = get_scheduler(
            optimizer=audio_optimizer,
            **self.config.audio_encoder.scheduler,
        )

        optimizers.append(audio_optimizer)
        schedulers.append(
            {
                "scheduler": audio_scheduler,
                "interval": "step",
            }
        )

        return optimizers, schedulers


class KWClip_GeneralTransformer(KWClipBase):
    def __init__(self, config: OrderedNamespace):
        super().__init__(config)     
        self.cascaded_branch = None
        self.parallel_branch = None

        if self.config.model_settings.cascaded_objective_weight > 0:
            cBranchType = self.config.model_settings.cascaded_branch.type
            # Unify model name aliases
            cBranchType = self.config.model_settings.cascaded_branch.type.replace(
                "KW_", ""
            )
            cBranchType = cBranchType.replace("dynamic", "plus")
            if cBranchType == "CascadedBranch":
                logger.info("Create Cascaded Branch")
                # cascaded_branch
                self.cascaded_branch = KW_CascadedBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif cBranchType == "CascadedBranch_plus":
                logger.info("Create Cascaded Branch Plus")
                # cascaded_branch_plus
                self.cascaded_branch = KW_CascadedBranchPlus(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif cBranchType == "HybridBranch":
                assert (
                    self.config.model_settings.parallel_objective_weight > 0
                ), self.config.model_settings.parallel_objective_weight
                logger.info("Create Hybrid Branch")
                # hybrid_branch
                self.cascaded_branch = KW_HybridBranch(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            elif cBranchType == "HybridBranch_plus":
                assert (
                    self.config.model_settings.parallel_objective_weight > 0
                ), self.config.model_settings.parallel_objective_weight
                logger.info("Create Hybrid Branch Plus")
                # hybrid_branch_plus
                self.cascaded_branch = KW_HybridBranchPlus(
                    config=self.config,
                    audio_dim=self.audio_embd_dim,
                    text_dim=self.subword_embd_dim,
                    out_dim=self.subword_embd_dim,
                    clip=self.clip,
                )
            else:
                raise NotImplementedError(cBranchType)

            # Keyword number
            if hasattr(self.cascaded_branch, "keyword_num"):
                self.keyword_num = self.cascaded_branch.keyword_num

            # CIF's quantity loss
            if (
                hasattr(self.config.model_settings.cascaded_branch, "downsampling")
                and self.config.model_settings.cascaded_branch.downsampling.type
                == "cif"
            ):
                self.quantity_loss_weight = getattr(
                    self.config.model_settings.cascaded_branch.downsampling.cif,
                    "quantity_loss_weight",
                    1.0,
                )
                self.quantity_loss_criteria = nn.L1Loss()

        if (
            self.config.model_settings.parallel_objective_weight > 0
            and self.cascaded_branch is None
        ):
            logger.info("Create Parallel Branch")
            # Parallel branch
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                text_dim=self.subword_embd_dim,
            )

        # projection network after CLIP image encoder
        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )

        # projection network after parallel branch
        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )

        # projection network after cascaded branch
        self.c_branch_proj_net = None
        cascaded_branch_projection = self.config.model_settings.get(
            "cascaded_branch_projection", None
        )
        if cascaded_branch_projection is not None:
            logger.info(
                f"cascaded_branch_projection dims:{cascaded_branch_projection.dimensions} droupout:{cascaded_branch_projection.dropout}"
            )
            self.c_branch_proj_net = MLPLayers(
                units=cascaded_branch_projection.dimensions,
                dropout=cascaded_branch_projection.dropout,
            )

    def getTrainableParams(self) -> list:
        """getTrainableParams

        Returns:
            list: list of trainable params in this class
        """
        _params = super().getTrainableParams()

        if self.cascaded_branch is not None:
            logger.info("Add cascaded_branch parameters")
            _params += list(self.cascaded_branch.parameters())

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        if self.c_branch_proj_net is not None:
            logger.info("Add cascaded_branch_projection parameters")
            _params += list(self.c_branch_proj_net.parameters())

        return _params

    def forward(
        self,
        batch: dict,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        id = batch["id"]

        # update device information to clip model
        self.clip.update_device(self.device)
        audio_feat, audio_feat_len = self.forward_audio(
            wav, wav_len, return_hidden_states=False
        )
        image_feat = self.forward_image(image)
        if self.img_enc_proj_net is not None:
            image_feat = self.img_enc_proj_net(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        if self.cascaded_branch is not None:
            # Collecting inputs for the cascaded and hybrid plus branches
            if isinstance(self.cascaded_branch, KW_CascadedBranchPlus):
                otherInputs = {"global_step": self.global_step}
                if getattr(self.cascaded_branch, "using_gt_len", False):
                    assert (
                        "text" in batch
                    ), f"Text captions are required, {batch.keys()}"
                    target_len = torch.LongTensor(
                        [(t.squeeze().tolist().index(49407) - 1) for t in batch["text"]]
                    ).to(wav.device)
                else:
                    target_len = (audio_feat_len / 20).round().long()
                otherInputs["target_len"] = target_len
            else:
                otherInputs = None
            output = self.cascaded_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
                otherInputs=otherInputs,
            )
        if self.parallel_branch is not None:
            output = self.parallel_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )

        # Extract outputs
        parallel_audio_feat = output["parallel_audio_feat"]
        cascaded_audio_feat = output["cascaded_audio_feat"]
        vq_results = output["vq_results"]
        keywords = output["keywords"]
        dsample_results = output["dsample_results"]
        if dsample_results is not None:
            keywords_len = dsample_results["dsample_feats_length"]
        else:
            keywords_len = None

        # Extract losses
        losses = {
            "id": id,
            "image_feat": image_feat,
        }
        if cascaded_audio_feat is not None:
            if self.c_branch_proj_net is not None:
                cascaded_audio_feat = self.c_branch_proj_net(cascaded_audio_feat)
            cascaded_audio_feat = cascaded_audio_feat / cascaded_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["cascaded_audio_feat"] = cascaded_audio_feat

        if parallel_audio_feat is not None:
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)
            parallel_audio_feat = parallel_audio_feat / parallel_audio_feat.norm(
                dim=-1, keepdim=True
            )
            losses["parallel_audio_feat"] = parallel_audio_feat

        if (
            self.cascaded_branch is not None
            and hasattr(self.cascaded_branch, "downsampling_type")
            and self.cascaded_branch.downsampling_type == "cif"
        ):
            assert (
                "target_len" in dsample_results and "quantity_out" in dsample_results
            ), f"{dsample_results.keys()}"
            losses["cif_quantity_out"] = dsample_results["quantity_out"]
            losses["cif_target_len"] = dsample_results["target_len"]

        # Logging metrics
        log_metrics = {"cl_temp": self.criterion.current_temperature}

        if vq_results is not None:
            log_metrics["softmax_temp"] = vq_results["temp"]

        if self.cascaded_branch is not None:
            if dsample_results is not None and "dsample_len_diff" in dsample_results:
                log_metrics["dsample_len_diff"] = dsample_results["dsample_len_diff"]
            vq_results_log_keys = [
                "temp",
                "code_perplexity",
                "prob_perplexity",
                "ent_per_t",
            ]
            assert set(vq_results_log_keys).issubset(
                set(vq_results.keys())
            ), f"log keys: {vq_results_log_keys}, result: {vq_results.keys()}"
            vq_log_dict = {k: vq_results[k] for k in vq_results_log_keys}
            log_metrics.update(vq_log_dict)

        return (
            losses,
            log_metrics,
            {
                "id": id,
                "image_feat": image_feat,
                "parallel_audio_feat": parallel_audio_feat,
                "cascaded_audio_feat": cascaded_audio_feat,
                "vq_results": vq_results,
                "keywords": keywords,
                "dsample_results": dsample_results,
                "keywords_len": keywords_len,
            },
        )

    def feature_extractor_s3prl(self, wav) -> Tuple[torch.Tensor, Tuple]:
        """feature_extractor_s3prl

        Args:
            wav (list): list of wavforms

        Returns:
            Tuple: (output_embeddings, tuples of all hidden states)
        """

        wav, wav_len = self.processWavs(wav)

        audio_feat, audio_len, hidden_states = self.forward_audio(
            wav, wav_len, return_hidden_states=True
        )
        assert isinstance(hidden_states, tuple)

        cascaded_hidden_states = None
        parallel_hidden_states = None
        if self.cascaded_branch is not None:
            cascaded_hidden_states = self.cascaded_branch.extract_hidden_states(
                audio_feat, audio_len
            )
            assert isinstance(cascaded_hidden_states, tuple)
            hidden_states = hidden_states + tuple(cascaded_hidden_states[1:])
        if self.parallel_branch is not None:
            parallel_hidden_states = self.parallel_branch.extract_hidden_states(
                audio_feat, audio_len
            )
            assert isinstance(parallel_hidden_states, tuple)
            hidden_states = hidden_states + tuple(parallel_hidden_states[1:])

        return hidden_states[-1], hidden_states

    def compute_loss(self, inputDict: dict):
        """compute the loss here

        Args:
            inputDict (dict): the features required for computing loss
        """
        assert isinstance(inputDict, dict)
        required_keys = {"id", "image_feat"}
        assert required_keys.issubset(
            set(inputDict.keys())
        ), f"required: {required_keys}, input: {inputDict.keys()}"

        losses = {"loss": 0}
        image_feat = inputDict["image_feat"].float()
        id = inputDict["id"]

        branchTypeList = ["cascaded", "parallel"]
        for branchType in branchTypeList:
            loss_weight = getattr(
                self.config.model_settings, f"{branchType}_objective_weight", 0.0
            )
            if loss_weight > 0.0:
                feats_key = f"{branchType}_audio_feat"
                assert feats_key in inputDict, f"{inputDict.keys()}"
                losses[f"{branchType[0]}_cl_loss"] = self.criterion(
                    feat_A=inputDict[feats_key].float(),
                    feat_B=image_feat,
                    index=id,
                )
                losses["loss"] += loss_weight * losses[f"{branchType[0]}_cl_loss"]

        if (
            "cif_quantity_out" in inputDict
            and "cif_target_len" in inputDict
            and hasattr(self, "quantity_loss_criteria")
        ):
            losses["quantity_loss"] = self.quantity_loss_criteria(
                inputDict["cif_quantity_out"], inputDict["cif_target_len"]
            )
            losses["loss"] += self.quantity_loss_weight * losses["quantity_loss"]

        return losses

    def encode_speech(
        self,
        wav,
    ) -> dict:
        """encode speech

        Args:
            wav (list): input list of waveforms

        Returns:
            dict: {
                "cascaded_audio_feat" : not None if cascaded branch exists
                "parallel_audio_feat" : not None if parallel branch exists
                "vq_results"          : not None if cascaded branch exists
                "keywords"            : not None if cascaded branch exists
            }
        """

        wav, wav_len = self.processWavs(wav)
        device = torch.device("cuda")
        wav = wav.to(device)
        audio_feat, audio_feat_len = self.forward_audio(wav, wav_len)

        if self.cascaded_branch is not None:
            if self.cascaded_branch.clip.device != audio_feat.device:
                self.cascaded_branch.clip = self.cascaded_branch.clip.to(
                    audio_feat.device
                )
            output = self.cascaded_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )
        if self.parallel_branch is not None:
            output = self.parallel_branch(
                audio_feat=audio_feat,
                audio_feat_len=audio_feat_len,
            )
            if self.p_branch_proj_net is not None:
                parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)

        # Extract outputs
        parallel_audio_feat = output["parallel_audio_feat"]
        cascaded_audio_feat = output["cascaded_audio_feat"]
        vq_results = output["vq_results"]
        keywords = output["keywords"]

        return {
            "cascaded_audio_feat": cascaded_audio_feat,
            "parallel_audio_feat": parallel_audio_feat,
            "vq_results": vq_results,
            "keywords": keywords,
        }

    def extract_keywords(self, wav):
        audio_feat, audio_feat_len = self.forward_audio(wav, [len(wav)])

        _, _, vq_results, _, dsample_results = self.cascaded_branch(
            audio_feat, audio_feat_len, {}
        )
        vq_results["targets"] = vq_results["targets"].flatten()
        vq_results["targets"] = [
            self.clip.reducedl2Original[tok.item()] for tok in vq_results["targets"]
        ]
        return {"vq_results": vq_results, "dsample_results": dsample_results}
