import logging

logger = logging.getLogger(__name__)

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

MAX_FEAT_LEN = 75


def get_keypadding_mask(max_length: int, data_lens: torch.Tensor) -> torch.Tensor:
    bsz = data_lens.shape[0]
    key_padding_mask = torch.ones([bsz, max_length])
    for mask, len in zip(key_padding_mask, data_lens):
        mask[:len] = 0.0
    key_padding_mask = key_padding_mask.type_as(data_lens).bool()

    return key_padding_mask


class CIF(nn.Module):
    def __init__(
        self,
        cif_threshold=1.0,
        cif_output_dim=768,
        encoder_embed_dim=768,
        produce_weight_type="conv",
        num_layer=1,
        conv_cif_width=3,
        conv_cif_dropout=0.1,
        apply_scaling=True,
        apply_tail_handling=True,
        tail_handling_firing_threshold=0.5,
        scaling_step=-1,
        **config,
    ):
        super().__init__()

        # Load configurations
        self.cif_threshold = cif_threshold
        self.cif_output_dim = cif_output_dim
        self.encoder_embed_dim = encoder_embed_dim
        self.produce_weight_type = produce_weight_type
        self.conv_cif_width = conv_cif_width
        self.conv_cif_dropout = conv_cif_dropout
        self.apply_scaling = apply_scaling
        self.apply_tail_handling = apply_tail_handling
        self.tail_handling_firing_threshold = tail_handling_firing_threshold
        self.scaling_step = scaling_step
        self.num_layer = num_layer

        if self.apply_scaling:
            logger.info(f"Apply scaling strategy step: {self.scaling_step}")

        # Build weight generator
        if self.produce_weight_type == "dense":
            self.dense_proj = nn.Sequential(
                nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim), nn.ReLU()
            )
        elif self.produce_weight_type == "conv":
            conv_list = []
            for _ in range(self.num_layer):
                conv_list += [
                    nn.Conv1d(
                        self.encoder_embed_dim,
                        self.encoder_embed_dim,
                        self.conv_cif_width,
                        stride=1,
                        padding=int(self.conv_cif_width / 2),
                        dilation=1,
                        groups=1,
                        padding_mode="zeros",
                    ),
                    nn.Dropout(),
                    nn.ReLU(),
                ]
            self.conv = nn.Sequential(*conv_list)
        else:
            raise NotImplementedError(self.produce_weight_type)

        self.weight_proj = nn.Sequential(
            nn.Dropout(), nn.Linear(self.encoder_embed_dim, 1), nn.Sigmoid()
        )

        # Build the final projection layer (if encoder_embed_dim is not equal to cif_output_dim)
        if self.cif_output_dim != self.encoder_embed_dim:
            logger.info(
                f"Built projection layer to match the dimension of input {self.encoder_embed_dim} and output {self.cif_output_dim}"
            )
            self.cif_output_proj = nn.Linear(
                self.encoder_embed_dim, self.cif_output_dim, bias=False
            )

    def forward(self, input_dict, target_lengths=None, eps=1e-5):
        input_feats = input_dict["audio_feat"]  # B x T x D
        input_feats_pad_mask = input_dict["audio_feat_pad_mask"].bool()  # B x T
        original_length = (~input_feats_pad_mask).sum(-1).long()  # B

        if self.scaling_step >= 0:
            if self.apply_scaling and input_dict["global_step"] >= self.scaling_step:
                self.apply_scaling = False

        # Produce weights for integration (accumulation)
        if self.produce_weight_type == "dense":
            proj_out = self.dense_proj(input_feats)
            alpha = self.weight_proj(proj_out)  # B x T x 1
        elif self.produce_weight_type == "conv":
            conv_input = input_feats.permute(0, 2, 1)
            proj_input = self.conv(conv_input).permute(0, 2, 1)
            if hasattr(self, "conv_dropout"):
                logits = self.conv_dropout(proj_input)
            else:
                logits = proj_input
            alpha = (
                self.weight_proj(logits).clip(min=0.0, max=1.0).float().squeeze(-1)
            )  # B x T

            alpha[input_feats_pad_mask] = 0.0
            orig_alpha = alpha
            alpha_sum = alpha.sum(1)
            assert (alpha_sum > 0).any(), f"alphas are all zero:\n{alpha_sum}"

            # Apply scaling strategies
            if self.apply_scaling and target_lengths is not None:
                desired_sum = self.cif_threshold * target_lengths.type_as(alpha) + eps
                alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)

        # Integrate and fire
        result_dict = {
            "quantity_out": alpha_sum,
            "orig_alpha": orig_alpha,
            "original_length": original_length,
            "target_len": target_lengths,
        }
        dsmaple_dict = self.integrate_and_fire(
            input_feats,
            alpha,
            target_lengths=target_lengths,
        )
        dsmaple_dict["input_feats_pad_mask"] = input_feats_pad_mask

        result_dict = {**result_dict, **dsmaple_dict}

        if self.cif_output_dim != self.encoder_embed_dim:
            result_dict["dsample_feats"] = self.cif_output_proj(
                result_dict["dsample_feats"]
            )
            result_dict["dsample_feats"] = (
                result_dict["dsample_feats"] * result_dict["dsample_feats_pad_mask"]
            )

        return result_dict

    def integrate_and_fire(
        self,
        input: torch.Tensor,
        alpha: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """Integrate and fire the input feature sequence
        Author:
            https://github.com/George0828Zhang/torch_cif/blob/main/cif.py

        Args:
            input (torch.Tensor): input feature sequence
            alpha (torch.Tensor): prodcued alpha weights
            target_lengths (Optional[torch.Tensor], optional): the ground truth lengths of the corresponding text captions. Defaults to None.

        Returns:
            dict: {
                "dsample_feats_pad_mask" (torch.Tensor): key padding mask of downsampled features
                "dsample_feats" (torch.Tensor): downsampled features,
                "dsample_feats_length" (torch.Tensor): lengths of downsampled features,
                "alpha" (torch.Tensor): alpha weights in the CIF module,
                "fired_marks" (torch.Tensor): During the accumulation of alpha, which indices fire (aggregate) features. 1 (True) indicates firing.
            }
        """
        B, S, C = input.size()
        assert tuple(alpha.size()) == (B, S), f"{alpha.size()} != {(B, S)}"
        feat_lengths = (
            (alpha.sum(1) / self.cif_threshold)
            .floor()
            .clip(min=1, max=MAX_FEAT_LEN)
            .long()
        )
        T = feat_lengths.max()

        # aggregate and integrate
        csum = alpha.cumsum(-1)
        with torch.no_grad():
            # indices used for scattering
            right_idx = (csum / self.cif_threshold).floor().long().clip(min=0, max=T)
            left_idx = right_idx.roll(1, dims=1)
            left_idx[:, 0] = 0

            # count # of fires from each source
            fire_num = right_idx - left_idx
            extra_weights = (fire_num - 1).clip(min=0)

        # The extra entry in last dim is for tail handling
        output = input.new_zeros((B, T + 1, C))
        source_range = torch.arange(1, S + 1).unsqueeze(0).type_as(input)
        zero = alpha.new_zeros((1,))

        # right scatter
        fire_mask = fire_num > 0
        right_weight = torch.where(
            fire_mask, csum - right_idx.type_as(alpha) * self.cif_threshold, zero
        ).type_as(input)
        output.scatter_add_(
            1,
            right_idx.unsqueeze(-1).expand(-1, -1, C),
            right_weight.unsqueeze(-1) * input,
        )

        # left scatter
        left_weight = (
            alpha - right_weight - extra_weights.type_as(alpha) * self.cif_threshold
        ).type_as(input)
        output.scatter_add_(
            1,
            left_idx.unsqueeze(-1).expand(-1, -1, C),
            left_weight.unsqueeze(-1) * input,
        )

        # extra scatters
        if extra_weights.ge(0).any():
            extra_steps = extra_weights.max().item()
            tgt_idx = left_idx
            src_feats = input * self.cif_threshold
            for _ in range(extra_steps):
                tgt_idx = (tgt_idx + 1).clip(max=T)
                # (B, S, 1)
                src_mask = extra_weights > 0
                output.scatter_add_(
                    1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, C),
                    src_feats * src_mask.unsqueeze(2),
                )
                extra_weights = extra_weights - 1

        # tail handling
        if self.apply_tail_handling:
            if target_lengths is not None:
                # training time -> ignore tail
                output = output[:, :T, :]
            else:
                # find out contribution to output tail
                # note: w/o scaling, extra weight is all 0
                zero = right_weight.new_zeros((1,))
                r_mask = right_idx == feat_lengths.unsqueeze(1)
                tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
                l_mask = left_idx == feat_lengths.unsqueeze(1)
                tail_weights = tail_weights + torch.where(
                    l_mask, left_weight, zero
                ).sum(-1)

                # a size (B,) mask that extends position that passed threshold.
                extend_mask = tail_weights >= self.tail_handling_firing_threshold

                # extend 1 fire and upscale the weights
                if extend_mask.any():
                    # (B, T, C), may have infs so need the mask
                    upscale = (
                        torch.ones_like(output)
                        .scatter(
                            1,
                            feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                            (
                                self.cif_threshold
                                / tail_weights.masked_fill(
                                    ~extend_mask, self.cif_threshold
                                )
                                .view(B, 1, 1)
                                .expand(-1, -1, C)
                            ).to(output.dtype),
                        )
                        .detach()
                    )
                    output = output * upscale
                    feat_lengths = feat_lengths + extend_mask.long()
                    fire_mask[:, feat_lengths - 1] = (
                        fire_mask[:, feat_lengths - 1] + extend_mask
                    )
                    feat_lengths = feat_lengths.clip(max=MAX_FEAT_LEN)
                    T = feat_lengths.max()
                output = output[:, :T, :]

                # a size (B, T) mask to erase weights
                tail_mask = torch.arange(T, device=output.device).unsqueeze(
                    0
                ) >= feat_lengths.unsqueeze(1)
                output[tail_mask] = 0
        else:
            output = output[:, :T, :]

        result_dict = {
            "dsample_feats_pad_mask": get_keypadding_mask(
                output.shape[1], feat_lengths
            ),
            "dsample_feats": output,
            "dsample_feats_length": feat_lengths,
            "alpha": alpha,
            "fired_marks": fire_mask,
            # "tail_weights": [tail_weights] if target_lengths is None else []
        }

        return result_dict
