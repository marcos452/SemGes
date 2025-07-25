from collections import defaultdict
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

__all__ = [
    "freeze_model",
    "unfreeze_model",
    "extract_fixed_keyword_neighbors",
    "extract_dynamic_keyword_neighbors",
]


class SpeechCLIPDecoder:
    """Decoder of SpeechCLIP models"""

    def __init__(self, decoder: dict, indexMapping=None) -> None:
        self.decoder = decoder
        self.indexMapping = indexMapping  # Mapping of the reduced tokens' ids to the original tokens' ids

    def decode(self, tokenId) -> str:
        if self.indexMapping is not None:
            return self.decoder[self.indexMapping[tokenId]]
        else:
            return self.decoder[tokenId]


def freeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def unfreeze_model(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = True


def extract_fixed_keyword_neighbors(
    model,
    K: int,
    retrieve_method: str,
    tokenEmbeddings: torch.Tensor,
    keywordEmbeddings: torch.Tensor,
    gold_texts: list,
) -> List[dict]:
    """Extract K neighnors of fixed numbers of keywords

    Args:
        model: SpeechCLIP model
        K (int): number of neighbors to be extracted
        retrieve_method (str): Method of retreiving keywords, either be 'pseudo_inverse' or 'cosine'
        tokenEmbeddings (torch.Tensor): embeddings of CLIP's tokens
        keywordEmbeddings (torch.Tensor): embeddings of SpeechCLIP's output keyword
        gold_texts (list): List of the gold text captions corresponding to the input utterances

    Returns:
        List[dict]: Detokenized keywords' closest k neighbors of each utterance
    """
    all_retok_outputs = []
    batch_size = model.config.data.dev_batch_size
    decoder = SpeechCLIPDecoder(
        decoder=model.clip.tokenizer.decoder,
        indexMapping=(
            model.clip.reducedl2Original
            if model.clip.selected_text_emb_ids is not None
            else None
        ),
    )

    for i in tqdm(range(0, len(gold_texts) + batch_size, batch_size)):
        goldTextsBatch = gold_texts[i : i + batch_size]
        currBsz = len(
            goldTextsBatch
        )  # might be different from batch_size for the last batch
        if currBsz == 0:
            break
        if retrieve_method == "pseudo_inverse":
            emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()
            kw_retrevial_score = (
                emb_pinv.float()
                @ keywordEmbeddings[i : i + currBsz]
                .view(-1, model.subword_embd_dim)
                .float()
                .reshape(-1, model.subword_embd_dim)
                .permute(1, 0)
            ).permute(1, 0)
        else:
            kw_retrevial_score = F.cosine_similarity(
                keywordEmbeddings[i : i + currBsz].view(-1, model.subword_embd_dim, 1),
                tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                dim=1,
            )
        # Compute closest k CLIP's keywords tokens' distance and indices
        kVals, kInds = torch.topk(kw_retrevial_score, K)

        assert kVals.shape == (
            currBsz * model.keyword_num,
            K,
        ), kVals.shape
        kVals = kVals.view(currBsz, model.keyword_num, K)
        kInds = kInds.view(currBsz, model.keyword_num, K)

        for x in range(currBsz):
            neighbors = defaultdict(list)
            for kw_i in range(model.keyword_num):
                for idx, dist in zip(kInds[x, kw_i], kVals[x, kw_i]):
                    neighbors["keyword_{}".format(kw_i)].append(
                        [
                            decoder.decode(idx.item()),
                            dist.item(),
                        ]
                    )

            all_retok_outputs.append(
                {
                    "gold": gold_texts[x],
                    "neighbors": neighbors,
                }
            )

    return all_retok_outputs


def extract_dynamic_keyword_neighbors(
    model,
    K: int,
    retrieve_method: str,
    outputs: dict,
    tokenEmbeddings: torch.Tensor,
    keywordEmbeddings_list: List[List[torch.Tensor]],
    gold_texts: list,
    kwEmbedLengths: list,
) -> List[dict]:
    """Extract K neighnors of dynmaic numbers of keywords

    Args:
        model: SpeechCLIP model
        K (int): number of neighbors for each utterance to be extracted
        retrieve_method (str): Method of retreiving keywords, either be 'pseudo_inverse' or 'cosine'
        outputs (dict): outputs (list): list of aggregated results in the validation_epoch_end()
        tokenEmbeddings (torch.Tensor): embeddings of CLIP's tokens
        keywordEmbeddings_list (List[List[torch.Tensor]]): embeddings of SpeechCLIP's output keyword
        gold_texts (list): list of the gold text captions corresponding to the input utterances
        kwEmbedLengths (list): list which saves the lengths of the keyword embeddings

    Returns:
        List[dict]: detokenized keywords' closest k neighbors of each utterance
    """

    all_retok_outputs = []
    batch_size = model.config.data.dev_batch_size
    decoder = SpeechCLIPDecoder(
        decoder=model.clip.tokenizer.decoder,
        indexMapping=(
            model.clip.reducedl2Original
            if model.clip.selected_text_emb_ids is not None
            else None
        ),
    )
    for b_idx, i in tqdm(
        zip(
            range(len(outputs)),
            range(0, len(gold_texts), batch_size),
        )
    ):
        goldTextsBatch = gold_texts[
            i : i + batch_size
        ]  # A small batch of all gold texts
        kwEmbedLenBatch = kwEmbedLengths[
            i : i + batch_size
        ]  # A small batch of all legnths of the keyword embeddings
        idxInBatch = 0  # Indicate the index of batch samples we should start to process
        for keyword_embeddings in keywordEmbeddings_list[b_idx]:
            # bszGPU stands for the batch size on a single GPU.
            # If you are using multiple GPUs with Data Parallelism (DP) for training, then bszGPU might differ from batch_size.
            # For example, if batch_size = 8 and you use 3 GPUs with DP for training,
            # then bszGPU = 3 for GPU 0, bszGPU = 3 for GPU 1, and bszGPU = 2 for GPU 2.
            bszGPU, maxEmbedLen = keyword_embeddings.shape[:2]
            if retrieve_method == "pseudo_inverse":
                emb_pinv = torch.linalg.pinv(tokenEmbeddings.T).float()
                kw_retrevial_score = (
                    emb_pinv.float()
                    @ keyword_embeddings.view(-1, model.subword_embd_dim)
                    .float()
                    .reshape(-1, model.subword_embd_dim)
                    .permute(1, 0)
                ).permute(1, 0)
            else:
                kw_retrevial_score = F.cosine_similarity(
                    keyword_embeddings.view(-1, model.subword_embd_dim, 1),
                    tokenEmbeddings.transpose(0, 1).unsqueeze(0),
                    dim=1,
                )
            # Compute closest k CLIP's keywords tokens' distance and indices
            kVals, kInds = torch.topk(kw_retrevial_score, K)

            assert kVals.shape == (
                bszGPU * maxEmbedLen,
                K,
            ), kVals.shape
            kVals = kVals.view(bszGPU, maxEmbedLen, K)
            kInds = kInds.view(bszGPU, maxEmbedLen, K)

            # Decode the closest k CLIP's tokens
            for x in range(bszGPU):
                neighbors = defaultdict(list)
                kwEmbedLen = kwEmbedLenBatch[idxInBatch + x]
                for kw_i in range(kwEmbedLen):
                    for idx, dist in zip(kInds[x, kw_i], kVals[x, kw_i]):
                        neighbors["keyword_{}".format(kw_i)].append(
                            [
                                decoder.decode(idx.item()),
                                dist.item(),
                            ]
                        )
                all_retok_outputs.append(
                    {
                        "gold": goldTextsBatch[idxInBatch + x],
                        "neighbors": neighbors,
                    }
                )
            idxInBatch += bszGPU

    return all_retok_outputs
