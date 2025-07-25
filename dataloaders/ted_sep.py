import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
import pyarrow
import librosa
import smplx
import copy
import pickle
from .build_vocab_ted import Vocab
# from .build_vocab import Vocab
# from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        self.loader_type = loader_type
        self.lmdb_dir = self.args.cache_path
        self.n_poses = 34
        self.subdivision_stride = 10
        self.skeleton_resampling_fps = 15
        self.remove_word_timing = False

        self.expected_audio_length = int(round(self.n_poses / self.skeleton_resampling_fps * 16000))
        self.expected_spectrogram_length = self.calc_spectrogram_length_from_motion_length(
            self.n_poses, self.skeleton_resampling_fps)


        preloaded_dir = self.args.cache_path + self.loader_type + f"_cache_pickle"  
        # logging.info('Found the cache {}'.format(preloaded_dir))
        # self.speaker_model = None
        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']
        
        self.lang_model = None
        
        # make a speaker model

        # precomputed_model = self.args.cache_path + self.loader_type + '_speaker_model.pkl'
        # if not os.path.exists(precomputed_model):
        #     self.speaker_model = self._make_speaker_model(self.args.cache_path, precomputed_model)
        # else:
        #     with open(precomputed_model, 'rb') as f:
        #         self.speaker_model = pickle.load(f)
                       

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pickle.loads(sample)
            word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample
            # print("aux_info: ", aux_info.keys())



        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']

            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if self.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i+1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()
        vids = aux_info['vid']
        # print('vid max',max(self.speaker_model.word2index.values()))
        vid_indices = self.vid2index(vids)[vids]
        # print(vid_indices)
        vid_indices = np.array([vid_indices])

        # print(vid_indices)
        # vid_indices = torch.LongTensor(vid_indices).to(device)
        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = self.make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None
        
        # to tensors
        # word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        vec_seq = torch.from_numpy(copy.copy(vec_seq)).reshape((vec_seq.shape[0], -1)).float()
        pose_seq = torch.from_numpy(copy.copy(pose_seq)).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(copy.copy(audio)).float()
        vid_indices = torch.from_numpy(copy.copy(vid_indices)).float()
        
        return {"pose":pose_seq, "audio":audio, "word":extended_word_seq, "id":vid_indices}

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model
    
    def vid2index(self, vid):
        pickle_path = self.args.cache_path+"vids.pkl"

        with open(pickle_path, "rb") as f:
            unique_vids = pickle.load(f)

        # 创建 vid 到 index 的映射
        vid_to_index = {vid: idx for idx, vid in enumerate(sorted(unique_vids))}

        return vid_to_index

    # def _make_speaker_model(self, lmdb_dir, cache_path):
    #     # logging.info('  building a speaker model...')
    #     speaker_model = Vocab('vid', insert_default_tokens=False)
    #     lmdb_dir_new= lmdb_dir+ self.loader_type + f"_pickle"
    #     lmdb_env = lmdb.open(lmdb_dir_new, readonly=True, lock=False)
    #     txn = lmdb_env.begin(write=False)
    #     cursor = txn.cursor()
    #     for key, value in cursor:
    #         print(key)
    #         video = pickle.loads(value)
    #         vid = video['vid']
    #         speaker_model.index_word(vid)

    #     lmdb_env.close()
    #     # logging.info('    indexed %d videos' % speaker_model.n_words)
    #     self.speaker_model = speaker_model

    #     # cache
    #     with open(cache_path, 'wb') as f:
    #         pickle.dump(self.speaker_model, f)

    #     return self.speaker_model

    def calc_spectrogram_length_from_motion_length(self, n_frames, fps):
        ret = (n_frames / fps * 16000 - 1024) / 512 + 1
        return int(round(ret))
    
    def make_audio_fixed_length(self, audio, expected_audio_length):
        n_padding = expected_audio_length - len(audio)
        if n_padding > 0:
            audio = np.pad(audio, (0, n_padding), mode='symmetric')
        else:
            audio = audio[0:expected_audio_length]
        return audio
