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

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools

class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        self.args = args
        # train, val, test
        self.loader_type = loader_type


        self.rank = dist.get_rank()
        self.new_cache = args.new_cache
        self.pose_length = args.pose_length #34
        self.stride = args.stride #10
        self.pose_fps = args.pose_fps #15
        self.pose_dims = args.pose_dims # 141

        self.speaker_dims = args.speaker_dims
        self.audio_rep = args.audio_rep
        self.pose_rep = args.pose_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.audio_fps = args.audio_fps
        # self.id_rep = args.speaker_id
        self.ori_joint_list = joints_list[self.args.ori_joints]
        #self.tar_joint_list = joints_list[self.args.tar_joints]
        # if 'smplx' in self.args.pose_rep:
        #     self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        #     self.joints = len(list(self.tar_joint_list.keys()))  
        #     for joint_name in self.tar_joint_list:
        #         self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # else:
        self.joints = len(list(self.ori_joint_list.keys()))
        # self.joint_mask = np.zeros(self.joints*3)
        # for joint_name in self.tar_joint_list:
        #     if joint_name == "Hips":
        #         self.joint_mask[3:6] = 1
        #     else:
        #         self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1        

        
        self.ori_stride = self.stride
        self.ori_length = self.pose_length
        self.alignment = [0,0] # for beat
        
        

        
        if args.word_rep is not None:
            with open(f"{args.data_path_1}vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)

        # preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"         
        preloaded_dir = self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"      

        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 

        
    def __len__(self):
        return self.n_samples
    

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            tar_pose, in_audio, in_facial, in_word, emo, sem, vid = sample
            #print(in_shape)
            #vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int() 
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                #trans = torch.from_numpy(trans).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                #in_shape = torch.from_numpy(in_shape).float()
            else:
                #in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                #trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "word":in_word, "id":vid, "emo":emo, "sem":sem}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False