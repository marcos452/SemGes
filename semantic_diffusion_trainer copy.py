import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
from avssl.model import KWClip_GeneralTransformer
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
from scipy.signal import argrelextrema
import math

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
                                                           
        self.args = args
        self.joints = self.train_data.joints
        self.tracker = other_tools.EpochTracker(["loss_latent", "vel_loss", "ver", "embedding_loss", "kl", "acceleration_loss","index_loss","g_loss_final"], [False, False, False, False, False, False, False, False])
        self.alignmenter = metric.alignment(0.3, 2)
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 512
        self.args.vae_layer = 2
        self.sigma = 0.3
        self.order = 2
        self.mean_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_mean.npy")
        self.std_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_std.npy")
        
        self.args.vae_test_dim = 228 # need to be changed
        self.vq_model_hand = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hand, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_3/beat_pre/pretrained_vq/hand/last_1900.bin", args.e_name)
        self.vq_model_hand.eval()

        self.args.vae_test_dim = 54 # need to be changed
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_3/beat_pre/pretrained_vq/body/last_1900.bin", args.e_name)
        self.vq_model_body.eval()
        
        self.speechclip_model = self.load_speechclip_model()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
        
    def load_speechclip_model(self):
        model_fp = self.args.root_path + "icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+/model.ckpt"
        model = KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
        model.to(self.rank)
        model.eval()
        return model
    
    def speechclip(self, wav_data):
        with torch.no_grad():
            # Use the preloaded model
            wav_data.to(self.rank) 
            output = self.speechclip_model.encode_speech(wav=wav_data)
        return output["parallel_audio_feat"]
    
    def _load_data(self, dict_data):
        tar_pose= dict_data["pose"]
        #print(tar_pose.shape)

        tar_pose = (tar_pose * self.std_pose).cpu().numpy() + self.mean_pose
        tar_pose = torch.Tensor(tar_pose).to(self.rank)

        in_audio = dict_data["audio"].to(self.rank)
        #print("in_audio",in_audio.shape)
        audio_feat = self.speechclip(in_audio)
        #print("audio_feat",audio_feat.shape)
        in_word = dict_data["word"].to(self.rank)
        #print("in_word:",np.shape(in_word))
        #print("in_word:",np.shape(in_word))
        in_sem = dict_data["sem"].to(self.rank)
        #print("in_sem.shape",np.shape(in_sem))
        #print("in_sem",in_sem)

        tar_id = dict_data["id"].to(self.rank).long()

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        
        # tar_pose_angle = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        # tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_angle).reshape(bs, n, j*6)
        pose_hand = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 114)).cuda()
        pose_hand[:, :, 0:57] = tar_pose[:, :, 18:75]
        pose_hand[:, :, 57:114] =tar_pose[:, :, 84:141]
        pose_hand = (torch.Tensor(pose_hand.reshape(-1, 3))/180)*np.pi
        pose_hand = rc.euler_angles_to_matrix(pose_hand.reshape(bs, n, 38, 3),"XYZ")
        pose_hand = rc.matrix_to_rotation_6d(pose_hand).reshape(bs, n, 38*6)        


        pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 27)).cuda()
        pose_body[:, :, 0:18] = tar_pose[:, :, 0:18]
        pose_body[:, :, 18:27] =tar_pose[:, :, 75:84]
        pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
        pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 9, 3),"XYZ")
        pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n, 9*6)

        
        tar_index_value_hand = self.vq_model_hand.map2index(pose_hand) # bs*n/4
        tar_index_value_body = self.vq_model_body.map2index(pose_body) # bs*n/4


        latent_hand = self.vq_model_hand.map2latent(pose_hand) # bs*n/4
        in_pre_pose_hand = latent_hand.new_zeros((bs, n, 512)).cuda()
        in_pre_pose_hand[:, 0:self.args.pre_frames, :-1] = latent_hand[:, 0:self.args.pre_frames, :-1]
        in_pre_pose_hand[:, 0:self.args.pre_frames, -1] = 1

        latent_body = self.vq_model_body.map2latent(pose_body) # bs*n/4
        in_pre_pose_body = latent_body.new_zeros((bs, n, 512)).cuda()
        in_pre_pose_body[:, 0:self.args.pre_frames, :-1] = latent_body[:, 0:self.args.pre_frames, :-1]
        in_pre_pose_body[:, 0:self.args.pre_frames, -1] = 1


        return {
            "in_audio": audio_feat,
            "in_word": in_word,
            "tar_pose": tar_pose,
            "in_sem": in_sem,
            "tar_id": tar_id,
            "in_pre_pose_hand": in_pre_pose_hand,
            "in_pre_pose_body": in_pre_pose_body,
            "tar_index_value_hand": tar_index_value_hand,
            "tar_index_value_body": tar_index_value_body,
            "latent_hand": latent_hand,
            "latent_body": latent_body,
        }
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # ------ full generatation task ------ #

        g_loss_final = 0

        net_out = self.model(loaded_data)
        loss_latent_hands = self.reclatent_loss(net_out["poses_feat_hands"], loaded_data["latent_hand"])
        loss_latent_body = self.reclatent_loss(net_out["poses_feat_body"], loaded_data["latent_body"])

        loss_latent = self.args.lh*loss_latent_hands + self.args.lu*loss_latent_body
        self.tracker.update_meter("loss_latent", "train", loss_latent.item())
        g_loss_final += loss_latent


        # rec_index_hand = self.log_softmax(net_out["index_hands"]).reshape(-1, self.args.vae_codebook_size)
        # rec_index_body = self.log_softmax(net_out["index_body"]).reshape(-1, self.args.vae_codebook_size)
        # index_loss = self.cls_loss(rec_index_hand, loaded_data["tar_index_value_hand"]) + self.cls_loss(rec_index_body, loaded_data["tar_index_value_body"])
        # self.tracker.update_meter("index_loss", "train", index_loss.item())
        # g_loss_final += index_loss  


        # if mode != 'train':

            
        #     _, rec_index_upper, _, _ = self.vq_model.quantizer(net_out["pre_latent"])
        #     rec_upper = self.vq_model.decoder(rec_index_upper)

        #     rec_pose = rec_upper.reshape(bs, n, j, 6)

        return g_loss_final
    

    def _g_val(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # ------ full generatation task ------ #

        g_loss_final = 0

        net_out = self.model(loaded_data)
        loss_latent_hands = self.reclatent_loss(net_out["poses_feat_hands"], loaded_data["latent_hand"])
        loss_latent_body = self.reclatent_loss(net_out["poses_feat_body"], loaded_data["latent_body"])


        loss_latent = self.args.lh*loss_latent_hands + self.args.lu*loss_latent_body

        self.tracker.update_meter("loss_latent", "val", loss_latent.item())
        g_loss_final += loss_latent

        # rec_index_hand = self.log_softmax(net_out["index_hands"]).reshape(-1, self.args.vae_codebook_size)
        # print("rec_index_hand",rec_index_hand.shape)
        # print("loaded_data['tar_index_value_hand']",loaded_data["tar_index_value_hand"].shape)
        # rec_index_body = self.log_softmax(net_out["index_body"]).reshape(-1, self.args.vae_codebook_size)
        # index_loss = self.cls_loss(rec_index_hand, loaded_data["tar_index_value_hand"]) + self.cls_loss(rec_index_body, loaded_data["tar_index_value_body"])
        # self.tracker.update_meter("index_loss", "val", index_loss.item())
        # g_loss_final += index_loss 

        _, rec_index_hand, _, _ = self.vq_model_hand.quantizer(net_out["poses_feat_hands"])
        #print("rec_index_hand",rec_index_hand.shape)
        rec_hand = self.vq_model_hand.decoder(rec_index_hand)

        _, rec_index_body, _, _ = self.vq_model_body.quantizer(net_out["poses_feat_body"])
        rec_body = self.vq_model_hand.decoder(rec_index_body)

        return g_loss_final
    




    def _load_data_test(self, dict_data):
        tar_pose= dict_data["pose"].to(self.rank)
        #print(tar_pose.shape)

        in_audio = dict_data["audio"].to(self.rank)
        #print("in_audio",in_audio.shape)
        #print("audio_feat",audio_feat.shape)
        in_word = dict_data["word"].to(self.rank)
        #print("in_word:",np.shape(in_word))
        #print("in_word:",np.shape(in_word))
        in_sem = dict_data["sem"].to(self.rank)
        #print("in_sem.shape",np.shape(in_sem))
        #print("in_sem",in_sem)

        tar_id = dict_data["id"].to(self.rank).long()

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_angle = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_angle).reshape(bs, n, j*6)
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames



        
        recon_pose = []
        for i in range(roundt):

            cat_tar_pose = tar_pose_6d[:,i*self.args.stride:i*self.args.stride+self.args.pose_length, :]
            bs, n, j = cat_tar_pose.shape[0], cat_tar_pose.shape[1], self.joints
            in_pre_pose = cat_tar_pose.new_zeros((bs, n, j*6)).cuda()

            in_pre_pose[:, 0:self.args.pre_frames, :-1] = tar_pose_6d[:, 0:self.args.pre_frames, :-1]
            in_pre_pose[:, 0:self.args.pre_frames, -1] = 1
            cat_sem = in_sem[:,i*self.args.stride:i*self.args.stride+self.args.pose_length]
            #print("in_word",in_word.shape)
            #print("self.args.stride",self.args.stride)
            cat_in_word = in_word[:,i*self.args.stride:i*self.args.stride+self.args.pose_length]
            #print("cat_in_word",cat_in_word.shape)
            #print("in_audio",in_audio.shape)
            # Calculate the start and end indices for the audio segment
            cat_in_audio = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            audio_feat = self.speechclip(cat_in_audio)


            cat_data = {
            "in_audio": audio_feat,
            "in_word": cat_in_word,
            "tar_pose": tar_pose,
            "tar_pose_6d": tar_pose_6d,
            "in_sem": cat_sem,
            "tar_id": tar_id,
            "latent": in_pre_pose,}

            net_out  = self.model(cat_data)
            recon_pose.append(net_out["rec_pose"])
        recon_pose = torch.cat(recon_pose, 1)
        rec_pose = recon_pose.reshape(bs, -1, j, 6)

        return {
            'rec_pose': rec_pose,
            "tar_pose": tar_pose
        }
        # return {
        #     "in_audio": audio_feat,
        #     "in_word": cat_in_word,
        #     "tar_pose": cat_tar_pose,
        #     "tar_pose_6d": tar_pose_6d,
        #     "tar_index": tar_index,
        #     "in_sem": cat_sem,
        #     "tar_id": tar_id,
        #     "latent": latent,

        # }    

    


    def _g_test(self, loaded_data):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 

        mode = "test"
        net_out  = self.model(loaded_data,mode)
        rec_index_val = self.log_softmax(net_out["latent_index"]).reshape(-1, self.args.vae_codebook_size)
        _, rec_index_upper = torch.max(rec_index_val.reshape(-1, self.args.pose_length, self.args.vae_codebook_size), dim=2)
        rec_upper = self.vq_model.decoder(rec_index_upper)

        rec_pose = rec_upper.reshape(bs, n, j, 6)
            
        
        return {
            'rec_pose': rec_pose,
            # rec_trans': rec_pose_trans,
            'tar_pose': loaded_data["tar_pose_6d"],
        }
    

    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 


    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        # self.d_model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)
                bs, _, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints    
                g_loss_final = self._g_val(loaded_data, False, 'val', epoch)
                self.tracker.update_meter("g_loss_final", "val", g_loss_final.item())

 
            self.val_recording(epoch)
    
    def load_audio(self, wave, t_start, t_end, without_file=False, sr_audio=16000):
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        short_y = y[t_start*sr:t_end*sr]
        self.oenv = librosa.onset.onset_strength(y=short_y, sr=sr)
        self.times = librosa.times_like(self.oenv)
        # Detect events without backtracking
        onset_raw = librosa.onset.onset_detect(onset_envelope=self.oenv, backtrack=False)
        onset_bt = librosa.onset.onset_backtrack(onset_raw, self.oenv)
        if len(short_y)==0:
            print("empty audio")
        self.S = np.abs(librosa.stft(y=short_y))
        self.rms = librosa.feature.rms(S=self.S)
        onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, self.rms[0])
        return onset_raw, onset_bt, onset_bt_rms

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
        data_each_file = np.array(data_each_file)
        vel= data_each_file[1:, :] - data_each_file[:-1, :]
        # l1 
        # vel_rigth_shoulder = abs(vel[:, 0]) + abs(vel[:, 1]) + abs(vel[:, 2])
        # vel_rigth_arm = abs(vel[:, 3]) + abs(vel[:, 4]) + abs(vel[:, 5])
        # vel_rigth_wrist = abs(vel[:, 6]) + abs(vel[:, 7]) + abs(vel[:, 8])
        # l2
        vel_right_shoulder = np.linalg.norm(np.array([vel[:, 0], vel[:, 1], vel[:, 2]]), axis=0)
        vel_right_arm = np.linalg.norm(np.array([vel[:, 3], vel[:, 4], vel[:, 5]]), axis=0)
        vel_right_wrist = np.linalg.norm(np.array([vel[:, 6], vel[:, 7], vel[:, 8]]), axis=0)
        beat_right_arm = argrelextrema(vel_right_arm[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        beat_right_shoulder = argrelextrema(vel_right_shoulder[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        beat_right_wrist = argrelextrema(vel_right_wrist[t_start*pose_fps:t_end*pose_fps], np.less, order=self.order)
        vel_left_shoulder = np.linalg.norm(np.array([vel[:, 9], vel[:, 10], vel[:, 11]]), axis=0)
        vel_left_arm = np.linalg.norm(np.array([vel[:, 12], vel[:, 13], vel[:, 14]]), axis=0)
        vel_left_wrist = np.linalg.norm(np.array([vel[:, 15], vel[:, 16], vel[:, 17]]), axis=0)
        beat_left_arm = argrelextrema(vel_left_arm, np.less, order=self.order)
        beat_left_shoulder = argrelextrema(vel_left_shoulder, np.less, order=self.order)
        beat_left_wrist = argrelextrema(vel_left_wrist, np.less, order=self.order)
        return beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist


    def calculate_align(self, onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, pose_fps=15):
        # more stable solution
        # avg_dis_all_b2a = 0
        # for audio_beat in [onset_raw, onset_bt, onset_bt_rms]:
        #     for pose_beat in [beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist]:
        #         audio_bt = librosa.frames_to_time(audio_beat)
        #         pose_bt = self.motion_frames2time(pose_beat, 0, pose_fps)
        #         dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        #         avg_dis_all_b2a += dis_all_b2a
        # avg_dis_all_b2a /= 18
        audio_bt = librosa.frames_to_time(onset_bt_rms)
        pose_bt = self.motion_frames2time(beat_right_wrist, 0, pose_fps)
        avg_dis_all_b2a = self.GAHR(pose_bt, audio_bt, self.sigma)
        return avg_dis_all_b2a  
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        if isinstance(vel, (tuple, list, np.ndarray)):
            time_vel = tuple(v / pose_fps + offset for v in vel)
        else:
            time_vel = vel / pose_fps + offset
        return time_vel

    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        
        # 确保 a 和 b 是一维数组或列表
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()
        
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2) / (2 * sigma**2))
        
        dis_all_b2a /= len(b)
        return dis_all_b2a 
    
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.args.test_data_path+f"bvh_rot_vis/")
        test_seq_list.sort()
        align = 0 
        latent_out = []
        latent_ori = []
        t_start = 10
        t_end = 500
        self.model.eval()
        self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                net_out = self._load_data_test(batch_data)          
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                # in_audio = loaded_data["in_audio"]
                # in_sem = loaded_data["in_sem"]
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                # if (30/self.args.pose_fps) != 1:
                #     assert 30%self.args.pose_fps == 0
                #     n *= int(30/self.args.pose_fps)
                #     tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                #     rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # num_divs = (tar_pose.shape[1]-self.args.pose_length)//self.args.stride+1

                # for i in range(num_divs):
                #     if i == 0:
                #         cat_results = rec_pose[:,i*self.args.stride:i*self.args.stride+self.args.pose_length, :]
                #         cat_targets = tar_pose[:,i*self.args.stride:i*self.args.stride+self.args.pose_length, :]
                #         #cat_sem = in_sem[:,i*self.stride:i*self.stride+self.pose_length]
                #     else:
                #         cat_results = torch.cat([cat_results, rec_pose[:,i*self.args.stride:i*self.args.stride+self.args.pose_length, :]], 0)
                #         cat_targets = torch.cat([cat_targets, tar_pose[:,i*self.args.stride:i*self.args.stride+self.args.pose_length, :]], 0)
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.view(-1, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).view(-1, j*3).cpu().numpy()
                np_cat_results = (rec_pose.reshape(-1, self.args.pose_dims) * self.std_pose) + self.mean_pose
                # _ = self.l1_calculator.run(np_cat_results)
                

                # latent_out = self.eval_model(cat_results)
                # latent_ori = self.eval_model(cat_targets)

                # if its == 0:
                #     latent_out_all = latent_out.cpu().numpy()
                #     latent_ori_all = latent_ori.cpu().numpy()
                # else:
                #     latent_out_all = np.concatenate([latent_out_all, latent_out.cpu().numpy()], axis=0)
                #     latent_ori_all = np.concatenate([latent_ori_all, latent_ori.cpu().numpy()], axis=0)
                

                # out_final = (rec_pose.reshape(-1, self.args.pose_dims) * self.std_pose) + self.mean_pose
                # np_cat_results = out_final   
                # tar_pose = rc.rotation_6d_to_matrix(tar_pose.view(-1, j, 6))
                # tar_pose = rc.matrix_to_axis_angle(tar_pose).view(-1, j*3).cpu().numpy()
                # np_cat_targets = (tar_pose.reshape(-1, self.args.pose_dims) * self.std_pose) + self.mean_pose
                # # _ = self.srgr_calculator.run(np_cat_results, np_cat_targets, in_sem.cpu().numpy())
                # total_length += out_final.shape[0]

                
                # onset_raw, onset_bt, onset_bt_rms = self.load_audio(in_audio.cpu().numpy().reshape(-1), t_start, t_end, True)
                # beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(out_final, t_start, t_end, self.args.pose_fps, True)
                # align += self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist, self.args.pose_fps)

                
                # with open(f"/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/SemGesture/SemDiffusion_latent_2/datasets/beat_cache/beat_english_15_141_origin/test/bvh_rot_vis/{test_seq_list[its]}", "r") as f_demo:
                #     with open(f"{results_save_path}gt_{test_seq_list[its]}", 'w+') as f_gt:
                #         with open(f"{results_save_path}res_+{test_seq_list[its]}", 'w+') as f_real:
                #             for i, line_data in enumerate(f_demo.readlines()):
                #                 if i < 431:
                #                     f_real.write(line_data)
                #                     f_gt.write(line_data)
                #                 else: break
                #             for line_id in range(n): #,args.pre_frames, args.pose_length
                #                 line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                #                 f_real.write(line_data[1:-2]+'\n')
                #             for line_id in range(n): #,args.pre_frames, args.pose_length
                #                 line_data = np.array2string(tar_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                #                 f_gt.write(line_data[1:-2]+'\n')
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(np_cat_results.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(np_cat_results[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')  
            
        # align_avg = align/len(self.test_loader)
        # logger.info(f"align score: {align_avg}")
        # srgr = self.srgr_calculator.avg()
        # logger.info(f"srgr score: {srgr}")
        # l1div = self.l1_calculator.avg()
        # logger.info(f"l1div score: {l1div}")
        #fgd_motion = data_tools.FIDCalculator.frechet_distance(latent_out_motion_all, latent_ori_all)
        #logger.info(f"fgd_motion: {fgd_motion}")
        data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        self.test_recording(epoch) 

