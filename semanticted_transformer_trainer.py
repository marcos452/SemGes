import train_ted
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
# from avssl.model import KWClip_GeneralTransformer
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
from scipy.signal import argrelextrema
import math
from hydra.utils import instantiate
import copy

class CustomTrainer(train_ted.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
                                                           
        self.args = args
        self.joints = 129
        self.tracker = other_tools.EpochTracker(["huber_value", "loss_rec_index_hand", "loss_rec_index_body", "loss_rec_hand", "loss_rec_body", "acceleration_loss","index_loss","g_loss_final", "cosine_loss","cosine_loss_hand","cosine_loss_body"], [False, False, False, False, False, False, False, False, False, False, False])
        self.alignmenter = metric.alignment(0.3, 2)
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 512
        self.args.vae_layer = 2
        self.sigma = 0.3
        self.order = 2
        # self.mean_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_mean.npy")
        # self.std_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_std.npy")
        
        self.args.vae_test_dim = 180 # need to be changed
        self.vq_model_hand = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hand, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/DDL_1/SemDiffusion_latent_3_ted/beat_pre/pretrained_vq_ted/hand/last_400.bin", args.e_name)
        self.vq_model_hand.eval()

        self.args.vae_test_dim = 78 # need to be changed
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/DDL_1/SemDiffusion_latent_3_ted/beat_pre/pretrained_vq_ted/body/last_400.bin", args.e_name)
        self.vq_model_body.eval()
        
        # self.speechclip_model = self.load_speechclip_model()
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.huber_loss = get_loss_func("huber_loss").to(self.rank) 
        self.audio_extract_config = {
            'name':args.name, 
            '_target_': args.target,  # 使用 argparse 传递的 target 类
            'modelpath_processor': args.modelpath_processor,  # 传递音频处理器路径
            'modelpath_audiomodel': args.modelpath_audiomodel,  # 传递音频模型路径
            'finetune': args.finetune,  # 传递是否微调的参数
            'output_framerate': args.output_framerate  # 传递输出帧率
        }

        # 使用 Hydra 的 instantiate 来动态实例化 audio_extract
        self.audio_extract = instantiate(self.audio_extract_config)

        logger.info(f"1. Audio feature extractor '{self.audio_extract.hparams.name}' loaded")
        audio_encoded_dim = self.audio_extract.audio_encoded_dim  
        
    # def load_speechclip_model(self):
    #     model_fp = self.args.root_path + "icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+/model.ckpt"
    #     model = KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
    #     model.to(self.rank)
    #     model.eval()
    #     return model
    
    # def speechclip(self, wav_data):
    #     with torch.no_grad():
    #         # Use the preloaded model
    #         wav_data.to(self.rank) 
    #         output_embs, hidden_states = self.speechclip_model.feature_extractor_s3prl(wav=wav_data)
    #         output = self.speechclip_model.encode_speech(wav=wav_data)
    #     return output["parallel_audio_feat"], output_embs
    def audio_resize(self, audio_feature, pose):
        resample_audio_feature = []
        for idx in range(len(audio_feature)):
            if audio_feature[idx].shape[1] % 2 != 0:
                audio_feature_one = audio_feature[idx][:, :audio_feature[idx].shape[1] - 1, :]
            else:
                audio_feature_one = audio_feature[idx]


                # print("NOT GENERATION", audio_feature_one.shape, batch['motion'][0].shape)
            if audio_feature_one.shape[1] > pose.shape[1] * 2:
                #print("Shape checking:", audio_feature_one.shape, pose.shape)
                audio_feature_one = audio_feature_one[:, :pose.shape[1] * 2, :]

            audio_feature_one = torch.reshape(audio_feature_one,
                                              (1, audio_feature_one.shape[1] // 2, audio_feature_one.shape[2] * 2))
            #print("audio_feature_one",audio_feature_one.shape)
            resample_audio_feature.append(audio_feature_one)

        return resample_audio_feature
    
    def create_one_hot(self, keyids, emotion_ids):
        style_batch = []
        #print("keyids",keyids)
        for keyid in keyids:
            # 获取身份的独热编码向量
            #print(keyid)
            mapping_dict = {1: 0, 3: 1, 5: 2, 7: 3}
            input_value = int(keyid.item())
            keyid = mapping_dict[input_value]
            keyid = torch.tensor(keyid).to(self.rank)
            identity_vector = torch.eye(4).to(self.rank)[keyid]
            # 获取情绪的独热编码向量
            # emotion_vector = torch.eye(8).to(self.rank)[emotion_id]
            
            # # 拼接身份向量和情绪向量
            # style_vector = torch.cat([identity_vector, emotion_vector], dim=0)
            style_batch.append(identity_vector)
        
        style_batch = torch.stack(style_batch, dim=0)
        return style_batch
        

    def _load_data(self, dict_data):
        tar_pose= dict_data["pose"]
        #print(tar_pose.shape)
        self.audio_extract.to(self.rank)
        # tar_pose = (tar_pose * self.std_pose).cpu().numpy() + self.mean_pose
        tar_pose = torch.Tensor(tar_pose).to(self.rank)

        in_audio = dict_data["audio"]
        #print("in_audio",in_audio.shape)
        audio_feature = self.audio_extract(in_audio, False)
        audio_feature = self.audio_resize(audio_feature, tar_pose)
        audio_feature = torch.tensor([item.cpu().detach().numpy() for item in audio_feature])
        audio_feature = audio_feature.squeeze(1)
        #print("audio_feat",audio_feature.shape)

        in_word = dict_data["word"].to(self.rank)
        #print("in_word:",np.shape(in_word))
        #print("in_word:",np.shape(in_word))
        # in_sem = dict_data["sem"].to(self.rank)
        #print("in_sem.shape",np.shape(in_sem))
        #print("in_sem",in_sem)
        # in_emo = dict_data["emo"].to(self.rank)

        tar_id = dict_data["id"].to(self.rank).long()
        # self.all_identity_onehot = torch.eye(4)

        # style_ont_hot = self.create_one_hot(keyids=tar_id,
        #                                 emotion_ids=in_emo)
        

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        # tar_pose_angle = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        # tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_angle).reshape(bs, n, j*6)
        pose_hand = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 90)).cuda()
        pose_hand = tar_pose[:, :, 24:114]
        pose_hand_matrix = rc.euler_angles_to_matrix(pose_hand.reshape(bs, n, 30, 3),"XYZ")
        pose_hand = rc.matrix_to_rotation_6d(pose_hand_matrix).reshape(bs, n, 30*6)        


        pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 39)).cuda()
        pose_body[:, :, 0:24] = tar_pose[:, :, 0:24]
        pose_body[:, :, 24:39] =tar_pose[:, :, 114:129]
        pose_body_matrix = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 13, 3),"XYZ")
        pose_body = rc.matrix_to_rotation_6d(pose_body_matrix).reshape(bs, n, 13*6)


        latent_hand = self.vq_model_hand.encoder(pose_hand) # bs*n/4

        latent_body = self.vq_model_body.encoder(pose_body) # bs*n/4



        return {
            "in_audio": audio_feature,
            "in_word": in_word,
            "tar_pose": tar_pose,
            "tar_id": tar_id,
            "pose_hand_matrix": pose_hand_matrix,
            "pose_body_matrix": pose_body_matrix,
            "latent_hand": latent_hand,
            "latent_body": latent_body   
        }
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints

        #------ full generatation task ------ #
        # mask_ratio = (epoch / self.args.epochs) * 0.95 + 0.05  
        # mask = torch.rand(bs, n, 512) < mask_ratio
        # mask = mask.float().cuda()
        g_loss_final = 0

        net_out = self.model(loaded_data)
        
        _, rec_index_hand, _, _ = self.vq_model_hand.quantizer(net_out["poses_feat_hands"])
        _, ref_index_hand, _, _ = self.vq_model_hand.quantizer(loaded_data["latent_hand"])

        _, rec_index_body, _, _ = self.vq_model_body.quantizer(net_out["poses_feat_body"])
        _, ref_index_body, _, _ = self.vq_model_body.quantizer(loaded_data["latent_body"])

        loss_rec_index_hand = self.vectices_loss(rec_index_hand, ref_index_hand)
        loss_rec_index_body = self.vectices_loss(rec_index_body, ref_index_body)
        self.tracker.update_meter("loss_rec_index_hand", "train", loss_rec_index_hand.item())
        self.tracker.update_meter("loss_rec_index_body", "train", loss_rec_index_body.item())
        index_loss =loss_rec_index_hand + loss_rec_index_body
        self.tracker.update_meter("index_loss", "train", index_loss.item())
        g_loss_final += index_loss

        
        text_feats = net_out["word_feat_seq"]
        hand_feat_norm = loaded_data["latent_hand"] / loaded_data["latent_hand"].norm(dim=0, keepdim=True)
        cos_hand = self.cosine_sim(text_feats, hand_feat_norm)
        cosine_loss_hand = (1 - cos_hand).mean()
        self.tracker.update_meter("cosine_loss_hand", "train", cosine_loss_hand.item())
        body_feat_norm = loaded_data["latent_body"] / loaded_data["latent_body"].norm(dim=0, keepdim=True)
        cos_body = self.cosine_sim(text_feats, body_feat_norm)
        cosine_loss_body = (1 - cos_body).mean()
        self.tracker.update_meter("cosine_loss_body", "train", cosine_loss_body.item())
        cosine_loss = cosine_loss_hand + cosine_loss_body
        self.tracker.update_meter("cosine_loss", "train", cosine_loss.item())
        g_loss_final += cosine_loss


        rec_body = self.vq_model_body.decoder(rec_index_body)
        rec_pose_body = rc.rotation_6d_to_matrix(rec_body.reshape(bs, -1, 13, 6))
        rec_pose_body = rc.matrix_to_euler_angles(rec_pose_body, "XYZ")
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13, 3).cpu().detach().numpy()
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13*3)
        rec_pose_hand = self.vq_model_hand.decoder(rec_index_hand)
        rec_pose_hand = rc.rotation_6d_to_matrix(rec_pose_hand.reshape(bs, -1, 30, 6))
        rec_pose_hand = rc.matrix_to_euler_angles(rec_pose_hand, "XYZ")
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30, 3).cpu().detach().numpy()
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30*3) 
        rec_pose_final = torch.zeros((loaded_data["tar_pose"].shape[0], rec_pose_body.shape[1], 129)).numpy()
        rec_pose_final[:, :,0:24] = rec_pose_body[:, :, 0:24]
        rec_pose_final[:, :, 114:129] = rec_pose_body[:, :, 24:39]
        rec_pose_final[:, :, 24:114] = rec_pose_hand[:, :, 0:90]
        rec_pose_final = torch.Tensor(rec_pose_final).to(self.rank)
        rec_pose_final = rc.euler_angles_to_matrix(rec_pose_final.reshape(bs, n, 43, 3),"XYZ")
        tar_pose = rc.euler_angles_to_matrix(loaded_data["tar_pose"].reshape(bs, n, 43, 3),"XYZ")

        huber_value = self.huber_loss(tar_pose, rec_pose_final)
        self.tracker.update_meter("huber_value", "train", huber_value.item())
        g_loss_final += huber_value
        self.tracker.update_meter("g_loss_final", "train", g_loss_final.item())

        
        return g_loss_final
    

    def _g_val(self, loaded_data, use_adv, mode="train", epoch=0):


        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints

        g_loss_final = 0

        net_out = self.model(loaded_data)

        text_feats = net_out["word_feat_seq"]
        hand_feat_norm = loaded_data["latent_hand"] / loaded_data["latent_hand"].norm(dim=0, keepdim=True)
        cos_hand = self.cosine_sim(text_feats, hand_feat_norm)
        cosine_loss_hand = (1 - cos_hand).mean()
        self.tracker.update_meter("cosine_loss_hand", "val", cosine_loss_hand.item())
        body_feat_norm = loaded_data["latent_body"] / loaded_data["latent_body"].norm(dim=0, keepdim=True)
        cos_body = self.cosine_sim(text_feats, body_feat_norm)
        cosine_loss_body = (1 - cos_body).mean()
        self.tracker.update_meter("cosine_loss_body", "val", cosine_loss_body.item())
        cosine_loss = cosine_loss_hand + cosine_loss_body
        self.tracker.update_meter("cosine_loss", "val", cosine_loss.item())
        g_loss_final += cosine_loss

        _, rec_index_hand, _, _ = self.vq_model_hand.quantizer(net_out["poses_feat_hands"])
        _, ref_index_hand, _, _ = self.vq_model_hand.quantizer(loaded_data["latent_hand"])

        _, rec_index_body, _, _ = self.vq_model_body.quantizer(net_out["poses_feat_body"])
        _, ref_index_body, _, _ = self.vq_model_body.quantizer(loaded_data["latent_body"])

        
        loss_rec_index_hand = self.vectices_loss(rec_index_hand, ref_index_hand)
        loss_rec_index_body = self.vectices_loss(rec_index_body, ref_index_body)
        self.tracker.update_meter("loss_rec_index_hand", "val", loss_rec_index_hand.item())
        self.tracker.update_meter("loss_rec_index_body", "val", loss_rec_index_body.item())
        index_loss =loss_rec_index_hand + loss_rec_index_body
        self.tracker.update_meter("index_loss", "val", index_loss.item())
        g_loss_final += index_loss

        
        rec_body = self.vq_model_body.decoder(rec_index_body)
        rec_pose_body = rc.rotation_6d_to_matrix(rec_body.reshape(bs, -1, 13, 6))
        rec_pose_body = rc.matrix_to_euler_angles(rec_pose_body, "XYZ")
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13, 3).cpu().detach().numpy()
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13*3)
        rec_pose_hand = self.vq_model_hand.decoder(rec_index_hand)
        rec_pose_hand = rc.rotation_6d_to_matrix(rec_pose_hand.reshape(bs, -1, 30, 6))
        rec_pose_hand = rc.matrix_to_euler_angles(rec_pose_hand, "XYZ")
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30, 3).cpu().detach().numpy()
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30*3) 
        rec_pose_final = torch.zeros((loaded_data["tar_pose"].shape[0], rec_pose_body.shape[1], 129)).numpy()
        rec_pose_final[:, :,0:24] = rec_pose_body[:, :, 0:24]
        rec_pose_final[:, :, 114:129] = rec_pose_body[:, :, 24:39]
        rec_pose_final[:, :, 24:114] = rec_pose_hand[:, :, 0:90]
        rec_pose_final = torch.Tensor(rec_pose_final).to(self.rank)
        rec_pose_final = rc.euler_angles_to_matrix(rec_pose_final.reshape(bs, n, 43, 3),"XYZ")
        tar_pose = rc.euler_angles_to_matrix(loaded_data["tar_pose"].reshape(bs, n, 43, 3),"XYZ")
        huber_value = self.huber_loss(tar_pose, rec_pose_final)
        self.tracker.update_meter("huber_value", "val", huber_value.item())
        g_loss_final += huber_value
        self.tracker.update_meter("g_loss_final", "val", g_loss_final.item())

        return g_loss_final
    




    def _load_data_test(self, dict_data):
        tar_pose= dict_data["pose"]
        #print(tar_pose.shape)
        self.audio_extract.to(self.rank)
        tar_pose = torch.Tensor(tar_pose).to(self.rank)
        in_audio = dict_data["audio"]
        #print("in_audio",in_audio.shape)
        audio_feature = self.audio_extract(in_audio, False)
        audio_feature = self.audio_resize(audio_feature, tar_pose)
        audio_feature = torch.tensor([item.cpu().detach().numpy() for item in audio_feature])
        audio_feature = audio_feature.squeeze(1)

        in_word = dict_data["word"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()


        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
        # tar_pose_angle = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        # tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_angle).reshape(bs, n, j*6)
        pose_hand = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 90)).cuda()
        pose_hand = tar_pose[:, :, 24:114]
        pose_hand_matrix = rc.euler_angles_to_matrix(pose_hand.reshape(bs, n, 30, 3),"XYZ")
        pose_hand = rc.matrix_to_rotation_6d(pose_hand_matrix).reshape(bs, n, 30*6)        


        pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 39)).cuda()
        pose_body[:, :, 0:24] = tar_pose[:, :, 0:24]
        pose_body[:, :, 24:39] =tar_pose[:, :, 114:129]
        pose_body_matrix = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 13, 3),"XYZ")
        pose_body = rc.matrix_to_rotation_6d(pose_body_matrix).reshape(bs, n, 13*6)


        latent_hand = self.vq_model_hand.encoder(pose_hand) # bs*n/4

        latent_body = self.vq_model_body.encoder(pose_body) # bs*n/4


        cat_data = {
        "in_audio": audio_feature,
        "in_word": in_word,
        "tar_pose": tar_pose,
        "latent_hand": latent_hand,
        "latent_body": latent_body,
        "tar_id": tar_id}

        net_out  = self.model(cat_data)
        _, rec_index_body, _, _ = self.vq_model_body.quantizer(net_out["poses_feat_body"])
        _, rec_index_hand, _, _ = self.vq_model_hand.quantizer(net_out["poses_feat_hands"])

        rec_pose_body = self.vq_model_body.decoder(rec_index_body)
        rec_pose_body = rc.rotation_6d_to_matrix(rec_pose_body.reshape(bs, -1, 13, 6))
        rec_pose_body = rc.matrix_to_euler_angles(rec_pose_body, "XYZ")
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13, 3).cpu().numpy() 
        rec_pose_body = rec_pose_body.reshape(bs, -1, 13*3)

        rec_pose_hand = self.vq_model_hand.decoder(rec_index_hand)
        rec_pose_hand = rc.rotation_6d_to_matrix(rec_pose_hand.reshape(bs, -1, 30, 6))
        rec_pose_hand = rc.matrix_to_euler_angles(rec_pose_hand, "XYZ")
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30, 3).cpu().numpy() 
        rec_pose_hand = rec_pose_hand.reshape(bs, -1, 30*3) 

        rec_pose_final = torch.zeros((tar_pose.shape[0], rec_pose_body.shape[1], 129)).numpy()
        rec_pose_final[:, :,0:24] = rec_pose_body[:, :, 0:24]
        rec_pose_final[:, :, 114:129] = rec_pose_body[:, :, 24:39]
        rec_pose_final[:, :, 24:114] = rec_pose_hand[:, :, 0:90]

        return {
            'rec_pose': rec_pose_final,
            "tar_pose": tar_pose,
            "in_audio": in_audio,
            "in_word": in_word
        }
    


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
        # test_seq_list = os.listdir(self.args.test_data_path+f"bvh_rot_vis/")
        # test_seq_list.sort()
        align = 0 
        latent_out = []
        latent_ori = []
        angle_pair = [
        (0, 1),
        (0, 2),
        (1, 3),
        (3, 4),
        (5, 6),
        (6, 7),
        (8, 9),
        (9, 10),
        (11, 12),
        (12, 13),
        (14, 15),
        (15, 16),
        (17, 18),
        (18, 19),
        (17, 5),
        (5, 8),
        (8, 14),
        (14, 11),
        (2, 20),
        (20, 21),
        (22, 23),
        (23, 24),
        (25, 26),
        (26, 27),
        (28, 29),
        (29, 30),
        (31, 32),
        (32, 33),
        (34, 35),
        (35, 36),
        (34, 22),
        (22, 25),
        (25, 31),
        (31, 28),
        (0, 37),
        (37, 38),
        (37, 39),
        (38, 40),
        (39, 41),
        # palm
        (4, 42),
        (21, 43)]

        change_angle = [0.0027804733254015446, 0.002761547453701496, 0.005953566171228886, 0.013764726929366589, 
            0.022748252376914024, 0.039307352155447006, 0.03733552247285843, 0.03775784373283386, 0.0485558956861496, 
            0.032914578914642334, 0.03800227493047714, 0.03757007420063019, 0.027338404208421707, 0.01640886254608631, 
            0.003166505601257086, 0.0017252820543944836, 0.0018696568440645933, 0.0016072227153927088, 0.005681346170604229, 
            0.013287615962326527, 0.021516695618629456, 0.033936675637960434, 0.03094293735921383, 0.03378918394446373, 
            0.044323261827230453, 0.034706637263298035, 0.03369896858930588, 0.03573163226246834, 0.02628341130912304, 
            0.014071882702410221, 0.0029828345868736506, 0.0015706412959843874, 0.0017107439925894141, 0.0014634154504165053, 
            0.004873405676335096, 0.002998138777911663, 0.0030240598134696484, 0.0009890805231407285, 0.0012279648799449205, 
            0.047324635088443756, 0.04472292214632034]
        mean_dir_vec = [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 ,  0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616,  0.2257808,  0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, -0.2062069,  0.2305175, -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, -0.6464546, -0.5284583,  0.0457585, -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618, -0.3054556, -0.3556116, -0.281753 , -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526,  0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 ,  0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361,  0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372]
        t_start = 10
        t_end = 500
        thres = 0.001
        sigma = 0.1
        self.model.eval()
        self.eval_copy.eval()
        latent_out_all = []
        latent_ori_all = []
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                print(its)
                net_out = self._load_data_test(batch_data)          
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                in_audio = net_out['in_audio']

                print("len(self.test_loader)",len(self.test_loader))
                bs, n= tar_pose.shape[0], tar_pose.shape[1]
                batch_size = bs
                # print(bs)
                roundt = n  // self.args.pose_length
                # tar_pose = tar_pose.cpu()

                # _ = self.srgr_calculator.run(rec_pose, tar_pose.cpu().numpy(), in_sem.cpu().numpy())
                latent_ori_temp = []
                latent_out_temp = []
                # for i in range(roundt)
                rec_pose = torch.tensor(rec_pose, dtype=torch.float32).to(self.rank)
                pose_hand_rec = torch.zeros((rec_pose.shape[0], rec_pose.shape[1], 90)).cpu()
                pose_hand_rec = rec_pose[:, :, 24:114]

                pose_hand_rec = rc.euler_angles_to_matrix(pose_hand_rec.reshape(bs, n, 30, 3),"XYZ")
                pose_hand_rec = rc.matrix_to_rotation_6d(pose_hand_rec).reshape(bs, n, 30*6)
                pose_hand_rec = pose_hand_rec.to(self.rank)
                latent_hand_rec = self.vq_model_hand.encoder(pose_hand_rec) # bs*n/4

                pose_body_rec = torch.zeros((rec_pose.shape[0], rec_pose.shape[1], 39)).cpu()
                pose_body_rec[:, :, 0:24] = rec_pose[:, :, 0:24]
                pose_body_rec[:, :, 24:39] =rec_pose[:, :, 114:129]
                pose_body_rec = rc.euler_angles_to_matrix(pose_body_rec.reshape(bs, n, 13, 3),"XYZ")
                pose_body_rec = rc.matrix_to_rotation_6d(pose_body_rec).reshape(bs, n, 13*6)
                pose_body_rec = pose_body_rec.to(self.rank)
                latent_body_rec = self.vq_model_body.encoder(pose_body_rec) # bs*n/4
                latent_ori = torch.cat([latent_hand_rec, latent_body_rec], 1)
                # latent_ori_temp.append(latent_ori)
                

                pose_hand_tar = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 90)).cuda()
                pose_hand_tar = tar_pose[:, :, 24:114]
                pose_hand_tar = rc.euler_angles_to_matrix(pose_hand_tar.reshape(bs, n, 30, 3),"XYZ")
                pose_hand_tar = rc.matrix_to_rotation_6d(pose_hand_tar).reshape(bs, n, 30*6)        
                latent_hand_tar = self.vq_model_hand.encoder(pose_hand_tar) 
                pose_body_tar = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 39)).cuda()
                pose_body_tar[:, :, 0:24] = tar_pose[:, :, 0:24]
                pose_body_tar[:, :, 24:39] =tar_pose[:, :, 114:129]
                pose_body_tar = rc.euler_angles_to_matrix(pose_body_tar.reshape(bs, n, 13, 3),"XYZ")
                pose_body_tar = rc.matrix_to_rotation_6d(pose_body_tar).reshape(bs, n, 13*6)
                latent_body_tar = self.vq_model_body.encoder(pose_body_tar)
                latent_out = torch.cat([latent_hand_tar, latent_body_tar], 1)
                # latent_out_temp.append(latent_out)
                # latent_ori = torch.cat(latent_ori_temp, 1)
                # latent_out = torch.cat(latent_out_temp, 1)

            
                # if its == 0:
                #     latent_out_all = latent_out.cpu().numpy()
                #     latent_ori_all = latent_ori.cpu().numpy()
                # else:
                #     if its < 23:
                
                latent_out_all.append(latent_out.cpu().numpy())
                latent_ori_all.append(latent_ori.cpu().numpy())
                print('latent_out_all:',np.shape(latent_out_all), np.shape(latent_out))
                print('latent_ori_all":',np.shape(latent_ori_all), np.shape(latent_ori_all))
                if its+1 ==len(self.test_loader):
                    latent_out_all = np.concatenate(latent_out_all, axis=0)
                    print('latent_out_all:',np.shape(latent_out_all))
                    latent_ori_all = np.concatenate(latent_ori_all, axis=0)
                    print('latent_out_all:',np.shape(latent_ori_all))

                # rec_pose = rec_pose.reshape(-1, 129)
                rec_pose_diversity = copy.deepcopy(rec_pose)
                rec_pose_diversity = rec_pose_diversity.cpu().numpy()
                _ = self.l1_calculator.run(rec_pose_diversity)
                # print(rec_pose.shape)
                # print(np.shape(mean_dir_vec))
                # print(torch.tensor(mean_dir_vec).unsqueeze(0).unsqueeze(0).shape)
                mean_dir_vec_tensor = torch.tensor(mean_dir_vec)
                # 添加一个0，扩展到129个元素
                mean_dir_vec_tensor = torch.cat((mean_dir_vec_tensor, torch.tensor([0.0,0.0,0.0])))
                # 使用 unsqueeze 来添加额外的维度，变成 (128, 1, 129)
                mean_dir_vec_tensor = mean_dir_vec_tensor.unsqueeze(0).unsqueeze(0)
                out_dir_vec_bc = rec_pose + mean_dir_vec_tensor.to(self.rank)
                left_palm = torch.cross(out_dir_vec_bc[:, :, 11 * 3 : 12 * 3], out_dir_vec_bc[:, :, 17 * 3 : 18 * 3], dim = 2)
                right_palm = torch.cross(out_dir_vec_bc[:, :, 28 * 3 : 29 * 3], out_dir_vec_bc[:, :, 34 * 3 : 35 * 3], dim = 2)
                beat_vec = torch.cat((out_dir_vec_bc, left_palm, right_palm), dim = 2)
                beat_vec = F.normalize(beat_vec, dim = -1)
                all_vec = beat_vec.reshape(beat_vec.shape[0] * beat_vec.shape[1], -1, 3)
                
                for idx, pair in enumerate(angle_pair):
                    vec1 = all_vec[:, pair[0]]
                    vec2 = all_vec[:, pair[1]]
                    inner_product = torch.einsum('ij,ij->i', [vec1, vec2])
                    inner_product = torch.clamp(inner_product, -1, 1, out=None)
                    angle = torch.acos(inner_product) / math.pi
                    angle_time = angle.reshape(batch_size, -1)
                    if idx == 0:
                        angle_diff = torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                    else:
                        angle_diff += torch.abs(angle_time[:, 1:] - angle_time[:, :-1]) / change_angle[idx] / len(change_angle)
                angle_diff = torch.cat((torch.zeros(batch_size, 1).to(self.rank), angle_diff), dim = -1)
                
                for b in range(batch_size):
                    motion_beat_time = []
                    for t in range(2, 33):
                        if (angle_diff[b][t] < angle_diff[b][t - 1] and angle_diff[b][t] < angle_diff[b][t + 1]):
                            if (angle_diff[b][t - 1] - angle_diff[b][t] >= thres or angle_diff[b][t + 1] - angle_diff[b][t] >= thres):
                                motion_beat_time.append(float(t) / 15.0)
                    if (len(motion_beat_time) == 0):
                        continue
                    audio = in_audio[b].cpu().numpy()
                    audio_beat_time = librosa.onset.onset_detect(y=audio, sr=16000, units='time')
                    sum = 0
                    for audio in audio_beat_time:
                        sum += np.power(math.e, -np.min(np.power((audio - motion_beat_time), 2)) / (2 * sigma * sigma))
                    self.bc.update(sum / len(audio_beat_time), len(audio_beat_time))
            
        align_avg = self.bc.avg
        logger.info(f"align score: {align_avg}")
        data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        latent_out_all = latent_out_all.reshape(-1, 512)
        latent_ori_all = latent_ori_all.reshape(-1, 512)
        fgd_motion = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fgd_motion: {fgd_motion}")
        # srgr = self.srgr_calculator.avg()
        # logger.info(f"srgr score: {srgr}")
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        self.test_recording(epoch) 

