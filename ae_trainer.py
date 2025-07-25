import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation
import torch
import os,sys 
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,parentdir) 
from avssl.model import KWClip_GeneralTransformer
import sys

import librosa
import numpy as np
# load model to GPU


class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.joints = self.train_data.joints
        self.tracker = other_tools.EpochTracker(["rec_loss", "vel_loss", "ver", "embedding_loss", "kl", "acceleration_loss","cosine_loss","g_loss_final"], [False, False, False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        #self.speechclip_model = self.load_speechclip_model()
        self.mean_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_mean.npy")
        self.std_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_std.npy")

        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 512
        self.args.vae_layer = 2

        self.args.vae_test_dim = 228 # need to be changed
        self.vq_model_hand = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hand, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_5/beat_pre/pretrained_vq/hand/last_1900.bin", args.e_name)
        self.vq_model_hand.eval()

        self.args.vae_test_dim = 54 # need to be changed
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_5/beat_pre/pretrained_vq/body/last_1900.bin", args.e_name)
        self.vq_model_body.eval()



    def load_speechclip_model(self):
        device = torch.device("cuda")
        model_fp = "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_3/icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+/model.ckpt"
        model = KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
        model.to(device)
        model.eval()
        return model

    def speechclip(self, wav_data):
        with torch.no_grad():
            # Use the preloaded model
            device = torch.device("cuda")
            wav_data.to(device)
            output = self.speechclip_model.encode_speech(wav=wav_data)
            return output["parallel_audio_feat"]
            
        

    def train(self, epoch):
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data["pose"]
            #print("tar_pose:", np.shape(tar_pose))
            tar_pose = (tar_pose*self.std_pose)+self.mean_pose
            pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 114)).cuda()
            pose_body[:, :, 0:57] = tar_pose[:, :, 18:75]
            pose_body[:, :, 57:114] =tar_pose[:, :, 84:141]


            audio = dict_data["audio"]
            #print("audio:", np.shape(audio))
            #print("pose:", np.shape(tar_pose))
            #audio_feat = self.speechclip(audio)
            # tar_beta = dict_data["beta"].cuda()
            # tar_trans = dict_data["trans"].cuda()
            tar_pose = tar_pose.cuda()
            pose_body = pose_body.cuda()  
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            # tar_exps = torch.zeros((bs, n, 100)).cuda()
            pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
            pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 38, 3),"XYZ")
            pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n, 38*6)
            t_data = time.time() - t_start
            
            self.opt.zero_grad()
            g_loss_final = 0
            net_out = self.model(pose_body)
            rec_pose = net_out["rec_pose"]
            # poses_feat = net_out["poses_feat"]
            # audio_feats = net_out["audio_feature"]
            
            # audio_feats_norm = audio_feats / audio_feats.norm(dim=0, keepdim=True)
            # poses_feat_norm = poses_feat / poses_feat.norm(dim=0, keepdim=True)
            # cos = self.cosine_sim(audio_feats_norm, poses_feat_norm)
            # cosine_loss = (1 - cos).mean()
            # #print("cosine_loss:", cosine_loss)
            # self.tracker.update_meter("cosine_loss", "train", cosine_loss.item())            
            # g_loss_final += cosine_loss * self.args.cosine_weight
 

            rec_pose = rec_pose.reshape(bs, n, 38, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            pose_body = rc.rotation_6d_to_matrix(pose_body.reshape(bs, n, 38, 6))
            loss_rec = self.vectices_loss(rec_pose, pose_body) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec_loss", "train", loss_rec.item())
            g_loss_final += loss_rec

            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], pose_body[:, 1:] - pose_body[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], pose_body[:, 2:] + pose_body[:, :-2] - 2 * pose_body[:, 1:-1]) * self.args.rec_weight
            self.tracker.update_meter("vel_loss", "train", velocity_loss.item())
            self.tracker.update_meter("acceleration_loss", "train", acceleration_loss.item())
            g_loss_final += velocity_loss 
            g_loss_final += acceleration_loss 
            

            # ---------------------- vae -------------------------- #
            if "VQVAE" in self.args.g_name:
                loss_embedding = net_out["embedding_loss"]
                g_loss_final += loss_embedding
                self.tracker.update_meter("embedding_loss", "train", loss_embedding.item())
            log = net_out["perplexity"]
            self.tracker.update_meter("kl", "train", log.item())

            g_loss_final.backward()
            self.tracker.update_meter("g_loss_final", "train", g_loss_final.item())
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"]
                tar_pose = (tar_pose*self.std_pose)+self.mean_pose
                audio = dict_data["audio"]
                audio  = audio.cuda()  
                #print("audio:", np.shape(audio))
                #print("pose:", np.shape(tar_pose))
                #audio_feat = self.speechclip(audio)            
                #tar_beta = dict_data["beta"].cuda()
                #tar_trans = dict_data["trans"].cuda()
                pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 114)).cuda()
                pose_body[:, :, 0:57] = tar_pose[:, :, 18:75]
                pose_body[:, :, 57:114] =tar_pose[:, :, 84:141]
                tar_pose = tar_pose.cuda() 
                pose_body = pose_body.cuda() 
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                #tar_exps = torch.zeros((bs, n, 100)).cuda()
                pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
                pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 38, 3),"XYZ")
                pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n, 38*6)
                t_data = time.time() - t_start 

                #self.opt.zero_grad()
                g_loss_final = 0
                net_out = self.model(pose_body)
                rec_pose = net_out["rec_pose"]
                # poses_feat = net_out["poses_feat"]
                # audio_feats = net_out["audio_feature"]
                
                # audio_feats_norm = audio_feats / audio_feats.norm(dim=-1, keepdim=True)
                # poses_feat_norm = poses_feat / poses_feat.norm(dim=-1, keepdim=True)
                # cos = self.cosine_sim(audio_feats_norm, poses_feat_norm)
                # cosine_loss = (1 - cos).mean()
                # self.tracker.update_meter("cosine_loss", "val", cosine_loss.item())
                # g_loss_final += cosine_loss         



                rec_pose = rec_pose.reshape(bs, n, 38, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                pose_body = rc.rotation_6d_to_matrix(pose_body.reshape(bs, n, 38, 6))
                
                loss_rec = self.vectices_loss(rec_pose, pose_body) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec_loss", "val", loss_rec.item())
                g_loss_final += loss_rec

                velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], pose_body[:, 1:] - pose_body[:, :-1]) * self.args.rec_weight
                acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], pose_body[:, 2:] + pose_body[:, :-2] - 2 * pose_body[:, 1:-1]) * self.args.rec_weight
                self.tracker.update_meter("vel_loss", "val", velocity_loss.item())
                self.tracker.update_meter("acceleration_loss", "val", acceleration_loss.item())
                g_loss_final += velocity_loss 
                g_loss_final += acceleration_loss 

                if "VQVAE" in self.args.g_name:
                    loss_embedding = net_out["embedding_loss"]
                    g_loss_final += loss_embedding
                    self.tracker.update_meter("embedding_loss", "val", loss_embedding.item())
                self.tracker.update_meter("g_loss_final", "val", g_loss_final.item())
                log = net_out["perplexity"]
                self.tracker.update_meter("kl", "val", log.item())
 
            self.val_recording(epoch)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        self.args.test_data_path = "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent/datasets/beat_cache/beat_english_15_141_origin/test/"
        self.args.test_demo = "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent/datasets/beat_cache/beat_english_15_141_origin/test/bvh_rot_vis/"
        test_seq_list = os.listdir(self.args.test_data_path+f"bvh_rot_vis/")
        test_seq_list.sort()
        align = 0 
        latent_out = []
        latent_ori = []
        t_start = 10
        t_end = 500
        self.model.eval()
        #self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                tar_pose = batch_data["pose"]
                #print("tar_pose:", np.shape(tar_pose))
                tar_pose = (tar_pose*self.std_pose)+self.mean_pose

                audio = batch_data["audio"]
                #print("audio:", np.shape(audio))
                #print("pose:", np.shape(tar_pose))
                #audio_feat = self.speechclip(audio)
                # tar_beta = dict_data["beta"].cuda()
                # tar_trans = dict_data["trans"].cuda()
                tar_pose = tar_pose.cuda()  
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                # tar_exps = torch.zeros((bs, n, 100)).cuda()
                # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                t_data = time.time() - t_start 
                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]


                pose_hand = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 114)).cuda()
                pose_hand[:, :, 0:57] = tar_pose[:, :, 18:75]
                pose_hand[:, :, 57:114] =tar_pose[:, :, 84:141]
                
                pose_hand = (torch.Tensor(pose_hand.reshape(-1, 3))/180)*np.pi
                pose_hand = rc.euler_angles_to_matrix(pose_hand.reshape(bs, n-remain, 38, 3),"XYZ")
                pose_hand = rc.matrix_to_rotation_6d(pose_hand).reshape(bs, n-remain, 38*6)



                pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 27)).cuda()
                pose_body[:, :, 0:18] = tar_pose[:, :, 0:18]
                pose_body[:, :, 18:27] =tar_pose[:, :, 75:84]
                
                pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
                pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n-remain, 9, 3),"XYZ")
                pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n-remain, 9*6)

                


                net_out_body = self.vq_model_body(pose_body)
                net_out_hand = self.vq_model_hand(pose_hand) 

                rec_pose_body = net_out_body["rec_pose"]
                rec_pose_body = rc.rotation_6d_to_matrix(rec_pose_body.reshape(bs, n-remain, 9, 6))
                rec_pose_body = rc.matrix_to_euler_angles(rec_pose_body, "XYZ")
                rec_pose_body = rec_pose_body * 180 / np.pi
                rec_pose_body = rec_pose_body.reshape(bs, n-remain, 9, 3).cpu().numpy() 
                rec_pose_body = rec_pose_body.reshape(bs, n-remain, 9*3) 




                rec_pose_hand = net_out_hand["rec_pose"]
                rec_pose_hand = rc.rotation_6d_to_matrix(rec_pose_hand.reshape(bs, n-remain, 38, 6))
                rec_pose_hand = rc.matrix_to_euler_angles(rec_pose_hand, "XYZ")
                rec_pose_hand = rec_pose_hand * 180 / np.pi
                rec_pose_hand = rec_pose_hand.reshape(bs, n-remain, 38, 3).cpu().numpy() 
                rec_pose_hand = rec_pose_hand.reshape(bs, n-remain, 38*3) 

                rec_pose_final = torch.zeros((tar_pose.shape[0], n-remain, 141)).numpy()


                print("rec_pose_body:", np.shape(rec_pose_body))
                print("rec_pose_hand:", np.shape(rec_pose_hand)) 


                rec_pose_final[:, :, 0:18] = rec_pose_body[:, :, 0:18]
                rec_pose_final[:, :, 75:84] = rec_pose_body[:, :, 18:27]

                rec_pose_final[:, :, 18:75] = rec_pose_hand[:, :, 0:57]
                rec_pose_final[:, :, 84:141] = rec_pose_hand[:, :, 57:114]



                rec_pose = rec_pose_final.reshape(-1, 141)
                print("rec_pose:", np.shape(rec_pose))
                # poses_feat = net_out["poses_feat"]
                # audio_feats = net_out["audio_feature"]
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
                

                # rec_pose = (rec_pose.reshape(-1, self.args.pose_dims) * self.std_pose) + self.mean_pose
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
                    for line_id in range(rec_pose.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(rec_pose[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')  
            
        # align_avg = align/len(self.test_loader)
        # logger.info(f"align score: {align_avg}")
        # srgr = self.srgr_calculator.avg()
        # logger.info(f"srgr score: {srgr}")
        # l1div = self.l1_calculator.avg()
        # logger.info(f"l1div score: {l1div}")
        #fgd_motion = data_tools.FIDCalculator.frechet_distance(latent_out_motion_all, latent_ori_all)
        #logger.info(f"fgd_motion: {fgd_motion}")
        data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.args.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        #self.test_recording(epoch) 