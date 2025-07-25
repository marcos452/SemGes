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
# from torch.utils.tensorboard import SummaryWriter
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
# from avssl.model import KWClip_GeneralTransformer
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
        self.joints = 43
        self.tracker = other_tools.EpochTracker(["rec_loss", "vel_loss", "ver", "embedding_loss", "kl", "acceleration_loss","cosine_loss","g_loss_final"], [False, False, False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        self.cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        #self.speechclip_model = self.load_speechclip_model()
        mean_dir_vec= np.array( [-0.0737964, -0.9968923, -0.1082858,  0.9111595,  0.2399522, -0.102547 , -0.8936886,  0.3131501, -0.1039348,  0.2093927, 0.958293 ,  0.0824881, -0.1689021, -0.0353824, -0.7588258, -0.2794763, -0.2495191, -0.614666 , -0.3877234,  0.005006 , -0.5301695, -0.5098616,  0.2257808,  0.0053111, -0.2393621, -0.1022204, -0.6583039, -0.4992898,  0.1228059, -0.3292085, -0.4753748,  0.2132857,  0.1742853, -0.2062069,  0.2305175, -0.5897119, -0.5452555,  0.1303197, -0.2181693, -0.5221036, 0.1211322,  0.1337591, -0.2164441,  0.0743345, -0.6464546, -0.5284583,  0.0457585, -0.319634 , -0.5074904,  0.1537192, 0.1365934, -0.4354402, -0.3836682, -0.3850554, -0.4927187, -0.2417618, -0.3054556, -0.3556116, -0.281753 , -0.5164358, -0.3064435,  0.9284261, -0.067134 ,  0.2764367,  0.006997 , -0.7365526,  0.2421269, -0.225798 , -0.6387642,  0.3788997, 0.0283412, -0.5451686,  0.5753376,  0.1935219,  0.0632555, 0.2122412, -0.0624179, -0.6755542,  0.5212831,  0.1043523, -0.345288 ,  0.5443628,  0.128029 ,  0.2073687,  0.2197118, 0.2821399, -0.580695 ,  0.573988 ,  0.0786667, -0.2133071, 0.5532452, -0.0006157,  0.1598754,  0.2093099,  0.124119, -0.6504359,  0.5465003,  0.0114155, -0.3203954,  0.5512083, 0.0489287,  0.1676814,  0.4190787, -0.4018607, -0.3912126, 0.4841548, -0.2668508, -0.3557675,  0.3416916, -0.2419564, -0.5509825,  0.0485515, -0.6343101, -0.6817347, -0.4705639, -0.6380668,  0.4641643,  0.4540192, -0.6486361,  0.4604001, -0.3256226,  0.1883097,  0.8057457,  0.3257385,  0.1292366, 0.815372])
        mean_pose = np.array( [-0.0046788, -0.5397806,  0.007695 , -0.0171913, -0.7060388,-0.0107034,  0.1550734, -0.6823077, -0.0303645, -0.1514748,   -0.6819547, -0.0268262,  0.2094328, -0.469447 , -0.0096073,   -0.2318253, -0.4680838, -0.0444074,  0.1667382, -0.4643363,   -0.1895118, -0.1648597, -0.4552845, -0.2159728,  0.1387546,   -0.4859474, -0.2506667,  0.1263615, -0.4856088, -0.2675801,   0.1149031, -0.4804542, -0.267329 ,  0.1414847, -0.4727709,   -0.2583424,  0.1262482, -0.4686185, -0.2682536,  0.1150217,   -0.4633611, -0.2640182,  0.1475897, -0.4415648, -0.2438853,   0.1367996, -0.4383164, -0.248248 ,  0.1267222, -0.435534 ,   -0.2455436,  0.1455485, -0.4557491, -0.2521977,  0.1305471,   -0.4535603, -0.2611591,  0.1184687, -0.4495366, -0.257798 ,   0.1451682, -0.4802511, -0.2081622,  0.1301337, -0.4865308,   -0.2175783,  0.1208341, -0.4932623, -0.2311025, -0.1409241,-0.4742868, -0.2795303, -0.1287992, -0.4724431, -0.2963172,-0.1159225, -0.4676439, -0.2948754, -0.1427748, -0.4589126,-0.2861245, -0.126862 , -0.4547355, -0.2962466, -0.1140265,-0.451308 , -0.2913815, -0.1447202, -0.4260471, -0.2697673,-0.1333492, -0.4239912, -0.2738043, -0.1226859, -0.4238346,-0.2706725, -0.1446909, -0.440342 , -0.2789209, -0.1291436,-0.4391063, -0.2876539, -0.1160435, -0.4376317, -0.2836147,-0.1441438, -0.4729031, -0.2355619, -0.1293268, -0.4793807,-0.2468831, -0.1204146, -0.4847246, -0.2613876, -0.0056085,-0.9224338, -0.1677302, -0.0352157, -0.963936 , -0.1388849,0.0236298, -0.9650772, -0.1385154, -0.0697098, -0.9514691,-0.055632 ,  0.0568838, -0.9565502, -0.0567985])
        # self.mean_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_mean.npy")
        # self.std_pose = np.load(self.args.root_path +"datasets/beat_cache/beat_english_15_141_origin/train/bvh_rot/bvh_std.npy")

        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 512
        self.args.vae_layer = 2

        self.args.vae_test_dim = 228 # need to be changed
        self.vq_model_hand = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hand, "/data/scratch/final_ted/SemDiffusion_latent_4/beat_pre/pretrained_vq/hand/last_1900.bin", args.e_name)
        self.vq_model_hand.eval()

        self.args.vae_test_dim = 54 # need to be changed
        self.vq_model_body = getattr(vq_model_module, "VQVAEConvZero_1")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_body, "/data/scratch/final_ted/SemDiffusion_latent_4/beat_pre/pretrained_vq/body/last_1900.bin", args.e_name)
        self.vq_model_body.eval()



    # def load_speechclip_model(self):
    #     device = torch.device("cuda")
    #     model_fp = "/data/clusterfs/mld/users/lanliu/SIGGRAGH_2024/Final/SemDiffusion_latent_3/icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+/model.ckpt"
    #     model = KWClip_GeneralTransformer.load_from_checkpoint(model_fp)
    #     model.to(device)
    #     model.eval()
    #     return model

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
            # tar_pose = (tar_pose*self.std_pose)+self.mean_pose
            pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 90)).cuda()
            pose_body = tar_pose[:, :, 24:114]


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
            # pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
            pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 30, 3),"XYZ")
            pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n, 30*6)
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
 

            rec_pose = rec_pose.reshape(bs, n, 30, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            pose_body = rc.rotation_6d_to_matrix(pose_body.reshape(bs, n, 30, 6))
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
                # tar_pose = (tar_pose*self.std_pose)+self.mean_pose
                audio = dict_data["audio"]
                audio  = audio.cuda()  
                #print("audio:", np.shape(audio))
                #print("pose:", np.shape(tar_pose))
                #audio_feat = self.speechclip(audio)            
                #tar_beta = dict_data["beta"].cuda()
                #tar_trans = dict_data["trans"].cuda()
                pose_body = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 90)).cuda()
                print("tar_pose:", np.shape(tar_pose))
                pose_body = tar_pose[:, :, 24:114]
                tar_pose = tar_pose.cuda() 
                pose_body = pose_body.cuda() 
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                #tar_exps = torch.zeros((bs, n, 100)).cuda()
                # pose_body = (torch.Tensor(pose_body.reshape(-1, 3))/180)*np.pi
                pose_body = rc.euler_angles_to_matrix(pose_body.reshape(bs, n, 30, 3),"XYZ")
                pose_body = rc.matrix_to_rotation_6d(pose_body).reshape(bs, n, 30*6)
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



                rec_pose = rec_pose.reshape(bs, n, 30, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                pose_body = rc.rotation_6d_to_matrix(pose_body.reshape(bs, n, 30, 6))
                
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

                rec_pose_final = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 141)).numpy()


                print("rec_pose_body:", np.shape(rec_pose_body))
                print("rec_pose_hand:", np.shape(rec_pose_hand)) 


                rec_pose_final[:, :, 0:18] = rec_pose_body[:, :, 0:18]
                rec_pose_final[:, :, 75:84] = rec_pose_body[:, :, 18:27]

                rec_pose_final[:, :, 18:75] = rec_pose_hand[:, :, 0:57]
                rec_pose_final[:, :, 84:141] = rec_pose_hand[:, :, 57:114]



                rec_pose = rec_pose_final.reshape(-1, self.args.pose_dims)
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