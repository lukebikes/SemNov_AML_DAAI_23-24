import sys
import os
import warnings
import numpy as np

sys.path.append(os.getcwd())
import os.path as osp
import time
import dgl
import dgl.geometry
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.utils import *
from utils.dist import *
 # noinspection PyUnresolvedReferences
from utils.data_utils import H5_Dataset
from datasets.modelnet import *
from datasets.scanobject import *
from models.classifiers import Classifier
from utils.ood_utils import get_confidence, eval_ood_sncore, iterate_data_odin, \
    iterate_data_energy, iterate_data_gradnorm, iterate_data_react, estimate_react_thres, print_ood_output, \
    get_penultimate_feats, get_network_output
import wandb
from base_args import add_base_args
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from models.common import convert_model_state, logits_entropy_loss
from models.ARPL_utils import Generator, Discriminator
from classifiers.common import train_epoch_cla, train_epoch_rsmix_exposure, train_epoch_cs


def get_args():
    parser = argparse.ArgumentParser("OOD on point clouds via contrastive learning")
    parser = add_base_args(parser)

    # experiment specific arguments
    parser.add_argument("--augm_set",
                        type=str, default="rw", help="data augmentation choice", choices=["st", "rw"])
    parser.add_argument("--grad_norm_clip",
                        default=-1, type=float, help="gradient clipping")
    parser.add_argument("--num_points",
                        default=1024, type=int, help="number of points sampled for each object view")
    parser.add_argument("--num_points_test",
                        default=2048, type=int, help="number of points sampled for each SONN object - only for testing")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="md-2-sonn-augmCorr")
    parser.add_argument("--wandb_proj", type=str, default="benchmark-3d-ood-cla")
    parser.add_argument("--loss", type=str, default="CE",
                        choices=["CE", "CE_ls", "cosface", "arcface", "subcenter_arcface", "ARPL", "cosine"],
                        help="Which loss to use for training. CE is default")
    parser.add_argument("--cs", action='store_true', help="Enable confusing samples for ARPL")
    parser.add_argument("--cs_gan_lr", type=float, default=0.0002, help="Confusing samples GAN lr")
    parser.add_argument("--cs_beta", type=float, default=0.1, help="Beta loss weight for CS")
    parser.add_argument("--save_feats", type=str, default=None, help="Path where to save feats of penultimate layer")

    # Adopt Corrupted data
    # this flag should be set also during evaluation if testing Synth->Real Corr/LIDAR Augmented models
    parser.add_argument("--corruption",
                        type=str, default=None, help="type of corrupted data (lidar,occlusion,all) - default is None")
    args = parser.parse_args()

    args.data_root = os.path.expanduser(args.data_root)
    args.tar1 = "none"
    args.tar2 = "none"

    if args.script_mode == 'eval':
        args.batch_size = 1

    return args


### data mgmt ###

def get_list_corr_data(opt, severity=None, split="train"):
    assert split in ['train', 'test']

    if opt.src == "SR1":
        prefix = "modelnet_set1"
    elif opt.src == "SR2":
        prefix = "modelnet_set2"
    else:
        raise ValueError(f"Expected SR source but received: {opt.src} ")

    print(f"get_list_corr_data for {prefix} - split {split}")

    # loads corrupted data
    if severity is None:
        severity = [1, 2, 3, 4]
    if opt.corruption == 'lidar' or opt.corruption == 'occlusion':
        print(f"loading {opt.corruption} data")
        root = osp.join(opt.data_root, "ModelNet40_corrupted", opt.corruption)
        file_names = [f"{root}/{prefix}_{split}_{opt.corruption}_sev" + str(i) + ".h5" for i in severity]
        print(f"corr list files: {file_names}\n")
    elif opt.corruption == 'all':
        print("loading both lidar and occlusion data")
        file_names = []
        root_lidar = osp.join(opt.data_root, "ModelNet40_corrupted", "lidar")
        file_names.extend([f"{root_lidar}/{prefix}_{split}_lidar_sev" + str(i) + ".h5" for i in severity])
        root_occ = osp.join(opt.data_root, "ModelNet40_corrupted", "occlusion")
        file_names.extend([f"{root_occ}/{prefix}_{split}_occlusion_sev" + str(i) + ".h5" for i in severity])
        print(f"corr list files: {file_names}\n")
    else:
        raise ValueError(f"Unknown corruption specified: {opt.corruption}")

    # augmentation mgmt
    if opt.script_mode.startswith("eval"):
        augm_set = None
    else:
        # synth -> real augm
        warnings.warn(f"Using RW augmentation set for corrupted data")
        augm_set = transforms.Compose([
            PointcloudToTensor(),
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()
        ])

    corrupted_datasets = []
    for h5_path in file_names:
        corrupted_datasets.append(H5_Dataset(h5_file=h5_path, num_points=opt.num_points, transforms=augm_set))

    return corrupted_datasets


# for training routine
def get_md_loaders(opt):
    assert opt.src.startswith('SR')
    ws, rank = get_ws(), get_rank()
    drop_last = not str(opt.script_mode).startswith('eval')

    if opt.augm_set == 'st':
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(lo=2 / 3, hi=3 / 2),
            AugmTranslate(translate_range=0.2)]
    elif opt.augm_set == 'rw':
        # transformation used for Synthetic->Real-World
        set_transforms = [
            PointcloudToTensor(),
            RandomSample(opt.num_points),
            AugmScale(),
            AugmRotate(axis=[0.0, 1.0, 0.0]),
            AugmRotatePerturbation(),
            AugmTranslate(),
            AugmJitter()]
    else:
        raise ValueError(f"Unknown augmentation set: {opt.augm_set}")

    print(f"Train transforms: {set_transforms}")
    train_transforms = transforms.Compose(set_transforms)

    train_data = ModelNet40_OOD(  # sampling performed as dataugm
        data_root=opt.data_root,
        train=True,
        num_points=10000,  # sampling as data augm
        class_choice=opt.src,  # modelnet40 or modelnet10,
        transforms=train_transforms
    )

    print(f"{opt.src} train_data len: {len(train_data)}")

    if opt.corruption is not None:
        # load corrupted datasets
        assert opt.augm_set == 'rw'
        l_corr_data = get_list_corr_data(opt)
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(train_data)
        train_data = torch.utils.data.ConcatDataset(l_corr_data)
        print(f"{opt.src} + corruption {opt.corruption} - train data len: {len(train_data)}")

    test_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=False,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    train_sampler = DistributedSampler(train_data, num_replicas=ws, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_data, num_replicas=ws, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=train_sampler, worker_init_fn=init_np_seed)
    test_loader = DataLoader(
        test_data, batch_size=opt.batch_size, drop_last=drop_last, num_workers=opt.num_workers,
        sampler=test_sampler, worker_init_fn=init_np_seed)
    return train_loader, test_loader

########################################################################
#The definition of the OpenShape models.
import torch
import torch.nn as nn
import numpy as np
import torch_redstone as rst
import torch.nn.functional as F
import dgl
import dgl.geometry
import sys
import os
from huggingface_hub import hf_hub_download
from einops import rearrange
sys.path.insert(0, '/content/drive/MyDrive/SemNov_AML_DAAI_23-24/OpenShape')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *extra_args, **kwargs):
        return self.fn(self.norm(x), *extra_args, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rel_pe = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, rel_pe = rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
    
def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


    
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    #print( 'shapexyz',xyz.shape)
    #print( 'xyz',xyz)
    
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[..., :1].repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #print( 'shapexyz',xyz.shape)
    #print( 'xyz',xyz)
    #print( 'npoints',npoint)
    #print('number we want',sample_point*batch_size)
    #return dgl.geometry.farthest_point_sampler(xyz, npoint)
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    # torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    # torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    # torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

 

class Attention(nn.Module):
     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rel_pe = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

     def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (torch.matmul(q, k.transpose(-1, -2)) + pe) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Projected(nn.Module):
    def __init__(self, ppat, proj) -> None:
        super().__init__()
        self.ppat = ppat
        self.proj = proj

    def forward(self, xyz, return_penultimate=True):
       
        #'return_penultimate' is unused 
        b, n, c = xyz.size()
        assert c == 3, "expected BN3 input pointcloud"
        xyz = xyz.permute(0,2,1)  # [bnc] => [bcn]
        # padded features
        features = torch.zeros(b, 3, n)  # torch.ones(b, 3, n) * 0.4
        features = features.to(xyz)
        features = torch.cat([xyz, features, p], dim=1)
        print("self.proj: " + str(self.proj))
        res = self.ppat(xyz, features, p)
        if self.proj is not None:
            res = self.proj(res)

        # #### added by me####
        # x=self.layer1(x)
        # penultimate_features = self.layer2(x)
        # res = self.layer3(penultimate_features)
        # if return_penultimate:
        #     return penultimate_features, res
        # else:
          

        return res

class PointPatchTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp, in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0) -> None:
        super().__init__()
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp, in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(nn.Conv1d(sa_dim + 3, dim, 1), rst.Lambda(lambda x: torch.permute(x, [0, 2, 1])), nn.LayerNorm([dim]))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe)
 
    def forward(self, xyz: torch.Tensor, return_penultimate=False):
        b, n, c = xyz.size()
        assert c == 3, "expected BN3 input pointcloud"
        xyz = xyz.permute(0,2,1)  # [bnc] => [bcn]
        # padded features
        features = torch.zeros(b, 3, n)  # torch.ones(b, 3, n) * 0.4
        features = features.to(xyz)
        features = torch.cat([xyz, features], dim=1)
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        centroids, feature = self.sa(xyz, features)
 
        x = self.lift(torch.cat([centroids, feature], dim=1))
 
        x = rst.supercat([self.cls_token, x], dim=-2)
        centroids = rst.supercat([centroids.new_zeros(1), centroids], dim=-1)
 
        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.transformer(x, centroid_delta)
 
        return x[:, 0]

#The definition of models

def module(state_dict: dict, name):
    print(state_dict)
    return {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith(name + '.')}


def G14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 512*3, 256, 384, 0.2, 64, 6),
        nn.Linear(512, 1280)
    )
    model_path = os.path.join('/content/drive/My Drive/SemNov_AML_DAAI_23-24/Pointbert_G14','model.pt')
    s = torch.load(model_path)
    dic = model.load_state_dict(module(s['state_dict'], 'module'))
    print(dic)
    return model

def L14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
        nn.Linear(512, 768)
    )
    model_path = os.path.join('/content/drive/My Drive/SemNov_AML_DAAI_23-24/Pointbert_L14','model.pt')
    s = torch.load(model_path)
    dic = model.load_state_dict(module(s, 'pc_encoder'))
    print(dic)
    return model
    

def B32(s):
    model = PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6)
    model_path = os.path.join('/content/drive/My Drive/SemNov_AML_DAAI_23-24/Pointbert_vitB32','model.pt')
    s = torch.load(model_path)
    dic = model.load_state_dict(module(s, 'pc_encoder'))
    print(dic)
    return model

    model_list = {
        "openshape-pointbert-vitb32-rgb": B32,
        "openshape-pointbert-vitl14-rgb": L14,
        "openshape-pointbert-vitg14-rgb": G14,
    }

# def B32(s):
#     model = Projected(
#         PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
#         None
#     )
#     model_path = os.path.join('/content/drive/My Drive/SemNov_AML_DAAI_23-24/Pointbert_vitB32','model.pt')
#     s = torch.load(model_path)
#     dic = model.load_state_dict(module(s, 'pc_encoder'))
#     print(dic)
#     return model

# model_list = {
#     "openshape-pointbert-vitb32-rgb": B32,
#     "openshape-pointbert-vitl14-rgb": L14,
#     "openshape-pointbert-vitg14-rgb": G14,
# }


def load_pc_encoder(name):
    model_path = os.path.join("/content/drive/My Drive/OpenShape/SemNov_AML_DAAI_23-24/", name, 'model.pt')
    s = torch.load(model_path, map_location='cpu')
    model = model_list[name](s).eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

    """Using predictions and labels, return a dictionary containing all novelty
    detection performance statistics.
    
    These metrics conform to how results are reported in the paper 'Enhancing The 
    Reliability Of Out-of-Distribution Image Detection In Neural Networks'.
    
        preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
        labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

        pos_label: label of the positive class (1 by default)
    """
    
    return {
        'fpr_at_95_tpr': fpr_at_95_tpr(predictions, labels, pos_label=pos_label),
        'detection_error': detection_error(predictions, labels, pos_label=pos_label),
        'auroc': auroc(predictions, labels, pos_label=pos_label),
        'aupr_in': aupr(predictions, labels, pos_label=pos_label),
        'aupr_out': aupr([-a for a in predictions], [1 - a for a in labels], pos_label=pos_label)
    }


###########################################################################

### for evaluation routine ###
def get_md_eval_loaders(opt):
    assert opt.script_mode.startswith("eval")
    if not str(opt.src).startswith('SR'):
        raise ValueError(f"Unknown modelnet src: {opt.src}")

    train_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=True,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    print(f"{opt.src} train data len: {len(train_data)}")

    # append corrupted data to train dataset
    if opt.corruption:
        l_corr_data = get_list_corr_data(opt)  # list of corrupted datasets
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(train_data)  # appending clean data to list corrupted datasets
        train_data = torch.utils.data.ConcatDataset(l_corr_data)  # concat Dataset
        print(f"Cumulative (clean+corrupted) train data len: {len(train_data)}")

    # test data (only clean samples)
    test_data = ModelNet40_OOD(
        data_root=opt.data_root,
        train=False,
        num_points=opt.num_points,
        class_choice=opt.src,
        transforms=None)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                              worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                             worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    return train_loader, test_loader


def get_md_react_val_loader(opt):
    print("Building React validation loader...")
    assert opt.script_mode.startswith("eval")
    if not str(opt.src).startswith('SR'):
        raise ValueError(f"Unknown modelnet src: {opt.src}")

    test_data = ModelNet40_OOD(data_root=opt.data_root, train=False, num_points=opt.num_points,
                               class_choice=opt.src, transforms=None)

    print(f"React Val - {opt.src} data len: {len(test_data)}")

    # append corrupted test data
    if opt.corruption:
        print(f"React Val - adding corrupted synthetic data: {opt.corruption}")
        l_corr_data = get_list_corr_data(opt, split='test')  # list of corrupted test datasets
        assert isinstance(l_corr_data, list)
        assert isinstance(l_corr_data[0], data.Dataset)
        l_corr_data.append(test_data)  # appending clean data to list corrupted datasets
        test_data = torch.utils.data.ConcatDataset(l_corr_data)  # concat Dataset
        print(f"React Val - cumulative (clean+corrupted) data len: {len(test_data)}\n")

    val_data = test_data  # note: modelnet synthetic are not used in synth->real eval
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                            worker_init_fn=init_np_seed, shuffle=False, drop_last=False)
    return val_loader

###########################################################################
# def train(opt, config,s):
#     raise NotImplementedError("Training not implemented for this script")


def eval_ood_md2sonn(opt, config, s):
    print(f"Arguments: {opt}")
    set_random_seed(opt.seed)

    dataloader_config = {
        'batch_size': opt.batch_size, 'drop_last': False, 'shuffle': False,
        'num_workers': opt.num_workers, 'sampler': None, 'worker_init_fn': init_np_seed}

    # whole evaluation is done on ScanObject RW data
    sonn_args = {
        'data_root': opt.data_root,
        'sonn_split': opt.sonn_split,
        'h5_file': opt.sonn_h5_name,
        'split': 'all',  # we use both training (unused) and test samples during evaluation
        'num_points': opt.num_points_test,  # default: use all 2048 sonn points to avoid sampling randomicity
        'transforms': None  # no augmentation applied at inference time
    }

    train_loader, _ = get_md_eval_loaders(opt)
    if opt.src == 'SR1':
        print("Src is SR1\n")
        id_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet1", **sonn_args), **dataloader_config)
        ood1_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet2", **sonn_args), **dataloader_config)
    elif opt.src == 'SR2':
        print("Src is SR2\n")
        id_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet2", **sonn_args), **dataloader_config)
        ood1_loader = DataLoader(ScanObject(class_choice="sonn_2_mdSet1", **sonn_args), **dataloader_config)
    else:
        raise ValueError(f"OOD evaluation - wrong src: {opt.src}")

    # second SONN out-of-distribution set is common to both SR1 and SR2 sources
    # these are the samples from SONN categories with poor mapping to ModelNet categories
    ood2_loader = DataLoader(ScanObject(class_choice="sonn_ood_common", **sonn_args), **dataloader_config)

    classes_dict = eval(opt.src)
    n_classes = len(set(classes_dict.values()))
    
    # Build Openshape Model
    model_path = os.path.join('/content/drive/My Drive/SemNov_AML_DAAI_23-24/Pointbert_vitB32','model.pt')
    s = torch.load(model_path)
    model = B32(s)
    state_dict = module(s, 'pc_encoder')
    model.load_state_dict(state_dict)
    ckt_weights = sanitize_model_dict(state_dict)
    print("Load weights: ", model.load_state_dict(ckt_weights, strict=True))
    print(f"Model params count: {count_parameters(model) / 1000000 :.4f} M")
    model = model.cuda().eval()

    # FEATURES EVALUATION
    eval_OOD_with_feats(model, train_loader, id_loader, ood1_loader, ood2_loader, save_feats=opt.save_feats)
    return


def eval_OOD_with_feats(model, train_loader, src_loader, tar1_loader, tar2_loader, save_feats=None):
    from knn_cuda import KNN
    knn = KNN(k=1, transpose_mode=True)

    print("\n" + "#" * 80)
    print("Computing OOD metrics with distance from train features...")

    # extract penultimate features, compute distances
    train_feats, train_labels = get_penultimate_feats(model, train_loader)
    src_feats, src_labels = get_penultimate_feats(model, src_loader)
    tar1_feats, tar1_labels = get_penultimate_feats(model, tar1_loader)
    tar2_feats, tar2_labels = get_penultimate_feats(model, tar2_loader)
    train_labels = train_labels.cpu().numpy()

    labels_set = set(train_labels)
    prototypes = torch.zeros((len(labels_set), train_feats.shape[1]), device=train_feats.device)
    for idx, lbl in enumerate(labels_set):
        mask = train_labels == lbl
        prototype = train_feats[mask].mean(0)
        prototypes[idx] = prototype

    if save_feats is not None:
        if isinstance(train_loader.dataset, ModelNet40_OOD):
            labels_2_names = {v: k for k, v in train_loader.dataset.class_choice.items()}
        else:
            labels_2_names = {}

        output_dict = {}
        output_dict["labels_2_names"] = labels_2_names
        output_dict["train_feats"], output_dict["train_labels"] = train_feats.cpu(), train_labels
        output_dict["id_data_feats"], output_dict["id_data_labels"] = src_feats.cpu(), src_labels
        output_dict["ood1_data_feats"], output_dict["ood1_data_labels"] = tar1_feats.cpu(), tar1_labels
        output_dict["ood2_data_feats"], output_dict["ood2_data_labels"] = tar2_feats.cpu(), tar2_labels
        torch.save(output_dict, save_feats)
        print(f"Features saved to {save_feats}")

    ################################################
    print("Euclidean distances in a non-normalized space:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(train_feats.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(train_feats.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    # OOD tar2
    tar2_dist, _ = knn(train_feats.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_scores = 1 / tar2_dist

    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[src_pred, None, None],  # [src_pred, None, None],
        labels_list=[src_labels, None, None],  # [src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )

    print("\nEuclidean distances with prototypes:")
    # eucl distance in a non-normalized space
    src_dist, src_ids = knn(prototypes.unsqueeze(0), src_feats.unsqueeze(0))
    src_dist = src_dist.squeeze().cpu()
    src_ids = src_ids.squeeze().cpu()  # index of nearest training sample
    src_scores = 1 / src_dist
    src_pred = np.asarray([train_labels[i] for i in src_ids])  # pred is label of nearest training sample

    # OOD tar1
    tar1_dist, _ = knn(prototypes.unsqueeze(0), tar1_feats.unsqueeze(0))
    tar1_dist = tar1_dist.squeeze().cpu()
    tar1_scores = 1 / tar1_dist

    # OOD tar2
    tar2_dist, _ = knn(prototypes.unsqueeze(0), tar2_feats.unsqueeze(0))
    tar2_dist = tar2_dist.squeeze().cpu()
    tar2_scores = 1 / tar2_dist

    eval_ood_sncore(
        scores_list=[src_scores, tar1_scores, tar2_scores],
        preds_list=[src_pred, None, None],
        labels_list=[src_labels, None, None],
        src_label=1  # confidence should be higher for ID samples
    )


def main():
    args = get_args()
    config = load_yaml(args.config)

    if args.script_mode.startswith('train'):
        # launch trainer
        print("training...")
        assert args.checkpoints_dir is not None and len(args.checkpoints_dir)
        assert args.exp_name is not None and len(args.exp_name)
        args.log_dir = osp.join(args.checkpoints_dir, args.exp_name)
        args.tb_dir = osp.join(args.checkpoints_dir, args.exp_name, "tb-logs")
        args.models_dir = osp.join(args.checkpoints_dir, args.exp_name, "models")
        args.backup_dir = osp.join(args.checkpoints_dir, args.exp_name, "backup-code")
        train(args, config)
    else:
        # eval Modelnet -> SONN
        assert args.ckpt_path is not None and len(args.ckpt_path)
        print("out-of-distribution eval - Modelnet -> SONN ..")
        eval_ood_md2sonn(args, config , 's')


if __name__ == '__main__':
    main()
