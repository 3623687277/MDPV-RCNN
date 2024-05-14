# def+DGCNN+颜色信息

from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
from ....utils import common_utils
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils

def get_graph_feature_w(x, k=10, idx=None):
    batch_size = 1
    #batch_size = args.batch_size,
    x_points=x[:,:3,:]
    x_color=x[:,3:,:]
    # 对 点
    num_points = x_points.size(2)
    x_points = x_points.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x_points, k=k)  # (batch_size, num_points, k)  [1, 2048, 10]
    device = torch.device('cuda')
    # print(idx.shape,'idx1')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points    #[1, 1, 1]
    # print(idx_base.shape,'idx_base')
    idx = idx + idx_base
    # print(idx.shape,'idx2')
    idx = idx.view(-1)
    _, num_dims, _ = x_points.size()

    x_points = x_points.transpose(2,1).contiguous()  #  [1,2048,64](batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # print(x.shape)
    feature = x_points.view(batch_size * num_points, -1)[idx, :]  #[2048,64]
    # print(feature.shape,'feature')
    feature_p = feature.view(batch_size, num_points, k, num_dims)  #[1,2048,10,64]
    # print(feature.shape,'featurefeature')
    x_points = x_points.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #[1,2048,10,64]
    # print(x.shape,'x')
    # 对颜色
    num_points = x_color.size(2)
    x_color = x_color.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x_color, k=k)  # (batch_size, num_points, k)  [1, 2048, 10]
    device = torch.device('cuda')
    # print(idx.shape,'idx1')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points    #[1, 1, 1]
    # print(idx_base.shape,'idx_base')
    idx = idx + idx_base
    # print(idx.shape,'idx2')
    idx = idx.view(-1)
    _, num_dims, _ = x_color.size()

    x_color = x_color.transpose(2,1).contiguous()  #  [1,2048,64](batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # print(x.shape)
    feature = x_color.view(batch_size * num_points, -1)[idx, :]  #[2048,64]
    # print(feature.shape,'feature')
    feature_c = feature.view(batch_size, num_points, k, num_dims)  #[1,2048,10,64]
    # print(feature.shape,'featurefeature')
    x_color = x_color.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #[1,2048,10,64]
    # print(x.shape,'x')
    feature = torch.cat((feature_p - x_points, x_points,feature_c-x_color), dim=3).permute(0, 3, 1, 2).contiguous()
    # print(feature.shape) #([32, 6, 1024, 20])
    return feature


def find_nearest_points(raw_points, feature, k=10):
    """
    进行knn的函数
    k代表 knn的k

    """
    num_features = raw_points.shape[-1]
    a = raw_points.cpu().numpy()
    b = feature.squeeze(dim=0)
    b = b.permute(1, 0).cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(a)

    distances, indices = nbrs.kneighbors(b)
    nearest_points = a[indices.reshape(-1)].reshape(-1, k, num_features)  #维度
    device = torch.device('cuda:0')
    nearest_points = torch.from_numpy(nearest_points).to(device)
    nearest_points = nearest_points.unsqueeze(dim=0)
    nearest_points = nearest_points.permute(0, 3, 1, 2).contiguous()
    return nearest_points

def extract_features(x, conv1, conv2, conv3, conv4, conv6, k=10):
    """
    DGCNN提取特征get_graph_feature
    传入的x维度是[1,6,2048,10]
    """
    # x = conv1(x)  # [1, 64, 2048, 10]
    x1 = x.max(dim=-1, keepdim=False)[0]  # [1, 6, 2048]
    x_w =get_graph_feature_w(x1,k)   #[1, 9, 2048]
    x_w = conv1(x_w)
    x_w = x_w.max(dim=-1, keepdim=False)[0]  #[1, 32, 2048]
    x = get_graph_feature(x_w, k=k)  # [1, 64, 2048, 10]
    x = conv2(x)  # [1, 64, 2048, 10]
    x2 = x.max(dim=-1, keepdim=False)[0]  # [1,64,2048]

    x = get_graph_feature(x2, k=k)  # [1, 128, 2048, 10]
    x = conv3(x)  # [1, 128, 2048, 10]
    x3 = x.max(dim=-1, keepdim=False)[0]  # [1, 128, 2048]

    x = get_graph_feature(x3, k=k)  # [1, 256, 2048, 10]
    x = conv4(x)  # [1, 256, 2048, 10]
    x4 = x.max(dim=-1, keepdim=False)[0]  # [1, 256, 2048]

    x = torch.cat((x_w, x2, x3, x4), dim=1)  # [1, 512, 2048]

    # x = get_graph_feature(x, k=k)
    # x = conv6(x)
    # x5 = x.max(dim=-1, keepdim=False)[0]  # [1, 32, 2048]
    x5 = x.unsqueeze(-1)    #[1,512,2048,1]
    x5 = conv6(x5)       #[1,32,2048,1]
    x5 = x5.squeeze(dim=3)
    x5 = x5.squeeze(dim=0)

    pooled_features = x5.permute(1, 0).contiguous()

    return pooled_features

def knn(x, k):  # 如果没有使用该参数，则默认使用最近邻数量为20
    # print("---")
    # print(x.shape) # ([32, 3, 1024])
    x1=x[:,0:3]
    # print(x.shape)
    inner = -2 * torch.matmul(x1.transpose(2, 1), x1)
    xx = torch.sum(x1 ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    # print(idx.shape) #([32, 1024, 20])
    return idx


def get_graph_feature(x, k=10, idx=None):
    """
    传入的x维度是[1,64,2048,10]
    """
    batch_size = 1
    #batch_size = args.batch_size,
    # x = x.max(dim=-1, keepdim=False)[0]  # [1, 128, 2048]

    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)  [1, 2048, 10]
    device = torch.device('cuda')
    # print(idx.shape,'idx1')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points    #[1, 1, 1]
    # print(idx_base.shape,'idx_base')
    idx = idx + idx_base
    # print(idx.shape,'idx2')
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  #  [1,2048,64](batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # print(x.shape)
    feature = x.view(batch_size * num_points, -1)[idx, :]  #[2048,64]
    # print(feature.shape,'feature')
    feature = feature.view(batch_size, num_points, k, num_dims)  #[1,2048,10,64]
    # print(feature.shape,'featurefeature')
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #[1,2048,10,64]
    # print(x.shape,'x')
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    # print(feature.shape) #([32, 6, 1024, 20])
    return feature
class Self_Attention(nn.Module):
    """
    Capture global context for feature refinement
    """
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.channel_in = in_dim
        self.query_conv = nn.Sequential(nn.Conv1d(in_dim, in_dim // 8, 1), nn.BatchNorm1d(in_dim//8), nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_dim, in_dim // 8, 1), nn.BatchNorm1d(in_dim//8), nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_dim, in_dim, 1), nn.BatchNorm1d(in_dim), nn.ReLU())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        proj_query = self.query_conv(x).permute(0, 2, 1)  # B x N x C
        proj_key = self.key_conv(x)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.gamma * out + x
        return out, attention


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


class DefVoxelSetAbstraction(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(6)
        self.bn8 = nn.BatchNorm2d(3)




        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(64)
        self.bn11 = nn.BatchNorm2d(32)

        self.bn5 = nn.BatchNorm1d(2048)

        self.conv1 = nn.Sequential(nn.Conv2d(9, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2 , 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(3, 6, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(12, 3, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.Sigmoid())



        self.conv9 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False),
                                    self.bn11,
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(512, 2048, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(2048 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 40)
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR
            mlps = SA_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [mlps[k][0]] + mlps[k]

            if src_name in self.model_cfg.DEF_SOURCE:
                cur_layer = pointnet2_stack_modules.StackSAModuleMSGAdapt(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
                )
            else:
                cur_layer = pointnet2_stack_modules.StackSAModuleMSG(
                radii=SA_cfg[src_name].POOL_RADIUS,
                nsamples=SA_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool',
                )

            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += sum([x[-1] for x in mlps])

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            mlps = SA_cfg['raw_points'].MLPS
            for k in range(len(mlps)):
                mlps[k] = [num_rawpoint_features - 3] + mlps[k]
            self.SA_rawpoints = pointnet2_stack_modules.StackSAModuleMSGGated(
                radii=SA_cfg['raw_points'].POOL_RADIUS,
                nsamples=SA_cfg['raw_points'].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method='max_pool'
            )
            c_in += sum([x[-1] for x in mlps])

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        self.pred_bev_offset = nn.Sequential(nn.Conv1d(num_bev_features, 2, kernel_size=1, bias=False), nn.Tanh())
        self.mod_bev_offset = nn.Conv1d(num_bev_features, 1, kernel_size=1, bias=False)
        self.attention = Self_Attention(self.num_point_features)

    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        x_idxs = (keypoints[:, :, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, :, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            cur_x_idxs = x_idxs[k]
            cur_y_idxs = y_idxs[k]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            offsets = self.pred_bev_offset(point_bev_features.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).contiguous().squeeze(0)
            mod = self.mod_bev_offset(point_bev_features.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1).contiguous().squeeze(0)
            offsets = torch.mul(offsets, mod)
            cur_x_idxs = cur_x_idxs + offsets[:, 0]
            cur_y_idxs = cur_y_idxs + offsets[:, 1]
            cur_x_idxs = torch.clamp(cur_x_idxs, 0, bev_features.shape[3])
            cur_y_idxs = torch.clamp(cur_y_idxs, 0, bev_features.shape[2])
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features.unsqueeze(dim=0))

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (B, N, C0)
        return point_bev_features

    def get_sampled_points(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:7]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                    cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'FastFPS':
                raise NotImplementedError
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints1 = self.get_sampled_points(batch_dict)  #【1,2048,6】
        # 坐标+颜色
        keypoints = keypoints1[:, : , 0:3]  # 坐标  [1,2048,3]
        # print(keypoints.shape,'keypoints')
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            point_bev_features = self.interpolate_from_bev_features(
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                bev_stride=batch_dict['spatial_features_stride']
            )
            # print(point_bev_features.shape,'point_bev_features')  #[1, 2048, 256]
            point_features_list.append(point_bev_features)

        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)   #2048,3
        new_xyz = new_xyz.contiguous()
        # new_xyz = keypoints.squeeze(dim=0)

        # print(new_xyz.shape,'new_xyz')
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            xyz = raw_points[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()

            # pooled_points, pooled_features = self.SA_rawpoints(
            #     xyz=xyz.contiguous(),
            #     xyz_batch_cnt=xyz_batch_cnt,
            #     new_xyz=new_xyz,
            #     new_xyz_batch_cnt=new_xyz_batch_cnt,
            #     features=raw_points[:, 1:5],
            # )
            # =======DGCNN=====
            feature = keypoints1.permute(0, 2, 1).contiguous()  #[1,6,2048]
            # 进行点的knn    [1, 6, 2048, 10]
            raw_points_knn=raw_points[:, 1:7]
            nearest_points=find_nearest_points(raw_points_knn, feature, k=10)
            # 颜色的knn
            xyz_c=raw_points[:, 4:7]  #原始点颜色信息
            feature_c=feature[:,-3:,:]  #关键点颜色信息
            nearest_points_c=find_nearest_points(xyz_c, feature_c, k=10)  #[1,3,2048,10]
            #卷积
            nearest_points_c=self.conv7(nearest_points_c)   #[1, 6, 2048, 10]
            feature_c=feature_c.unsqueeze(-1)
            feature_c=self.conv7(feature_c)  #[1, 6, 2048, 1]
            #改维度
            feature_c=feature_c.squeeze(-1) #[1, 6, 2048]
            nearest_points_c =nearest_points_c.max(dim=-1, keepdim=False)[0]  # [1,64,2048]
            x = torch.cat((feature_c, nearest_points_c), dim=1)#[1,12,2048]
            x=x.unsqueeze(-1)     #[1,12,2048,1]
            W = self.conv8(x).expand(-1, -1, -1, 10)    #[1, 3, 2048, 1]   这个卷积的激活函数是 Sigmoid 激活函数

            # print(feature_c.shape,'feature_c')
            # print(nearest_points_c.shape,'nearest_points_c')
            # print(W.shape,'W')
            #DGCNN的一些操作
            x = nearest_points   #[1, 6, 2048, 10]
            # 将颜色权重加进去
            x_points=x[:,:3,:,:]
            x_coler=x[:,3:,:,:]
            x_coler_xxx=W*x_coler
            x = torch.cat((x_coler_xxx, x_points), dim=1)  #[1,6,2048,10]

            # print(x.shape)
            pooled_features = extract_features(x, self.conv1, self.conv2, self.conv3, self.conv4, self.conv6)

            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            pooled_points, pooled_features = self.SA_layers[k](
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=batch_dict['multi_scale_3d_features'][src_name].features.contiguous(),
            )
            point_features_list.append(pooled_features.view(batch_size, num_keypoints, -1))

        point_features = torch.cat(point_features_list, dim=2)

        batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1)
        point_coords = torch.cat((batch_idx.view(-1, 1).float(), keypoints.view(-1, 3)), dim=1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1]))

        # point_features, attn = self.attention(point_features.unsqueeze(0).permute(0, 2, 1).contiguous())  # (B, C, N)
        # point_features = point_features.permute(0, 2, 1).squeeze(0).contiguous()

        batch_dict['point_features'] = point_features  # (BxN, C)
        batch_dict['point_coords'] = point_coords  # (BxN, 4)
        return batch_dict

