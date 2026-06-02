import torch.nn as nn
from .pv_utils import MeanVFE, VoxelBackBone8x, VoxelSetAbstraction, HeightCompression# , BEVCompression , Conv2DCollapse
# from .data_processor import DataProcessor
import numpy as np


class pv_backbone(nn.Module):
    """
    Modified from https://github.com/open-mmlab/OpenPCDet
    """
    def __init__(self, model_cfg):
        super().__init__()
        
        voxel_size = model_cfg.DATA_PROCESSOR[-1]['VOXEL_SIZE']
        point_cloud_range = np.asarray(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / np.array(voxel_size)
        # bev_channel = model_cfg.BEV_CHANNEL
        input_channels = len(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list)

        # self.data_processor = DataProcessor(model_cfg.DATA_CONFIG.DATA_PROCESSOR, point_cloud_range, training, 6)

        self.vfe = MeanVFE()
        self.backbone_3d = VoxelBackBone8x(model_cfg, input_channels, grid_size.astype(np.int64))

        self.to_bev = HeightCompression(model_cfg.MODEL.MAP_TO_BEV)
        # self.to_bev = BEVCompression(model_cfg.MODEL.MAP_TO_BEV)
        # self.to_bev = Conv2DCollapse(model_cfg.MODEL.MAP_TO_BEV)

        self.vsa = VoxelSetAbstraction(model_cfg=model_cfg.MODEL.PFE, 
                                       voxel_size=voxel_size, 
                                       point_cloud_range=point_cloud_range,
                                       num_bev_features=model_cfg.MODEL.MAP_TO_BEV.BEV_OUTPUT_FEATURES,
                                       num_rawpoint_features=input_channels)

    # def prepare_input(self, point_cloud):
    #     batch_dict = {'points': point_cloud.cpu().numpy(), 'use_lead_xyz': True}
    #     batch_dict = self.data_processor.forward(batch_dict)
    #     return batch_dict

    def forward(self, batch_dict):

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.to_bev(batch_dict)
        # batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.vsa(batch_dict)
        
        return batch_dict
        # return batch_dict['point_coords'], batch_dict['point_features'], batch_dict['inds']

