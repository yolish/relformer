import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer.TransformerEncoder import TransformerEncoder
from models.transformer.PositionEncoder import PositionEmbeddingLearnedWithPoseToken, PositionEmbeddingLearned
from models.transformer.Transformer import Transformer
import torchvision
from util.qcqp_layers import convert_Avec_to_A

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

class DeltaRegressor(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, init_stride):
        super().__init__()
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvBnReLU(in_channels, hidden_channels, kernel_size=3, stride=init_stride, pad=1)
        self.conv2 = ConvBn(hidden_channels, hidden_channels, kernel_size=3, stride=1, pad=1)
        self.head = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, delta_img):
        delta_img = self.conv2(self.conv1(delta_img))
        delta_z = self.avg_pooling_2d(delta_img).flatten(start_dim=1)
        delta = self.head(delta_z)
        return delta


class Relformer(nn.Module):

    def __init__(self, hidden_dim, out_dim, head_dim=None, ):
        super().__init__()

        self.transformer_encoder = TransformerEncoder({"hidden_dim":hidden_dim})
        if head_dim is None:
            head_dim = hidden_dim
        self.position_encoder = PositionEmbeddingLearnedWithPoseToken(hidden_dim//2)
        self.rel_pose_token = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
        self.head = nn.Sequential(nn.Linear(head_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, delta_img, prior_z=None, return_delta_z=False):
        # make into a sequence and append token
        delta_seq = delta_img.flatten(start_dim=2).permute(2,0,1)
        batch_size = delta_img.shape[0]
        rel_token = self.rel_pose_token.unsqueeze(1).repeat(1, batch_size, 1)
        # S x B x D
        delta_seq = torch.cat([rel_token, delta_seq])

        # prepare position encoding
        token_posisition_encoding, activation_posistion_encoding = self.position_encoder(delta_img)
        position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                               activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

        # regress latent relative with transformer encoder
        delta_z = self.transformer_encoder(delta_seq, position_encoding)[:, 0, :]
        if prior_z is not None:
            delta_z = torch.cat((delta_z, prior_z), dim=1)

        delta = self.head(delta_z)
        if return_delta_z:
            return delta, delta_z
        else:
            return delta


##### Start: code from - https://github.com/utiasSTARS/bingham-rotation-learning
def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)  # Bx10 -> Bx4x4
    _, evs = torch.symeig(A, eigenvectors=True)
    q = evs[:, :, 0].squeeze()
    return q
### End



class DeltaNet(nn.Module):

    def __init__(self, config, backbone_path):
        super().__init__()

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320}
        self.reductions = config.get("reduction")
        self.delta_img_dim = config.get("delta_img_dim")
        self.hidden_dim = config.get("hidden_dim")
        self.proj_before_concat = config.get("proj_before_concat")
        rot_dim = config.get("rot_dim")
        if config.get("rot_repr_type") == '10d':
            rot_dim = 10
            self.convert_to_quat = True
        else:
            self.convert_to_quat = False

        if self.proj_before_concat:
            self.proj_x = nn.Conv2d(reduction_map[self.reductions[0]], self.hidden_dim, kernel_size=1)
            self.proj_q = nn.Conv2d(reduction_map[self.reductions[1]], self.hidden_dim, kernel_size=1)
            self.hidden_dim = self.hidden_dim * 2
        else:
            self.proj_with_transformer = config.get("proj_with_transformer")
            if self.proj_with_transformer:
                self.tr_x = Transformer({"hidden_dim":reduction_map[self.reductions[0]]})
                self.tr_q = Transformer({"hidden_dim": reduction_map[self.reductions[1]]})
                self.position_encoder_x = PositionEmbeddingLearned(reduction_map[self.reductions[0]] // 2)
                self.position_encoder_q = PositionEmbeddingLearned(reduction_map[self.reductions[1]] // 2)
                self.proj_x = nn.Conv2d(reduction_map[self.reductions[0]], self.hidden_dim, kernel_size=1)
                self.proj_q = nn.Conv2d(reduction_map[self.reductions[1]], self.hidden_dim, kernel_size=1)
            else:
                self.proj_x = nn.Conv2d(reduction_map[self.reductions[0]] * 2, self.hidden_dim, kernel_size=1)
                self.proj_q = nn.Conv2d(reduction_map[self.reductions[1]] * 2, self.hidden_dim, kernel_size=1)

        reg_type = config.get("regressor_type")
        self.estimate_position_with_prior = config.get("position_with_prior")
        self.estimate_rotation_with_prior = config.get("rotation_with_prior")
        if reg_type == "transformer":
            if self.estimate_position_with_prior:
                head_dim_x = self.hidden_dim*2
            else:
                head_dim_x = self.hidden_dim
            if self.estimate_rotation_with_prior:
                head_dim_q = self.hidden_dim*2
            else:
                head_dim_q = self.hidden_dim
            self.delta_reg_x = Relformer(self.hidden_dim, 3, head_dim_x)
            self.delta_reg_q = Relformer(self.hidden_dim, rot_dim, head_dim_q)
        elif reg_type == "conv":
            self.delta_reg_x = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, 3, 2)
            self.delta_reg_q = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, rot_dim, 1)
        else:
            raise NotImplementedError(reg_type)

        self._reset_parameters()

        # load backbone after param reset
        self.double_backbone = config.get("double_backbone")
        if self.double_backbone:
            self.backbone_x = torch.load(backbone_path)
            self.backbone_q = torch.load(backbone_path)
        else:
            self.backbone = torch.load(backbone_path)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')

        if self.double_backbone:
            query_endpoints_x = self.backbone_x.extract_endpoints(query)[self.reductions[0]]
            ref_endpoints_x = self.backbone_x.extract_endpoints(ref)[self.reductions[0]]
            query_endpoints_q = self.backbone_q.extract_endpoints(query)[self.reductions[1]]
            ref_endpoints_q = self.backbone_q.extract_endpoints(ref)[self.reductions[1]]
        else:
            query_endpoints = self.backbone.extract_endpoints(query)
            ref_endpoints = self.backbone.extract_endpoints(ref)
            query_endpoints_x = query_endpoints[self.reductions[0]]
            ref_endpoints_x = ref_endpoints[self.reductions[0]]
            query_endpoints_q = query_endpoints[self.reductions[1]]
            ref_endpoints_q = ref_endpoints[self.reductions[1]]

        if self.proj_before_concat:
            query_endpoints_x = self.proj_x(query_endpoints_x)  # N X hidden_dim x H_R x W_R
            ref_endpoints_x = self.proj_x(ref_endpoints_x)  # N X hidden_dim x H_R x W_R
            delta_img_x = torch.cat((query_endpoints_x, ref_endpoints_x), dim=1)
            query_endpoints_q = self.proj_q(query_endpoints_q)  # N X hidden_dim x H_R x W_R
            ref_endpoints_q = self.proj_q(ref_endpoints_q)  # N X hidden_dim x H_R x W_R
            delta_img_q = torch.cat((query_endpoints_q, ref_endpoints_q), dim=1)
        else:
            if self.proj_with_transformer:
                query_pos_encdoing_x = self.position_encoder_x(query_endpoints_x).flatten(start_dim=2).permute(2, 0, 1)
                query_pos_encdoing_q = self.position_encoder_q(query_endpoints_q).flatten(start_dim=2).permute(2, 0, 1)
                ref_pos_encdoing_x = self.position_encoder_x(ref_endpoints_x).flatten(start_dim=2).permute(2, 0, 1)
                ref_pos_encdoing_q = self.position_encoder_q(ref_endpoints_q).flatten(start_dim=2).permute(2, 0, 1)
                query_seq_x = query_endpoints_x.flatten(start_dim=2).permute(2, 0, 1)
                query_seq_q = query_endpoints_q.flatten(start_dim=2).permute(2, 0, 1)
                ref_seq_x = ref_endpoints_x.flatten(start_dim=2).permute(2, 0, 1)
                ref_seq_q = ref_endpoints_q.flatten(start_dim=2).permute(2, 0, 1)

                # replace concat with transformer
                b, c, h, w = query_endpoints_x.shape
                delta_img_x = self.tr_x(query_seq_x, ref_seq_x, query_pos_encdoing_x, ref_pos_encdoing_x).transpose(1,2).reshape((b,c,h,w))
                b, c, h, w = query_endpoints_q.shape
                delta_img_q = self.tr_q(query_seq_q, ref_seq_q, query_pos_encdoing_q, ref_pos_encdoing_q).transpose(1,2).reshape((b,c,h,w))

            else:
                # delta_img_x is N x 2D x H_R x W_R
                delta_img_x = torch.cat((query_endpoints_x, ref_endpoints_x), dim=1)
                # delta_img_q is N x 2D x H_R x W_R
                delta_img_q = torch.cat((query_endpoints_q, ref_endpoints_q), dim=1)

            delta_img_x = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
            delta_img_q = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R

        if self.estimate_position_with_prior:
            delta_q, delta_z = self.delta_reg_q(delta_img_q, return_delta_z=True)
            delta_x = self.delta_reg_x(delta_img_x, prior_z=delta_z)
        elif self.estimate_rotation_with_prior:
            delta_x, delta_z = self.delta_reg_x(delta_img_x, return_delta_z=True)
            delta_q = self.delta_reg_q(delta_img_q, prior_z=delta_z)
        else:
            delta_x = self.delta_reg_x(delta_img_x)
            delta_q = self.delta_reg_q(delta_img_q)
        if self.convert_to_quat:
            delta_q = A_vec_to_quat(delta_q)
        delta_p = torch.cat((delta_x, delta_q), dim=1)
        return {"rel_pose": delta_p}


class DeltaNetEquiv(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_type = config.get("backbone_type")
        if backbone_type == "resnet50":
            backbone_fn = torchvision.models.resnet50
            self.hidden_dim = 2048
        elif backbone_type == "resnet34":
            backbone_fn = torchvision.models.resnet34
            self.hidden_dim = 512
        else:
            raise NotImplementedError(backbone_type)

        self.delta_reg_x = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, 3, 2)
        self.delta_reg_q = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, 4, 1)
        self.proj_x = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
        self.proj_q = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)

        self._reset_parameters()
        backbone = backbone_fn(pretrained=True)
        self.backbone = nn.Sequential(*(list(backbone.children())[:-2]))

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')
        query_endpoints = self.backbone(query) # n x 512 x 7 x 7 for resnet34
        ref_endpoints = self.backbone(ref)
        query_endpoints_x = query_endpoints
        ref_endpoints_x = ref_endpoints
        query_endpoints_q = query_endpoints
        ref_endpoints_q = ref_endpoints

        # delta_img_x is N x 2D x H_R x W_R
        delta_img_x = torch.cat((query_endpoints_x, ref_endpoints_x), dim=1)
        # delta_img_q is N x 2D x H_R x W_R
        delta_img_q = torch.cat((query_endpoints_q, ref_endpoints_q), dim=1)

        delta_img_x = self.proj_x(delta_img_x)  # N X hidden_dim x H_R x W_R
        delta_img_q = self.proj_q(delta_img_q)  # #N X hidden_dim x H_R x W_R

        delta_x = self.delta_reg_x(delta_img_x)
        delta_q = self.delta_reg_q(delta_img_q)
        delta_p = torch.cat((delta_x, delta_q), dim=1)
        return {"rel_pose": delta_p}


class BaselineRPR(nn.Module):
    # efficientnet + avg. pooling - "traditional RPR"
    def __init__(self, backbone_path):
        super().__init__()

        hidden_dim = 1280*2
        self.head_x = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 3))
        self.head_q = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 4))

        self._reset_parameters()
        self.backbone = torch.load(backbone_path)
        self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        query = self.backbone.extract_features(data.get('query'))
        ref = self.backbone.extract_features(data.get('ref'))

        z_query = self.avg_pooling_2d(query).flatten(start_dim=1)
        z_ref = self.avg_pooling_2d(ref).flatten(start_dim=1)

        z = torch.cat((z_query, z_ref), dim=1)
        delta_x = self.head_x(z)
        delta_q = self.head_q(z)
        delta_p = torch.cat((delta_x, delta_q), dim=1)
        return {"rel_pose": delta_p}


class TDeltaNet(nn.Module):

    def __init__(self, config, backbone_path):
        super().__init__()

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320}
        self.reduction = config.get("reduction")
        self.hidden_dim = config.get("hidden_dim")
        self.type = config.get("type")
        if self.type == "rotation":
            rot_dim = config.get("rot_dim")
            if config.get("rot_repr_type") == '10d':
                rot_dim = 10
                self.convert_to_quat = True
            else:
                self.convert_to_quat = False
            out_dim = rot_dim
        else:
            out_dim = 3

        self.baseline = False
        if self.reduction is None:
            self.baseline = True
            hidden_dim = 1280 * 2
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
            self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
        else:
            self.proj = nn.Conv2d(reduction_map[self.reduction[0]] * 2, self.hidden_dim, kernel_size=1)
            reg_type = config.get("regressor_type")
            if reg_type == "transformer":
                self.delta_reg = Relformer(self.hidden_dim, out_dim, self.hidden_dim)
            elif reg_type == "conv":
                self.delta_reg = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, out_dim, 1)
            else:
                raise NotImplementedError(reg_type)

        self._reset_parameters()

        # load backbone after param reset
        self.backbone = torch.load(backbone_path)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')
        if self.baseline:
            query = self.backbone.extract_features(data.get('query'))
            ref = self.backbone.extract_features(data.get('ref'))

            z_query = self.avg_pooling_2d(query).flatten(start_dim=1)
            z_ref = self.avg_pooling_2d(ref).flatten(start_dim=1)

            z = torch.cat((z_query, z_ref), dim=1)
            delta = self.head(z)
        else:
            query_endpoints = self.backbone.extract_endpoints(query)
            ref_endpoints = self.backbone.extract_endpoints(ref)
            query_endpoints = query_endpoints[self.reduction[0]]
            ref_endpoints = ref_endpoints[self.reduction[0]]

            # delta_img_x is N x 2D x H_R x W_R
            delta_img = torch.cat((query_endpoints, ref_endpoints), dim=1)
            delta_img = self.proj(delta_img) #N X hidden_dim x H_R x W_R
            delta = self.delta_reg(delta_img)

        if self.type == "rotation":
            delta_q = delta
            if self.convert_to_quat:
                delta_q = A_vec_to_quat(delta_q)
            dummy_delta_x = torch.zeros((delta_q.shape[0], 3)).to(delta_q.device)
            delta_p = torch.cat((dummy_delta_x, delta_q), dim=1)
        else:
            delta_x = delta
            dummy_delta_q = torch.zeros((delta_x.shape[0], 4)).to(delta_x.device)
            delta_p = torch.cat((delta_x, dummy_delta_q), dim=1)

        return {"rel_pose": delta_p}


class MSDeltaNet(nn.Module):

    def __init__(self, config, backbone_path):
        super().__init__()

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320, "baseline":1280}
        self.reduction = config.get("reduction")
        self.hidden_dim = config.get("hidden_dim")
        self.type = config.get("type")
        if self.type == "rotation":
            rot_dim = config.get("rot_dim")
            if config.get("rot_repr_type") == '10d':
                rot_dim = 10
                self.convert_to_quat = True
            else:
                self.convert_to_quat = False
            out_dim = rot_dim
        else:
            out_dim = 3
        self.ms_proj = nn.ModuleDict()
        self.ms_delta_reg = nn.ModuleDict()
        for reduction in self.reduction:
            if reduction == "baseline":
                hidden_dim = reduction_map[reduction] * 2
                self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
                self.avg_pooling_2d = nn.AdaptiveAvgPool2d(1)
            else:
                self.ms_proj[reduction] = nn.Conv2d(reduction_map[reduction] * 2,
                                                 self.hidden_dim, kernel_size=1)
                reg_type = config.get("regressor_type")
                if reg_type == "transformer":
                    delta_reg = Relformer(self.hidden_dim, out_dim, self.hidden_dim)
                elif reg_type == "conv":
                    delta_reg = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, out_dim, 1)
                else:
                    raise NotImplementedError(reg_type)
                self.ms_delta_reg[reduction] = delta_reg

        self._reset_parameters()

        # load backbone after param reset
        self.backbone = torch.load(backbone_path)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')
        ret_val = {}
        for reduction in self.reduction:
            if reduction == "baseline":

                query = self.backbone.extract_features(data.get('query'))
                ref = self.backbone.extract_features(data.get('ref'))

                z_query = self.avg_pooling_2d(query).flatten(start_dim=1)
                z_ref = self.avg_pooling_2d(ref).flatten(start_dim=1)

                z = torch.cat((z_query, z_ref), dim=1)
                delta = self.head(z)
            else:
                query_endpoints = self.backbone.extract_endpoints(query)
                ref_endpoints = self.backbone.extract_endpoints(ref)
                query_endpoints = query_endpoints[reduction]
                ref_endpoints = ref_endpoints[reduction]

                # delta_img_x is N x 2D x H_R x W_R
                delta_img = torch.cat((query_endpoints, ref_endpoints), dim=1)
                delta_img = self.ms_proj[reduction](delta_img) #N X hidden_dim x H_R x W_R
                delta = self.ms_delta_reg[reduction](delta_img)

            if self.type == "rotation":
                delta_q = delta
                if self.convert_to_quat:
                    delta_q = A_vec_to_quat(delta_q)
                dummy_delta_x = torch.zeros((delta_q.shape[0], 3)).to(delta_q.device)
                delta_p = torch.cat((dummy_delta_x, delta_q), dim=1)
            else:
                delta_x = delta
                dummy_delta_q = torch.zeros((delta_x.shape[0], 4)).to(delta_x.device)
                delta_p = torch.cat((delta_x, dummy_delta_q), dim=1)
            ret_val[reduction] = delta_p

        return {"rel_pose": ret_val}

