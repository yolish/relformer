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

def batched_linear_layer(x, wb):
    # x: (B, N, D1); wb: (B, D1 + 1, D2) or (D1 + 1, D2)
    one = torch.ones(*x.shape[:-1], 1, device=x.device)
    #one = torch.ones(x.shape[1], device=x.device)
    linear_res = torch.matmul(torch.cat([x, one], dim=-1).unsqueeze(1), wb)
    return linear_res.squeeze(1)

def _swish(x):
    return x * F.sigmoid(x)

    def forward(self, delta_img):
        delta_img = self.conv2(self.conv1(delta_img))
        delta_z = self.avg_pooling_2d(delta_img).flatten(start_dim=1)
        delta = self.head(delta_z)
        return delta


class Relformer(nn.Module):

    def __init__(self, hidden_dim, out_dim, estimate_with_prior, reduction_dim, feature_dim, do_hyper, hyper_dim, delta_pos_reg):
        super().__init__()
        f_dim = hidden_dim*feature_dim*feature_dim
        self.hidden_dim = hidden_dim
        self.hyper_dim = hyper_dim
        self.do_hyper = do_hyper
        self.delta_pos_reg = delta_pos_reg
        self.out_dim = out_dim
        self.transformer_encoder = TransformerEncoder({"hidden_dim":hidden_dim})
        self.estimate_with_prior = estimate_with_prior
        self.position_encoder = PositionEmbeddingLearnedWithPoseToken(hidden_dim//2)
        self.rel_pose_token = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)        
        if self.estimate_with_prior:
            self.head_w_prior = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        else:
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        if self.delta_pos_reg:
            self.head2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))

        if self.do_hyper:
            self.tr_rel_pose_token_query = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
            self.tr_rel_pose_token_ref = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
            if self.do_hyper == 3 or self.do_hyper == 5:
                self.hypernet_input_proj = nn.Conv2d(reduction_dim*2, hidden_dim, kernel_size=1)
            if self.do_hyper == 2 or self.do_hyper == 3:
                self.hypernet_fc_1 = nn.Linear(f_dim, (hidden_dim+1) * out_dim)
            if self.do_hyper == 4 or self.do_hyper == 5:
                self.hypernet_fc_1 = nn.Linear(f_dim, (hidden_dim + 1) * self.hyper_dim)
                self.hypernet_fc_2 = nn.Linear(f_dim, (hyper_dim + 1) * out_dim)
            if self.do_hyper == 6:
                f_dim = feature_dim * feature_dim                
                self.hypernet_fc_1 = nn.Linear(f_dim, (hidden_dim+1) * out_dim)
                self.tr_h = Transformer({"hidden_dim":reduction_dim})
                self.position_encoder_h = PositionEmbeddingLearnedWithPoseToken(reduction_dim // 2)
            if self.do_hyper == 7:
                f_dim = feature_dim * feature_dim
                self.hypernet_input_proj = nn.Conv2d(reduction_dim, hidden_dim, kernel_size=1)
                self.hypernet_fc_1 = nn.Linear(hidden_dim, (hidden_dim+1) * out_dim)
                self.tr_h = Transformer({"hidden_dim":hidden_dim})
                self.position_encoder_h = PositionEmbeddingLearnedWithPoseToken(hidden_dim // 2)
            if self.do_hyper == 10 or self.do_hyper == 9:
                f_dim = feature_dim * feature_dim
                self.hypernet_fc_1 = nn.Linear(hidden_dim, (hidden_dim+1) * out_dim)
            if self.do_hyper == 8 and self.delta_pos_reg:
                f_dim = feature_dim * feature_dim
                self.hypernet_input_proj = nn.Conv2d(reduction_dim, hidden_dim, kernel_size=1)
                self.hypernet_fc_1 = nn.Linear(hidden_dim, (hidden_dim+1) * out_dim)
                self.tr_h = Transformer({"hidden_dim":hidden_dim})
                self.position_encoder_h = PositionEmbeddingLearnedWithPoseToken(hidden_dim // 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    

    def forward(self, delta_img_proj, delta_img, query_endpoints_h, ref_endpoints_h, global_hyper, prior_z=None, return_delta_z=False):
        # make into a sequence and append token
        delta_seq = delta_img_proj.flatten(start_dim=2).permute(2,0,1)
        batch_size = delta_img_proj.shape[0]
        rel_token = self.rel_pose_token.unsqueeze(1).repeat(1, batch_size, 1)
        # S x B x D
        delta_seq = torch.cat([rel_token, delta_seq])

        # prepare position encoding
        token_posisition_encoding, activation_posistion_encoding = self.position_encoder(delta_img_proj)
        position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                               activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

        # regress latent relative with transformer encoder
        delta_z = self.transformer_encoder(delta_seq, position_encoding)[:, 0, :]

        if self.estimate_with_prior:
            assert prior_z is not None
            delta_z = torch.cat((delta_z, prior_z), dim=1)
            delta = self.head_w_prior(delta_z)
        else:
            if self.do_hyper >= 1 and self.do_hyper <= 10:
                b, c, h, w = delta_img_proj.shape
                if self.do_hyper <= 2:
                    w_1 = _swish(self.hypernet_fc_1(delta_img_proj.view(b, -1)))
                elif self.do_hyper == 3:
                    delta_img_hyper = self.hypernet_input_proj(delta_img)
                    w_1 = _swish(self.hypernet_fc_1(delta_img_hyper.view(b, -1)))
                if self.do_hyper <= 3:
                    delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
                if self.do_hyper == 4:
                    w_1 = _swish(self.hypernet_fc_1(delta_img_proj.view(b, -1)))
                    w_2 = _swish(self.hypernet_fc_2(delta_img_proj.view(b, -1)))
                    delta_z1 = _swish(batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim + 1), self.hyper_dim)))
                    delta = batched_linear_layer(delta_z1, w_2.view(w_2.shape[0], (self.hyper_dim + 1), self.out_dim))
                if self.do_hyper == 5:
                    delta_img_hyper = self.hypernet_input_proj(delta_img)
                    w_1 = _swish(self.hypernet_fc_1(delta_img_hyper.view(b, -1)))
                    w_2 = _swish(self.hypernet_fc_2(delta_img_hyper.view(b, -1)))
                    delta_z1 = _swish(batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim + 1), self.hyper_dim)))
                    delta = batched_linear_layer(delta_z1, w_2.view(w_2.shape[0], (self.hyper_dim + 1), self.out_dim))
                if self.do_hyper == 6 or self.do_hyper == 7 or (self.do_hyper == 8 and self.delta_pos_reg):
                    if self.do_hyper == 7 or self.do_hyper == 8:
                        query_endpoints_h = self.hypernet_input_proj(query_endpoints_h)
                        ref_endpoints_h = self.hypernet_input_proj(ref_endpoints_h)

                    # prepare position encoding
                    token_posisition_encoding_query, activation_posistion_encoding_query = self.position_encoder(query_endpoints_h)
                    query_pos_encdoing_h = torch.cat([token_posisition_encoding_query.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_query.flatten(2).permute(2, 0, 1)])

                    token_posisition_encoding_ref, activation_posistion_encoding_ref = self.position_encoder(ref_endpoints_h)
                    ref_pos_encdoing_h = torch.cat([token_posisition_encoding_ref.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_ref.flatten(2).permute(2, 0, 1)])

                    #query_pos_encdoing_h = self.position_encoder_h(query_endpoints_h).flatten(start_dim=2).permute(2, 0, 1)                    
                    #ref_pos_encdoing_h = self.position_encoder_h(ref_endpoints_h).flatten(start_dim=2).permute(2, 0, 1)                    
                    query_seq_h = query_endpoints_h.flatten(start_dim=2).permute(2, 0, 1)                    
                    ref_seq_h = ref_endpoints_h.flatten(start_dim=2).permute(2, 0, 1)                    

                    rel_token_query = self.tr_rel_pose_token_query.unsqueeze(1).repeat(1, batch_size, 1)
                    rel_token_ref = self.tr_rel_pose_token_ref.unsqueeze(1).repeat(1, batch_size, 1)
                    # S x B x D
                    query_seq_h = torch.cat([rel_token_query, query_seq_h])
                    ref_seq_h = torch.cat([rel_token_ref, ref_seq_h])

                    # replace concat with transformer
                    b, c, h, w = query_endpoints_h.shape
                    local_res = self.tr_h(query_seq_h, ref_seq_h, query_pos_encdoing_h, ref_pos_encdoing_h)#.transpose(1,2).reshape((b,c,h,w))
                    global_hyper = local_res[:, 0, :]
                    w_1 = _swish(self.hypernet_fc_1(global_hyper.view(b, -1)))
                    delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
                               
                if self.do_hyper == 10 or self.do_hyper == 9:
                    w_1 = _swish(self.hypernet_fc_1(global_hyper.view(b, -1)))
                    delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))

                if self.do_hyper == 8 and not self.delta_pos_reg:
                    delta = self.head(delta_z)                    
                elif self.do_hyper >= 2 and self.do_hyper <=10:
                    delta_z1 = self.head(delta_z)
                    delta += delta_z1

            else:
                delta = self.head(delta_z)
                if self.delta_pos_reg:
                    delta2 = self.head2(delta_z)
                    delta += delta2

        if return_delta_z:
            return delta, delta_z
        else:
            return delta


class Relformer2(nn.Module):

    def __init__(self, hidden_dim, out_dim, reduction_dim, do_hyper, delta_pos_reg):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.do_hyper = do_hyper
        self.out_dim = out_dim
        self.transformer = Transformer({"hidden_dim":hidden_dim})        
        self.position_encoder = PositionEmbeddingLearnedWithPoseToken(hidden_dim // 2)
        self.input_proj = nn.Conv2d(reduction_dim, hidden_dim, kernel_size=1)
        self.rel_pose_token_query = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
        self.rel_pose_token_ref = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))               
        self.delta_pos_reg = delta_pos_reg
        if delta_pos_reg:
            self.head2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))
        if do_hyper:
            self.hypernet_fc_1 = nn.Linear(hidden_dim, (hidden_dim+1) * out_dim)
 
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)   

    def forward(self, query_endpoints, ref_endpoints):
        batch_size = query_endpoints.shape[0]

        query_endpoints = self.input_proj(query_endpoints)
        ref_endpoints = self.input_proj(ref_endpoints)
        # prepare position encoding
        token_posisition_encoding_query, activation_posistion_encoding_query = self.position_encoder(query_endpoints)
        query_pos_encdoing = torch.cat([token_posisition_encoding_query.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_query.flatten(2).permute(2, 0, 1)])

        token_posisition_encoding_ref, activation_posistion_encoding_ref = self.position_encoder(ref_endpoints)
        ref_pos_encdoing = torch.cat([token_posisition_encoding_ref.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_ref.flatten(2).permute(2, 0, 1)])
        
        query_seq = query_endpoints.flatten(start_dim=2).permute(2, 0, 1)                    
        ref_seq = ref_endpoints.flatten(start_dim=2).permute(2, 0, 1)                    

        rel_token_query = self.rel_pose_token_query.unsqueeze(1).repeat(1, batch_size, 1)
        rel_token_ref = self.rel_pose_token_ref.unsqueeze(1).repeat(1, batch_size, 1)
        # S x B x D
        query_seq = torch.cat([rel_token_query, query_seq])
        ref_seq = torch.cat([rel_token_ref, ref_seq])

        # replace concat with transformer        
        global_res = self.transformer(query_seq, ref_seq, query_pos_encdoing, ref_pos_encdoing)[:, 0, :].squeeze(0)

        if self.do_hyper == 20 and self.delta_pos_reg:            
            w_1 = _swish(self.hypernet_fc_1(global_res.view(b, -1)))
            delta = batched_linear_layer(global_res, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
            delta2 = self.head2(global_res)
            delta += delta2
        else:        
            delta = self.head(global_res)
            if self.delta_pos_reg:
                delta2 = self.head2(global_res)
                delta += delta2
        
        return delta


class Relformer3(nn.Module):

    def __init__(self, hidden_dim, out_dim, reduction_dim, feature_dim, do_hyper, delta_pos_reg, hybrid_mode):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.do_hyper = do_hyper
        self.delta_pos_reg = delta_pos_reg
        self.out_dim = out_dim
        self.transformer_encoder = TransformerEncoder({"hidden_dim":hidden_dim})
        self.position_encoder = PositionEmbeddingLearnedWithPoseToken(hidden_dim//2)
        self.rel_pose_token = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))                
        self.head2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))        
        self.hypernet_fc_1 = nn.Linear(hidden_dim, (hidden_dim+1) * out_dim)

        if self.do_hyper == 17 or self.do_hyper == 18 or self.do_hyper == 19 or self.do_hyper == 21:# or hybrid_mode:            
            self.hypernet_input_proj = nn.Conv2d(reduction_dim*2, hidden_dim, kernel_size=1)            
            self.transformer_encoder_h = TransformerEncoder({"hidden_dim":hidden_dim})            
            self.position_encoder_h = PositionEmbeddingLearnedWithPoseToken(hidden_dim // 2)
            self.rel_pose_token_h = nn.Parameter(torch.zeros((1,  hidden_dim)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)    

    #def forward(self, delta_img_proj):
    def forward(self, delta_img_proj, delta_img, query_endpoints_h, ref_endpoints_h, global_hyper):
        # make into a sequence and append token
        delta_seq = delta_img_proj.flatten(start_dim=2).permute(2,0,1)
        batch_size = delta_img_proj.shape[0]
        rel_token = self.rel_pose_token.unsqueeze(1).repeat(1, batch_size, 1)
        # S x B x D
        delta_seq1 = torch.cat([rel_token, delta_seq])

        # prepare position encoding
        token_posisition_encoding, activation_posistion_encoding = self.position_encoder(delta_img_proj)
        position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                               activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

        # regress latent relative with transformer encoder
        delta_z = self.transformer_encoder(delta_seq1, position_encoding)[:, 0, :]

        if self.do_hyper == 17 or self.do_hyper == 18 or (self.do_hyper == 15 and global_hyper == None):
            delta_img_proj2 = self.hypernet_input_proj(delta_img)
            rel_token_h = self.rel_pose_token_h.unsqueeze(1).repeat(1, batch_size, 1)
            # S x B x D
            delta_seq1 = torch.cat([rel_token_h, delta_seq])

            # prepare position encoding
            token_posisition_encoding, activation_posistion_encoding = self.position_encoder_h(delta_img_proj2)
            position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                                activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

            # regress latent relative with transformer encoder
            delta_z1 = self.transformer_encoder_h(delta_seq1, position_encoding)[:, 0, :]

            # replace concat with transformer
            w_1 = _swish(self.hypernet_fc_1(delta_z1.view(batch_size, -1)))
            delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
            delta2 = self.head2(delta_z)
            delta += delta2        
        elif self.do_hyper == 16 or self.do_hyper == 15:
            # replace concat with transformer
            w_1 = _swish(self.hypernet_fc_1(global_hyper.view(batch_size, -1)))
            delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
            delta2 = self.head2(delta_z)
            delta += delta2
        elif self.do_hyper == 19 or self.do_hyper == 21:
            delta_img_proj2 = self.hypernet_input_proj(delta_img)
            rel_token_h = self.rel_pose_token_h.unsqueeze(1).repeat(1, batch_size, 1)
            delta_seq2 = delta_img_proj2.flatten(start_dim=2).permute(2,0,1)
            # S x B x D
            delta_seq1 = torch.cat([rel_token_h, delta_seq2])

            # prepare position encoding
            token_posisition_encoding, activation_posistion_encoding = self.position_encoder_h(delta_img_proj2)
            position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                                activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

            # regress latent relative with transformer encoder
            delta_z1 = self.transformer_encoder_h(delta_seq1, position_encoding)[:, 0, :]

            # replace concat with transformer
            w_1 = _swish(self.hypernet_fc_1(delta_z1.view(batch_size, -1)))
            delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
            delta2 = self.head2(delta_z)
            delta += delta2     
        elif self.do_hyper == 22:   
            w_1 = _swish(self.hypernet_fc_1(delta_z.view(batch_size, -1)))
            delta = batched_linear_layer(delta_z, w_1.view(w_1.shape[0], (self.hidden_dim+1), self.out_dim))
            delta2 = self.head2(delta_z)
            delta += delta2                
        else:
            delta = self.head(delta_z)
            if self.delta_pos_reg:
                delta2 = self.head2(delta_z)
                delta += delta2

        return delta



##### Start: code from - https://github.com/utiasSTARS/bingham-rotation-learning
def A_vec_to_quat(A_vec):
    A = convert_Avec_to_A(A_vec)  # Bx10 -> Bx4x4
    _, evs = torch.symeig(A, eigenvectors=True)
    q = evs[:, :, 0].squeeze()
    return q
### End

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DeltaNet(nn.Module):

    def __init__(self, config, backbone_path):
        super().__init__()

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320, "reduction_6": 1280}
        feature_map = {"reduction_3": 28, "reduction_4": 14, "reduction_5": 7, "reduction_6": 7}
        self.reductions = config.get("reduction")
        self.do_hyper = config.get("do_hyper")
        hyper_dim = config.get("hyper_dim")
        self.delta_img_dim = config.get("delta_img_dim")
        self.hidden_dim = config.get("hidden_dim")
        self.proj_before_concat = config.get("proj_before_concat")
        delta_pos_reg = config.get("delta_pos_reg")
        rot_dim = config.get("rot_dim")
        self.proj_concat = config.get("proj_concat")
        self.hybrid_mode = config.get("hybrid_mode")
        self.is_reproj = config.get("reproj")
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
            elif self.proj_concat:
                self.proj_x = nn.Conv2d(reduction_map[self.reductions[0]] * 2, self.hidden_dim, kernel_size=1)
                self.proj_q = nn.Conv2d(reduction_map[self.reductions[1]] * 2, self.hidden_dim, kernel_size=1)
        
        if self.hybrid_mode == 1:
            self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]] * 2, self.hidden_dim, kernel_size=1)
        if self.hybrid_mode == 2:
            self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]] * 2, self.hidden_dim, kernel_size=1)
            self.proj_q2 = nn.Conv2d(reduction_map[self.reductions[1]] * 2, self.hidden_dim, kernel_size=1)

        if self.is_reproj == 1:
            bilinear = config.get("reproj_bilinear")
            factor = 1 if bilinear else 2
            in_channels = self.hidden_dim*2
            if self.reductions[0] != self.reductions[1]:
                in_channels = self.hidden_dim + self.hidden_dim // 4
            # x: 14x14x512, q: 14x14x512 => x|q
            self.reproj_up1 = Up(in_channels, in_channels * factor // 2, bilinear)  # 28x28x512
            self.reproj_up2 = Up(in_channels // 2, in_channels*factor // 4, bilinear)  # 56x56x256
            self.reproj_up3 = Up(in_channels // 4, in_channels*factor // 8, bilinear)  # 112x112x128
            if self.reductions[1] == "reduction_3":
                self.reproj_out = OutConv(in_channels // 8, 3)
            else:
                self.reproj_up4 = Up(in_channels // 8, in_channels*factor // 16, bilinear)  # 224x224x64
                self.reproj_out = OutConv(in_channels // 16, 3)

        reg_type = config.get("regressor_type")
        self.estimate_position_with_prior = config.get("position_with_prior")
        self.estimate_rotation_with_prior = config.get("rotation_with_prior")
        if reg_type == "transformer":
            self.delta_reg_x = Relformer(self.hidden_dim, 3, self.estimate_position_with_prior, reduction_map[self.reductions[0]], feature_map[self.reductions[0]], self.do_hyper, hyper_dim, delta_pos_reg)
            self.delta_reg_q = Relformer(self.hidden_dim, rot_dim, self.estimate_rotation_with_prior, reduction_map[self.reductions[1]], feature_map[self.reductions[1]], self.do_hyper, hyper_dim, delta_pos_reg)
        elif reg_type == "transformer2":
            self.delta_reg_x = Relformer2(self.hidden_dim, 3, reduction_map[self.reductions[0]], self.do_hyper, False)
            self.delta_reg_q = Relformer2(self.hidden_dim, rot_dim, reduction_map[self.reductions[1]], self.do_hyper, delta_pos_reg)
        elif reg_type == "transformer3":
            self.delta_reg_x = Relformer3(self.hidden_dim, 3, reduction_map[self.reductions[0]], feature_map[self.reductions[0]], self.do_hyper, delta_pos_reg, False)
            self.delta_reg_q = Relformer3(self.hidden_dim, rot_dim, reduction_map[self.reductions[1]], feature_map[self.reductions[0]], self.do_hyper, delta_pos_reg, self.hybrid_mode)
        elif reg_type == "conv":
            self.delta_reg_x = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, 3, 2)
            self.delta_reg_q = DeltaRegressor(self.hidden_dim, self.hidden_dim * 2, rot_dim, 1)
        else:
            raise NotImplementedError(reg_type)

        feature_dim_x = feature_map[self.reductions[0]]
        self.channels_dim_x = reduction_map[self.reductions[0]]
        f_dim_x = feature_dim_x*feature_dim_x
        feature_dim_q = feature_map[self.reductions[1]]
        self.channels_dim_q = reduction_map[self.reductions[1]]
        f_dim_q = feature_dim_q * feature_dim_q        
        
        if self.do_hyper == 11:            
            self.hypernet_fc_x_w = nn.Linear(f_dim_x*self.channels_dim_x*2, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(f_dim_x*self.channels_dim_x*2, self.hidden_dim)
            self.hypernet_fc_q_w = nn.Linear(f_dim_q * self.channels_dim_q * 2, self.hidden_dim * self.channels_dim_q * 2)
            self.hypernet_fc_q_b = nn.Linear(f_dim_q * self.channels_dim_q * 2, self.hidden_dim)

        if self.do_hyper == 12:
            self.hypernet_fc_x_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.tr_x = Transformer({"hidden_dim":self.channels_dim_x })
            self.position_encoder_x = PositionEmbeddingLearnedWithPoseToken(self.channels_dim_x // 2)

            self.hypernet_fc_q_w = nn.Linear(f_dim_q, self.hidden_dim * self.channels_dim_q * 2)
            self.hypernet_fc_q_b = nn.Linear(f_dim_q, self.hidden_dim)
            self.tr_q = Transformer({"hidden_dim":self.channels_dim_q })
            self.position_encoder_q = PositionEmbeddingLearnedWithPoseToken(self.channels_dim_q // 2)

        if self.do_hyper == 13 or self.do_hyper == 10 or self.do_hyper == 14 or self.do_hyper == 9:
            self.tr_rel_pose_token_query_x = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)
            self.tr_rel_pose_token_ref_x = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)
            self.tr_rel_pose_token_query_q = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)
            self.tr_rel_pose_token_ref_q = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)
            self.hypernet_fc_x_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.tr_x = Transformer({"hidden_dim":self.hidden_dim })
            self.position_encoder_x = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)

            self.hypernet_fc_q_w = nn.Linear(self.hidden_dim, self.hidden_dim * self.channels_dim_q * 2)
            self.hypernet_fc_q_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.tr_q = Transformer({"hidden_dim":self.hidden_dim })
            self.position_encoder_q = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)

            self.hypernet_input_proj_x = nn.Conv2d(self.channels_dim_x, self.hidden_dim, kernel_size=1)
            self.hypernet_input_proj_q = nn.Conv2d(self.channels_dim_q, self.hidden_dim, kernel_size=1)
            if self.do_hyper == 14 or self.do_hyper == 9:            
                self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]] * 2, self.hidden_dim, kernel_size=1)
                self.proj_q2 = nn.Conv2d(reduction_map[self.reductions[1]] * 2, self.hidden_dim, kernel_size=1)

        if self.do_hyper == 8:
            self.hypernet_fc_x_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.tr_x = Transformer({"hidden_dim":self.hidden_dim })
            self.position_encoder_x = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)
            self.hypernet_input_proj_x = nn.Conv2d(self.channels_dim_x, self.hidden_dim, kernel_size=1)
            self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)

        if self.do_hyper == 15 or self.do_hyper == 18 or self.do_hyper == 21:
            self.hypernet_fc_x_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.transformer_encoder_x = TransformerEncoder({"hidden_dim":self.hidden_dim})            
            self.position_encoder_x = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)
            self.rel_pose_token_x = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)            
            self.hypernet_input_proj_x = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)
            self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)            

        if self.do_hyper == 16:
            self.hypernet_fc_x_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_x*2)
            self.hypernet_fc_x_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.transformer_encoder_x = TransformerEncoder({"hidden_dim":self.hidden_dim})            
            self.position_encoder_x = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)
            self.rel_pose_token_x = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)            
            self.hypernet_input_proj_x = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)
            self.proj_x2 = nn.Conv2d(reduction_map[self.reductions[0]]*2, self.hidden_dim, kernel_size=1)

            self.hypernet_fc_q_w = nn.Linear(self.hidden_dim, self.hidden_dim*self.channels_dim_q*2)
            self.hypernet_fc_q_b = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.transformer_encoder_q = TransformerEncoder({"hidden_dim":self.hidden_dim})            
            self.position_encoder_q = PositionEmbeddingLearnedWithPoseToken(self.hidden_dim // 2)
            self.rel_pose_token_q = nn.Parameter(torch.zeros((1,  self.hidden_dim)), requires_grad=True)            
            self.hypernet_input_proj_q = nn.Conv2d(reduction_map[self.reductions[1]]*2, self.hidden_dim, kernel_size=1)
            self.proj_q2 = nn.Conv2d(reduction_map[self.reductions[1]]*2, self.hidden_dim, kernel_size=1)
   

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

    def forward_backbone(self, img):
        features = self.backbone.extract_endpoints(img)[self.reductions[0]]
        return features

    def forward(self, data):
        query = data.get('query')
        ref = data.get('ref')

        global_hyper_x = None
        global_hyper_q = None

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

            b, c, h, w = delta_img_x.shape
            if self.do_hyper == 11:                
                w_x_w = _swish(self.hypernet_fc_x_w(delta_img_x.view(b, -1)))
                w_x_b = _swish(self.hypernet_fc_x_b(delta_img_x.view(b, -1)))
                w_x_w = w_x_w.view(b, self.hidden_dim, self.channels_dim_x * 2, 1, 1)
                w_x_b = w_x_b.view(b, self.hidden_dim)
                delta_img_x_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_x.device)

                b, c, h, w = delta_img_q.shape
                w_q_w = _swish(self.hypernet_fc_q_w(delta_img_q.view(b, -1)))
                w_q_b = _swish(self.hypernet_fc_q_b(delta_img_q.view(b, -1)))
                w_q_w = w_q_w.view(b, self.hidden_dim, self.channels_dim_q * 2, 1, 1)
                w_q_b = w_q_b.view(b, self.hidden_dim)
                delta_img_q_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_q.device)

                for i in range(b):
                    w_x_w_i = torch.nn.Parameter(w_x_w[i], requires_grad=False)
                    self.proj_x.weight = w_x_w_i.to(delta_img_q.device)
                    w_x_b_i = torch.nn.Parameter(w_x_b[i], requires_grad=False)
                    self.proj_x.bias = w_x_b_i.to(delta_img_q.device)

                    w_q_w_i = torch.nn.Parameter(w_q_w[i], requires_grad=False)
                    self.proj_q.weight = w_q_w_i.to(delta_img_q.device)
                    w_q_b_i = torch.nn.Parameter(w_q_b[i], requires_grad=False)
                    self.proj_q.bias = w_q_b_i.to(delta_img_q.device)

                    with torch.no_grad():
                        delta_img_x_proj[i] = self.proj_x(delta_img_x[i]) # N X hidden_dim x H_R x W_R
                        delta_img_q_proj[i] = self.proj_q(delta_img_q[i]) # N X hidden_dim x H_R x W_R

            elif self.do_hyper == 12 or self.do_hyper == 13 or self.do_hyper == 10 or self.do_hyper == 14 or self.do_hyper == 9 or self.do_hyper == 8:
               
                query_endpoints_x = self.hypernet_input_proj_x(query_endpoints_x)
                ref_endpoints_x = self.hypernet_input_proj_x(ref_endpoints_x)
                if self.do_hyper != 8:
                    query_endpoints_q = self.hypernet_input_proj_q(query_endpoints_q)
                    ref_endpoints_q = self.hypernet_input_proj_q(ref_endpoints_q)
                
                token_posisition_encoding_query, activation_posistion_encoding_query = self.position_encoder_x(query_endpoints_x)
                query_pos_encdoing_x = torch.cat([token_posisition_encoding_query.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_query.flatten(2).permute(2, 0, 1)])
                token_posisition_encoding_ref, activation_posistion_encoding_ref = self.position_encoder_x(ref_endpoints_x)
                ref_pos_encdoing_x = torch.cat([token_posisition_encoding_ref.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_ref.flatten(2).permute(2, 0, 1)])

                #query_pos_encdoing_x = self.position_encoder_x(query_endpoints_x).flatten(start_dim=2).permute(2, 0, 1)
                #ref_pos_encdoing_x = self.position_encoder_x(ref_endpoints_x).flatten(start_dim=2).permute(2, 0, 1)
                query_seq_x = query_endpoints_x.flatten(start_dim=2).permute(2, 0, 1)                    
                ref_seq_x = ref_endpoints_x.flatten(start_dim=2).permute(2, 0, 1)

                b, c, h, w = query_endpoints_x.shape
                rel_token_query_x = self.tr_rel_pose_token_query_x.unsqueeze(1).repeat(1, b, 1)
                rel_token_ref_x = self.tr_rel_pose_token_ref_x.unsqueeze(1).repeat(1, b, 1)
                # S x B x D
                query_seq_x = torch.cat([rel_token_query_x, query_seq_x])
                ref_seq_x = torch.cat([rel_token_ref_x, ref_seq_x])

                # replace concat with transformer
                local_res_x = self.tr_x(query_seq_x, ref_seq_x, query_pos_encdoing_x, ref_pos_encdoing_x)#.transpose(1,2).reshape((b,c,h,w))
                global_hyper_x = local_res_x[:, 0, :]
                w_x_w = _swish(self.hypernet_fc_x_w(global_hyper_x.view(b, -1)))
                w_x_b = _swish(self.hypernet_fc_x_b(global_hyper_x.view(b, -1)))
                delta_img_x_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_x.device)

                if self.do_hyper != 8:
                    token_posisition_encoding_query, activation_posistion_encoding_query = self.position_encoder_q(query_endpoints_q)
                    query_pos_encdoing_q = torch.cat([token_posisition_encoding_query.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_query.flatten(2).permute(2, 0, 1)])
                    token_posisition_encoding_ref, activation_posistion_encoding_ref = self.position_encoder_q(ref_endpoints_q)
                    ref_pos_encdoing_q = torch.cat([token_posisition_encoding_ref.unsqueeze(2).permute(2, 0, 1), activation_posistion_encoding_ref.flatten(2).permute(2, 0, 1)])

                    #query_pos_encdoing_q = self.position_encoder_q(query_endpoints_q).flatten(start_dim=2).permute(2, 0, 1)
                    #ref_pos_encdoing_q = self.position_encoder_q(ref_endpoints_q).flatten(start_dim=2).permute(2, 0, 1)
                    query_seq_q = query_endpoints_q.flatten(start_dim=2).permute(2, 0, 1)                    
                    ref_seq_q = ref_endpoints_q.flatten(start_dim=2).permute(2, 0, 1)      

                    rel_token_query_q = self.tr_rel_pose_token_query_q.unsqueeze(1).repeat(1, b, 1)
                    rel_token_ref_q = self.tr_rel_pose_token_ref_q.unsqueeze(1).repeat(1, b, 1)
                    # S x B x D
                    query_seq_q = torch.cat([rel_token_query_q, query_seq_q])
                    ref_seq_q = torch.cat([rel_token_ref_q, ref_seq_q])              

                    b, c, h, w = query_endpoints_q.shape
                    local_res_q = self.tr_q(query_seq_q, ref_seq_q, query_pos_encdoing_q, ref_pos_encdoing_q)#.transpose(1,2).reshape((b,c,h,w))
                    global_hyper_q = local_res_q[:, 0, :]
                    w_q_w = _swish(self.hypernet_fc_q_w(global_hyper_q.view(b, -1)))
                    w_q_b = _swish(self.hypernet_fc_q_b(global_hyper_q.view(b, -1)))
                    delta_img_q_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_q.device)

                for i in range(b):
                    w_x_w_i = torch.nn.Parameter(w_x_w[i].view(self.hidden_dim, self.channels_dim_x*2, 1, 1), requires_grad=False)
                    self.proj_x.weight = w_x_w_i.to(delta_img_q.device)
                    w_x_b_i = torch.nn.Parameter(w_x_b[i], requires_grad=False)
                    self.proj_x.bias = w_x_b_i.to(delta_img_q.device)

                    if self.do_hyper != 8:
                        w_q_w_i = torch.nn.Parameter(w_q_w[i].view(self.hidden_dim, self.channels_dim_q*2, 1, 1), requires_grad=False)
                        self.proj_q.weight = w_q_w_i.to(delta_img_q.device)
                        w_q_b_i = torch.nn.Parameter(w_q_b[i], requires_grad=False)
                        self.proj_q.bias = w_q_b_i.to(delta_img_q.device)

                    with torch.no_grad():
                        delta_img_x_proj[i] = self.proj_x(delta_img_x[i]) # N X hidden_dim x H_R x W_R
                        if self.do_hyper != 8:
                            delta_img_q_proj[i] = self.proj_q(delta_img_q[i]) # N X hidden_dim x H_R x W_R
                
                if self.do_hyper == 14 or self.do_hyper == 9:
                    delta_img_x_proj2 = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                    delta_img_q_proj2 = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R
                    delta_img_x_proj += delta_img_x_proj2
                    delta_img_q_proj += delta_img_q_proj2
                if self.do_hyper == 8:
                    delta_img_x_proj2 = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                    delta_img_x_proj += delta_img_x_proj2
                    delta_img_q_proj = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R

            elif self.do_hyper == 15 or self.do_hyper == 18 or self.do_hyper == 21:                
                delta_img_x1 = self.hypernet_input_proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                delta_seq = delta_img_x1.flatten(start_dim=2).permute(2,0,1)                
                rel_token_x = self.rel_pose_token_x.unsqueeze(1).repeat(1, b, 1)
                # S x B x D
                delta_seq1 = torch.cat([rel_token_x, delta_seq])

                # prepare position encoding
                token_posisition_encoding, activation_posistion_encoding = self.position_encoder_x(delta_img_x1)
                position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                                    activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

                # regress latent relative with transformer encoder
                global_hyper_x = self.transformer_encoder_x(delta_seq1, position_encoding)[:, 0, :]
                w_x_w = _swish(self.hypernet_fc_x_w(global_hyper_x.view(b, -1)))
                w_x_b = _swish(self.hypernet_fc_x_b(global_hyper_x.view(b, -1)))
                delta_img_x_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_x.device)

                for i in range(b):
                    w_x_w_i = torch.nn.Parameter(w_x_w[i].view(self.hidden_dim, self.channels_dim_x*2, 1, 1), requires_grad=False)
                    self.proj_x.weight = w_x_w_i.to(delta_img_x.device)
                    w_x_b_i = torch.nn.Parameter(w_x_b[i], requires_grad=False)
                    self.proj_x.bias = w_x_b_i.to(delta_img_x.device)

                    with torch.no_grad():
                        delta_img_x_proj[i] = self.proj_x(delta_img_x[i]) # N X hidden_dim x H_R x W_R
                
                delta_img_x_proj2 = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                delta_img_x_proj += delta_img_x_proj2
                delta_img_q_proj = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R
            
            elif self.do_hyper == 16:                
                delta_img_x1 = self.hypernet_input_proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                delta_seq = delta_img_x1.flatten(start_dim=2).permute(2,0,1)                
                rel_token_x = self.rel_pose_token_x.unsqueeze(1).repeat(1, b, 1)
                # S x B x D
                delta_seq1 = torch.cat([rel_token_x, delta_seq])

                # prepare position encoding
                token_posisition_encoding, activation_posistion_encoding = self.position_encoder_x(delta_img_x1)
                position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                                    activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

                # regress latent relative with transformer encoder
                global_hyper_x = self.transformer_encoder_x(delta_seq1, position_encoding)[:, 0, :]
                w_x_w = _swish(self.hypernet_fc_x_w(global_hyper_x.view(b, -1)))
                w_x_b = _swish(self.hypernet_fc_x_b(global_hyper_x.view(b, -1)))
                delta_img_x_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_x.device)

                for i in range(b):
                    w_x_w_i = torch.nn.Parameter(w_x_w[i].view(self.hidden_dim, self.channels_dim_x*2, 1, 1), requires_grad=False)
                    self.proj_x.weight = w_x_w_i.to(delta_img_x.device)
                    w_x_b_i = torch.nn.Parameter(w_x_b[i], requires_grad=False)
                    self.proj_x.bias = w_x_b_i.to(delta_img_x.device)

                    with torch.no_grad():
                        delta_img_x_proj[i] = self.proj_x(delta_img_x[i]) # N X hidden_dim x H_R x W_R

                b, c, h, w = query_endpoints_q.shape
                delta_img_q1 = self.hypernet_input_proj_q(delta_img_q) #N X hidden_dim x H_R x W_R
                delta_seq = delta_img_q1.flatten(start_dim=2).permute(2,0,1)                
                rel_token_q = self.rel_pose_token_q.unsqueeze(1).repeat(1, b, 1)
                # S x B x D
                delta_seq1 = torch.cat([rel_token_q, delta_seq])

                # prepare position encoding
                token_posisition_encoding, activation_posistion_encoding = self.position_encoder_q(delta_img_q1)
                position_encoding = torch.cat([token_posisition_encoding.unsqueeze(2).permute(2, 0, 1),
                                    activation_posistion_encoding.flatten(2).permute(2, 0, 1)])

                # regress latent relative with transformer encoder
                global_hyper_q = self.transformer_encoder_q(delta_seq1, position_encoding)[:, 0, :]
                w_q_w = _swish(self.hypernet_fc_q_w(global_hyper_q.view(b, -1)))
                w_q_b = _swish(self.hypernet_fc_q_b(global_hyper_q.view(b, -1)))
                delta_img_q_proj = torch.zeros((b, self.hidden_dim, h, w)).to(delta_img_q.device)

                for i in range(b):
                    w_q_w_i = torch.nn.Parameter(w_q_w[i].view(self.hidden_dim, self.channels_dim_q*2, 1, 1), requires_grad=False)
                    self.proj_q.weight = w_q_w_i.to(delta_img_q.device)
                    w_q_b_i = torch.nn.Parameter(w_q_b[i], requires_grad=False)
                    self.proj_q.bias = w_q_b_i.to(delta_img_q.device)

                    with torch.no_grad():
                        delta_img_q_proj[i] = self.proj_q(delta_img_q[i]) # N X hidden_dim x H_R x W_R
                
                delta_img_x_proj2 = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                delta_img_x_proj += delta_img_x_proj2
                delta_img_q_proj2 = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R
                delta_img_q_proj += delta_img_q_proj2
                
            elif self.proj_concat:
                delta_img_x_proj = self.proj_x(delta_img_x) #N X hidden_dim x H_R x W_R
                delta_img_q_proj = self.proj_q(delta_img_q) # #N X hidden_dim x H_R x W_R
                if self.hybrid_mode == 1:
                    delta_img_x_proj2 =  self.proj_x2(delta_img_x)
                    delta_img_x_proj += delta_img_x_proj2
                if self.hybrid_mode == 2:
                    delta_img_x_proj2 =  self.proj_x2(delta_img_x)
                    delta_img_x_proj += delta_img_x_proj2
                    delta_img_q_proj2 =  self.proj_q2(delta_img_q)
                    delta_img_q_proj += delta_img_q_proj2

        x_q = None
        if self.is_reproj == 1:
            #if number of q and x channels is different - upscale x
            delta_img_x_proj1 = delta_img_x_proj
            if self.reductions[0] != self.reductions[1]:
                b,c,h,w = delta_img_q_proj.shape
                delta_img_x_proj1 = delta_img_x_proj.view(b, -1, h, w)
            x_q = torch.cat([delta_img_x_proj1, delta_img_q_proj], dim=1)
            x_q = self.reproj_up1(x_q)
            x_q = self.reproj_up2(x_q)
            x_q = self.reproj_up3(x_q)
            if self.reductions[1] == "reduction_4":
                x_q = self.reproj_up4(x_q)
            x_q = self.reproj_out(x_q)

        if self.estimate_position_with_prior:
            delta_q, delta_z = self.delta_reg_q(delta_img_q_proj, return_delta_z=True)
            delta_x = self.delta_reg_x(delta_img_x_proj, prior_z=delta_z)
        elif self.estimate_rotation_with_prior:
            delta_x, delta_z = self.delta_reg_x(delta_img_x_proj, return_delta_z=True)
            delta_q = self.delta_reg_q(delta_img_q_proj, prior_z=delta_z)
        elif self.proj_concat:
            delta_x = self.delta_reg_x(delta_img_x_proj, delta_img_x, query_endpoints_x, ref_endpoints_x, global_hyper_x)
            delta_q = self.delta_reg_q(delta_img_q_proj, delta_img_q, query_endpoints_q, ref_endpoints_q, global_hyper_q)
        else:
            delta_x = self.delta_reg_x(query_endpoints_x, ref_endpoints_x)
            delta_q = self.delta_reg_q(query_endpoints_q, ref_endpoints_q)

        if self.convert_to_quat:
            delta_q = A_vec_to_quat(delta_q)
        delta_p = torch.cat((delta_x, delta_q), dim=1)
        return {"rel_pose": delta_p, "reproj": x_q}


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

        reduction_map = {"reduction_3": 40, "reduction_4": 112, "reduction_5": 320, "reduction_6": 1280}
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

