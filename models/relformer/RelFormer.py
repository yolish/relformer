"""
The Multi-Scene TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone

class RelFormer(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True
        self.backbone = build_backbone(config)

        self.transformer = Transformer(config)
        decoder_dim = self.transformer.d_model

        self.input_proj = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.rel_pose_token = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        self.regressor_head_trans = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

    def forward(self, data):

        query = data.get('query')
        positive = data.get('ref')
        negative = data.get('negative')

        z_neg = None
        if negative is not None:
            if isinstance(negative, (list, torch.Tensor)):
                negative = nested_tensor_from_tensor_list(negative)
            negative, _ = self.backbone(negative)
            negative, _ = negative[0].decompose()
            z_neg = self.input_proj(negative).flatten(start_dim=1)

        batch_size = query.shape[0]

        # Handle data structures
        if isinstance(query, (list, torch.Tensor)):
            query = nested_tensor_from_tensor_list(query)

        ref = positive
        if isinstance(ref, (list, torch.Tensor)):
            ref = nested_tensor_from_tensor_list(ref)

        # Extract the features and the position embedding from the visual backbone
        features_query, pos_embed = self.backbone(query)
        _, pos_embed_seq = pos_embed[0]
        pos_embed = pos_embed_seq.flatten(2).permute(2, 0, 1)
        query, mask = features_query[0].decompose()
        query = self.input_proj(query)
        z_query = query.flatten(start_dim=1)

        features_ref, ref_pos_embed = self.backbone(ref)
        ref_pos_embed_tok, ref_pos_embed_seq = ref_pos_embed[0]
        ref_pos_embed = torch.cat((ref_pos_embed_tok.unsqueeze(2), ref_pos_embed_seq.flatten(2)), dim=2).permute(2, 0,
                                                                                                                 1)
        ref, _ = features_ref[0].decompose()
        ref = self.input_proj(ref)
        z_pos = ref.flatten(start_dim=1)
        ref = ref.flatten(start_dim=2).permute(2, 0, 1)

        z = self.transformer(query, mask, ref, mem_pos_embed[0])[0][0]
        z = z[:, 0, :]
        rel_x = self.regressor_head_trans(z)
        rel_q = self.regressor_head_rot(z)
        rel_pose = torch.cat((rel_x, rel_q), dim=1)
        return { "z_query":z_query, "z_pos":z_pos, "z_neg":z_neg, "rel_pose":rel_pose}

class RelFormer2(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = True
        self.backbone = build_backbone(config)

        self.transformer_x = Transformer(config)
        self.transformer_q = Transformer(config)
        decoder_dim = self.transformer_x.d_model

        self.input_proj_x = nn.Conv2d(self.backbone.num_channels[0], decoder_dim, kernel_size=1)
        self.input_proj_q = nn.Conv2d(self.backbone.num_channels[1], decoder_dim, kernel_size=1)
        self.rel_pose_token_x = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)
        self.rel_pose_token_q = nn.Parameter(torch.zeros((1, decoder_dim)), requires_grad=True)

        self.regressor_head_trans = PoseRegressor(decoder_dim, 3)
        self.regressor_head_rot = PoseRegressor(decoder_dim, 4)

    def prepare_seq(self, features, pos_encoding, layer_idx, append_token,
                    batch_size, seq_type="x"):

        pos_token, pos_embed_seq = pos_encoding[layer_idx]
        if append_token:
            # prepare position encoding
            pos_encoding = torch.cat((pos_token.unsqueeze(2), pos_embed_seq.flatten(2)), dim=2).permute(2,0,1)
        else:
            pos_encoding = pos_embed_seq.flatten(2).permute(2, 0, 1)

        src, mask = features[layer_idx].decompose()
        z = src.flatten(start_dim=1)

        if seq_type == "x":
            src = self.input_proj_x(src)
            if append_token:
                src = src.flatten(start_dim=2).permute(2, 0, 1)
                pose_token = self.rel_pose_token_x.unsqueeze(1).repeat(1, batch_size, 1)
                src = torch.cat((pose_token, src), dim=0)

        else:
            src = self.input_proj_q(src)
            if append_token:
                src = src.flatten(start_dim=2).permute(2, 0, 1)
                pose_token = self.rel_pose_token_q.unsqueeze(1).repeat(1, batch_size, 1)
                src = torch.cat((pose_token, src), dim=0)

        return z, src, mask, pos_encoding

    def forward(self, data):

        query = data.get('query')
        positive = data.get('ref')
        negative = data.get('negative')

        z_neg_x = None
        z_neg_q = None
        if negative is not None:
            if isinstance(negative, (list, torch.Tensor)):
                negative = nested_tensor_from_tensor_list(negative)
            negative, _ = self.backbone(negative)
            negative_x, _ = negative[0].decompose()
            negative_q, _ = negative[1].decompose()
            z_neg_x = self.input_proj(negative_x).flatten(start_dim=1)
            z_neg_q = self.input_proj(negative_q).flatten(start_dim=1)

        batch_size = query.shape[0]

        # Handle data structures
        if isinstance(query, (list, torch.Tensor)):
            query = nested_tensor_from_tensor_list(query)

        ref = positive
        if isinstance(ref, (list, torch.Tensor)):
            ref = nested_tensor_from_tensor_list(ref)

        # Extract the features and the position embedding from the visual backbone
        features_query, pos_embed = self.backbone(query)
        features_ref, ref_pos_embed = self.backbone(ref)

        z_query_x, query_x, mask_x, pos_embed_x = self.prepare_seq(features_query, pos_embed, 0, False,
                    batch_size, seq_type="x")
        z_query_q, query_q, mask_q, pos_embed_q = self.prepare_seq(features_query, pos_embed, 1, False,
                                                                   batch_size, seq_type="q")

        z_positive_x, ref_x, _, ref_pos_embed_x = self.prepare_seq(features_ref, ref_pos_embed, 0, True,
                                                                   batch_size, seq_type="x")
        z_positive_q, ref_q, _, ref_pos_embed_q = self.prepare_seq(features_ref, ref_pos_embed, 1, True,
                                                                   batch_size, seq_type="q")

        z_x = self.transformer_x(query_x, ref_x,  mask_x, pos_embed_x, ref_pos_embed_x)[0][0][:, 0, :]
        z_q = self.transformer_q(query_q, ref_q,  mask_q, pos_embed_q, ref_pos_embed_q)[0][0][:, 0, :]
        rel_x = self.regressor_head_trans(z_x)
        rel_q = self.regressor_head_rot(z_q)
        rel_pose = torch.cat((rel_x, rel_q), dim=1)
        return { "z_query_x":z_query_x, "z_pos_x":z_positive_x, "z_neg_x":z_neg_x,
                 "z_query_q": z_query_q, "z_pos_q": z_positive_q, "z_neg_q": z_neg_q,
                 "rel_pose":rel_pose}


class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)
