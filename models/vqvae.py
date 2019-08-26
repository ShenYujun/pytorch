# python3.7
# """Contains the implementation of VQ-VAE and VQ-VAE-2 model.

# Please refer to following papers for more details.

# VQ-VAE: https://arxiv.org/pdf/1711.00937.pdf
# VQ-VAE-2: https://arxiv.org/pdf/1906.00446.pdf
# """

# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = ['VQVAE2', 'VectorQuantisationLayer']

# Defines a dictionary, which maps the image resolution to strides of each
# encoder.
_RESOLUTIONS_TO_STRIDES = {
    64: [2, 2],
    128: [2, 2],
    256: [4, 2],
    512: [4, 2, 2],
    1024: [8, 2, 2],
}


# class VQVAE2(nn.Module):
#   """Defines the VQ-VAE-2 model."""

#   def __init__(self,
#                resolution=256,
#                in_channels=3,
#                hidden_channels=128,
#                residual_channels=64,
#                num_blocks=2,
#                embedding_dim=64,
#                num_embeddings=512,
#                commitment_loss_weight=0.25):
#     """Initializes with basic settings.

#     Args:
#       resolution: Resolution of the input image. (default: 256)
#       in_channels: Number of channels of the input image. (default: 3)
#       hidden_channels: Number of hidden channels for convolutional layer.
#         (default: 128)
#       residual_channels: Number of channels for residual block. (default: 64)
#       num_blocks: Number of residual blocks used in each encoder and decoder.
#         (default: 2)
#       embedding_dim: Dimensional of each embedding vector. (default: 64)
#       num_embeddings: Number of embedding vectors. (default: 512)
#       commitment_loss_weight: Weight for commitment loss. (default: 0.25)

#     Raises:
#       ValueError: If the input resolution is not suppported.
#     """
#     super().__init__()

#     try:
#       self.strides = _RESOLUTIONS_TO_STRIDES[resolution]
#     except KeyError:
#       raise ValueError(f'Invalid resolution: {resolution}!\n'
#                        f'Resolutions allowed: {list(_RESOLUTIONS_TO_STRIDES)}.')

#     self.resolution = resolution
#     self.in_channels = in_channels
#     self.hidden_channels = hidden_channels
#     self.residual_channels = residual_channels
#     self.num_blocks = num_blocks
#     self.embedding_dim = embedding_dim
#     self.num_embeddings = num_embeddings
#     self.commitment_loss_weight = commitment_loss_weight

#     for idx, stride in enumerate(self.strides):
#       # Encoder.
#       in_channels = self.in_channels if idx == 0 else self.hidden_channels
#       self.add_module(f'encoder{idx}',
#                       Encoder(in_channels=in_channels,
#                               hidden_channels=self.hidden_channels,
#                               residual_channels=self.residual_channels,
#                               num_blocks=self.num_blocks,
#                               stride=stride))

#       # Convolutional layer before each vector quantisation layer.
#       in_channels = (self.hidden_channels if idx == (len(self.strides) - 1) else
#                      (self.embedding_dim + hidden_channels))
#       self.add_module(f'quant_conv{idx}',
#                       nn.Conv2d(in_channels=in_channels,
#                                 out_channels=self.embedding_dim,
#                                 kernel_size=1,
#                                 padding=0))

#       # Quantisation layer.
#       self.add_module(f'quant{idx}',
#                       VectorQuantisationLayer(
#                           embedding_dim=self.embedding_dim,
#                           num_embeddings=self.num_embeddings,
#                           commitment_loss_weight=self.commitment_loss_weight))

#       # Decoder.
#       in_channels = (self.embedding_dim * 2 if idx == 0 else
#                      self.embedding_dim)
#       out_channels = self.in_channels if idx == 0 else self.embedding_dim
#       self.add_module(f'decoder{idx}',
#                       Decoder(in_channels=in_channels,
#                               out_channels=out_channels,
#                               hidden_channels=self.hidden_channels,
#                               residual_channels=self.residual_channels,
#                               num_blocks=self.num_blocks,
#                               stride=stride))

#       # Upsample layer.
#       if idx != 0:
#         in_channels = (self.embedding_dim if idx == (len(self.strides) - 1) else
#                        self.embedding_dim * 2)
#         self.add_module(f'upsample{idx}',
#                         nn.ConvTranspose2d(in_channels=in_channels,
#                                            out_channels=self.embedding_dim,
#                                            kernel_size=4,
#                                            stride=2,
#                                            padding=1))

#   def encode(self, x):
#     """Encodes given image to quantized codes."""
#     assert x.shape[2] == self.resolution
#     encoded_x = [None for _ in self.strides]
#     for idx in range(len(self.strides)):
#       if idx == 0:
#         encoded_x[idx] = self.__getattr__(f'encoder{idx}')(x)
#       else:
#         encoded_x[idx] = self.__getattr__(f'encoder{idx}')(encoded_x[idx - 1])

#     quantized_x = [None for _ in self.strides]
#     embedding_indices = [None for _ in self.strides]
#     total_loss = 0
#     concat = None
#     for idx in range(len(self.strides) - 1, -1, -1):
#       if idx == len(self.strides) - 1:
#         concat = encoded_x[idx]
#       else:
#         decoded_quant = self.__getattr__(f'decoder{idx + 1}')(
#             quantized_x[idx + 1])
#         concat = torch.cat([decoded_quant, encoded_x[idx]], dim=1)
#       concat = self.__getattr__(f'quant_conv{idx}')(concat).permute(0, 2, 3, 1)
#       quant, loss, indices = self.__getattr__(f'quant{idx}')(concat)
#       embedding_indices[idx] = indices
#       quantized_x[idx] = quant.permute(0, 3, 1, 2)
#       total_loss = total_loss + loss

#     return [quantized_x, total_loss, embedding_indices]

#   def decode(self, quantized_x):
#     """Decodes quantized codes to image."""
#     assert len(quantized_x) == len(self.strides)
#     concat = None
#     for idx in range(len(self.strides) - 1, -1, -1):
#       if idx == 0:
#         reconstructed_x = self.decoder0(concat)
#       else:
#         if idx == len(self.strides) - 1:
#           upsampled_quant = self.__getattr__(f'upsample{idx}')(quantized_x[idx])
#         else:
#           upsampled_quant = self.__getattr__(f'upsample{idx}')(concat)
#         concat = torch.cat([upsampled_quant, quantized_x[idx - 1]], dim=1)

#     return reconstructed_x

#   def decode_from_embedding_indices(self, embedding_indices):
#     """Decode embedding indices to image."""
#     assert len(embedding_indices) == len(self.strides)
#     quantized_x = []
#     for idx in range(len(self.strides)):
#       quantized_x.append(self.__getattr__(f'quant{idx}').get_code_from_index(
#           embedding_indices[idx]).permute(0, 3, 1, 2))
#     return self.decode(quantized_x)

#   def forward(self, x):
#     assert x.shape[2] == self.resolution
#     quantized_x, loss, _ = self.encode(x)
#     reconstructed_x = self.decode(quantized_x)
#     return reconstructed_x, loss


# class VectorQuantisationLayer(nn.Module):
#   """Implements the vector quantisation layer used in VQ-VAE.

#   Basically, this layer employs a latent embedding space R^{K * D} with K
#   embedding vectors, each of which is D-dimensional. Given any vector, it will
#   be firstly reshaped to size [N, D], and then quantized with aforementioned
#   K vectors.

#   NOTE: All N vectors will be quantized independently.
#   """

#   def __init__(self,
#                embedding_dim=64,
#                num_embeddings=512,
#                commitment_loss_weight=0.25,
#                moving_decay=0.99,
#                epsilon=1e-5):
#     """Initializes the vector quantisation layer with basic settings.

#     Args:
#       embedding_dim: Dimension of each embedding vector (i.e., D in paper).
#       num_embeddings: Number of embedding vectors (i.e., K in paper).
#       commitment_loss_weight: Weight for commitment loss (i.e., beta in paper).
#       moving_decay: Decay for moving average operation, which is used to update
#         embedding vectors.
#       epsilon: Small float constant to avoid numerical instability.
#     """
#     super().__init__()

#     self.embedding_dim = embedding_dim
#     self.num_embeddings = num_embeddings
#     self.commitment_loss_weight = commitment_loss_weight
#     self.moving_decay = moving_decay
#     self.epsilon = epsilon

#     init_embedding = torch.randn(self.embedding_dim, self.num_embeddings)
#     self.register_buffer('embeddings', init_embedding)
#     self.register_buffer('accumulate_embeddings', init_embedding.clone())
#     self.register_buffer('cluster_size', torch.zeros(self.num_embeddings))

#   def forward(self, x):
#     assert x.shape[-1] == self.embedding_dim
#     flattened_x = x.reshape(-1, self.embedding_dim)
#     distance = (torch.sum(flattened_x**2, dim=1, keepdim=True)
#                 + torch.sum(self.embeddings**2, dim=0, keepdim=True)
#                 - 2 * torch.matmul(flattened_x, self.embeddings))
#     indices = torch.argmax(-distance, dim=1)
#     onehot = F.one_hot(indices, self.num_embeddings).to(x)
#     indices = indices.view(*x.shape[:-1])
#     quantized_x = self.get_code_from_index(indices)

#     if self.training:
#       self.cluster_size.data.copy_(
#           self.moving_decay * self.cluster_size
#           + (1 - self.moving_decay) * torch.sum(onehot, dim=0))
#       delta_embeddings = torch.matmul(flattened_x.transpose(0, 1), onehot)
#       self.accumulate_embeddings.data.copy_(
#           self.moving_decay * self.accumulate_embeddings
#           + (1 - self.moving_decay) * delta_embeddings)
#       n = torch.sum(self.cluster_size)
#       cluster_size = (
#           (self.cluster_size + self.epsilon) /
#           (n + self.num_embeddings * self.epsilon) * n)
#       normalized_embeddings = (
#           self.accumulate_embeddings / cluster_size.view(1, -1))
#       self.embeddings.data.copy_(normalized_embeddings)

#     loss = torch.mean((quantized_x.detach() - x) ** 2)
#     quantized_x = x + (quantized_x - x).detach()

#     return quantized_x, loss * self.commitment_loss_weight, indices

#   def get_code_from_index(self, indices):
#     """Gets embedding codes according to indices."""
#     return F.embedding(indices, self.embeddings.transpose(0, 1))


# class ResidualBlock(nn.Module):
#   """Implements the basic residual block used in encoder and decoder."""

#   def __init__(self, in_channels, residual_channels):
#     super().__init__()

#     self.conv1 = nn.Conv2d(in_channels, residual_channels, 3, padding=1)
#     self.conv2 = nn.Conv2d(residual_channels, in_channels, 1, padding=0)
#     self.activate = nn.ReLU(inplace=True)

#   def forward(self, x):
#     res = self.activate(self.conv2(self.activate(self.conv1(x))))
#     x = x + res
#     return x


# class Encoder(nn.Module):
#   """Implements the encoder module."""

#   def __init__(self,
#                in_channels,
#                hidden_channels=128,
#                residual_channels=64,
#                num_blocks=2,
#                stride=2):
#     """Initializes the encoder module with basic settings.

#     Basically, this module downsamples the input tensor with a set of stride-2
#     convolutional layers, followed by several residual blocks.

#     Args:
#       in_channels: Number of channels of the input tensor fed into this block.
#       hidden_channels: Number of channels of the tensor fed into residual
#         blocks.
#       residual_channels: Number of channels used in residual branch.
#       num_blocks: Number of residual blocks used in this module.
#       stride: Must be `power(2, n)`. This field determines how many downsampling
#         layers are used (i.e., n).
#     """
#     super().__init__()

#     assert stride > 1 and int(np.log2(stride)) == np.log2(stride)

#     self.in_channels = in_channels
#     self.hidden_channels = hidden_channels
#     self.residual_channels = residual_channels
#     self.num_blocks = num_blocks
#     self.stride = stride
#     self.num_layers = int(np.log2(stride))

#     for layer_idx in range(self.num_layers):
#       in_channels = (self.in_channels if layer_idx == 0 else
#                      self.hidden_channels // 2)
#       out_channels = self.hidden_channels // 2
#       self.add_module(f'conv{layer_idx}',
#                       nn.Conv2d(in_channels,
#                                 out_channels,
#                                 kernel_size=4,
#                                 stride=2,
#                                 padding=1))
#     self.add_module(f'conv{self.num_layers}',
#                     nn.Conv2d(self.hidden_channels // 2,
#                               self.hidden_channels,
#                               kernel_size=3,
#                               padding=1))

#     for block_idx in range(self.num_blocks):
#       self.add_module(f'block{block_idx}',
#                       ResidualBlock(self.hidden_channels,
#                                     self.residual_channels))

#     self.activate = nn.ReLU(inplace=True)

#   def forward(self, x):
#     for layer_idx in range(self.num_layers + 1):
#       x = self.activate(self.__getattr__(f'conv{layer_idx}')(x))
#     for block_idx in range(self.num_blocks):
#       x = self.__getattr__(f'block{block_idx}')(x)
#     return x


# class Decoder(nn.Module):
#   """Implements the decoder module."""

#   def __init__(self,
#                in_channels,
#                out_channels,
#                hidden_channels=128,
#                residual_channels=64,
#                num_blocks=2,
#                stride=2):
#     """Initializes the encoder module with basic settings.

#     Basically, this module first processes the input tensor with a set of
#     residual blocks, and then upsamples the input tensor with a set of stride-2
#     deconvolutional layers.

#     Args:
#       in_channels: Number of channels of the input tensor fed into this block.
#       out_channels: Number of channels of the output tensor.
#       hidden_channels: Number of channels of the tensor fed into residual
#         blocks.
#       residual_channels: Number of channels used in residual branch.
#       num_blocks: Number of residual blocks used in this module.
#       stride: Must be `power(2, n)`. This field determines how many downsampling
#         layers are used (i.e., n).
#     """
#     super().__init__()

#     assert stride > 1 and int(np.log2(stride)) == np.log2(stride)

#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.hidden_channels = hidden_channels
#     self.residual_channels = residual_channels
#     self.num_blocks = num_blocks
#     self.stride = stride
#     self.num_layers = int(np.log2(stride))

#     self.conv0 = nn.Conv2d(self.in_channels,
#                            self.hidden_channels,
#                            kernel_size=3,
#                            padding=1)

#     for block_idx in range(self.num_blocks):
#       self.add_module(f'block{block_idx}',
#                       ResidualBlock(self.hidden_channels,
#                                     self.residual_channels))

#     for layer_idx in range(1, self.num_layers + 1):
#       in_channels = (self.hidden_channels if layer_idx == 1 else
#                      self.hidden_channels // 2)
#       out_channels = (self.out_channels if layer_idx == self.num_layers else
#                       self.hidden_channels // 2)
#       self.add_module(f'conv{layer_idx}',
#                       nn.ConvTranspose2d(in_channels,
#                                          out_channels,
#                                          kernel_size=4,
#                                          stride=2,
#                                          padding=1))

#     self.activate = nn.ReLU(inplace=True)

#   def forward(self, x):
#     x = self.activate(self.conv0(x))
#     for block_idx in range(self.num_blocks):
#       x = self.__getattr__(f'block{block_idx}')(x)
#     for layer_idx in range(1, self.num_layers + 1):
#       x = self.__getattr__(f'conv{layer_idx}')(x)
#       if layer_idx != self.num_layers:
#         x = self.activate(x)
#     return x

import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE2(nn.Module):
    def __init__(
        self,
        resolution=256,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.strides = _RESOLUTIONS_TO_STRIDES[resolution]
        self.resolution = resolution

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quants, diff, _ = self.encode(input)
        dec = self.decode(quants[0], quants[1])

        return dec, diff * 0.25

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        # diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        # diff_b = diff_b.unsqueeze(0)

        return [quant_b, quant_t], diff_t + diff_b, [id_b, id_t]

    def decode(self, quant_b, quant_t):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_b, code_t):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
