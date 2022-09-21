import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICE']='-1'
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D,UpSampling2D


CFGS = {
    'swin_tiny_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'swin_small_224': dict(input_size=(224, 224), window_size=7, embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'swin_base_224': dict(input_size=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_base_384': dict(input_size=(384, 384), window_size=12, embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'swin_large_224': dict(input_size=(224, 224), window_size=7, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
    'swin_large_384': dict(input_size=(384, 384), window_size=12, embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48])
}


def Gelu(x):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.
  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf

class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., prefix=''):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'{prefix}/mlp/fc1')
        self.fc2 = Dense(out_features, name=f'{prefix}/mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x,training=True):
        x = self.fc1(x)
        x = Gelu(x)
        x = self.drop(x,training)
        x = self.fc2(x)
        x = self.drop(x,training)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.get_shape().as_list()
    x = tf.reshape(x, shape=[-1, H // window_size,
                   window_size, W // window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    x = tf.reshape(windows, shape=[-1, H // window_size,
                   W // window_size, window_size, window_size, C])
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, shape=[-1, H, W, C])
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., prefix=''):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.prefix = prefix

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,
                         name=f'{self.prefix}/attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim, name=f'{self.prefix}/attn/proj')
        self.proj_drop = Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(f'{self.prefix}/attn/relative_position_bias_table',
                                                            shape=(
                                                                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(initial_value=tf.convert_to_tensor(
            relative_position_index), trainable=False, name=f'{self.prefix}/attn/relative_position_index')
        self.built = True

    def call(self, x, mask=None,training=False):
        B_, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(
            x), shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        relative_position_bias = tf.gather(self.relative_position_bias_table, tf.reshape(
            self.relative_position_index, shape=[-1]))
        relative_position_bias = tf.reshape(relative_position_bias, shape=[
                                            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1])
        relative_position_bias = tf.transpose(
            relative_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn,training)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x,training)
        return x


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.prefix = prefix

        self.norm1 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm1')
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, prefix=self.prefix)
        self.drop_path = DropPath(
            drop_path_prob if drop_path_prob > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'{self.prefix}/norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop, prefix=self.prefix)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size])
            attn_mask = tf.expand_dims(
                mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name=f'{self.prefix}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x,training=False):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, f"input feature has wrong size,{H},{W},{L},{H*W}"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask,training=training)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x,training)
        
        x = x + self.drop_path(self.mlp(self.norm2(x),training),training)

        return x


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,
                               name=f'{prefix}/downsample/reduction')
        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/downsample/norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchUpsampling(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, prefix=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expandsion = UpSampling2D(size=(2, 2),interpolation='nearest',name=f'{prefix}/upsample/upsampling')
        self.reduce_emb = Dense(dim // 2, use_bias=False,
                               name=f'{prefix}/upsample/up_emb')

    def call(self, x):
        H, W = self.input_resolution
        # # print(x.get_shape().as_list())
        B, H,W, C = x.get_shape().as_list()
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # x = tf.reshape(x, shape=[-1, H, W, C])
        x = self.expandsion(x)
        x = self.reduce_emb(x)

        return x


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, downsample=None, use_checkpoint=False, prefix='',
                 trajnet=False,traj_heads=1,traj_num=64,traj_dim=384,map_trajnet=False,map_num=256,map_dim=384):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = [SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)]
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.downsample = None
        
        self.trajnet = trajnet
        self.map_trajnet = map_trajnet
        if self.trajnet:
            self.traj_attn = TrajPicModule(dim,input_resolution,traj_heads,traj_num,traj_dim,map_num,map_dim,use_map=self.map_trajnet)

    def call(self, x,traj=None,mask=None,mapt=None,map_mask=None,training=False):
        for block in self.blocks:
            x = block(x,training)

        res = x
        if self.trajnet:
            res,mm_x = self.traj_attn(x,traj,mask,mapt,map_mask,training)
            x = x + mm_x

        if self.downsample is not None:
            x = self.downsample(x)
            return x,res
        else:
            return x,x

class BasicLayerDecoder(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0., norm_layer=LayerNormalization, upsample=None, use_checkpoint=False,res_connection=False,prefix=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.res_connection = res_connection

        self.blocks = [SwinTransformerBlock(dim=dim//2, input_resolution=(input_resolution[0]*2,input_resolution[0]*2),
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (
                                               i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(
                                               drop_path_prob, list) else drop_path_prob,
                                           norm_layer=norm_layer,
                                           prefix=f'{prefix}/blocks{i}') for i in range(depth)]
        if self.res_connection:
            self.conv_layer = Conv2D(dim//2, kernel_size=(1,1))
        if upsample is not None:
            self.upsample = upsample(
                input_resolution, dim=dim, norm_layer=norm_layer, prefix=prefix)
        else:
            self.upsample = None

        self.norm = norm_layer(epsilon=1e-5, name=f'{prefix}/upsample/norm')

    def call(self,x,res=None,training=False):
        if self.upsample is not None:
            x = self.upsample(x)
        if self.res_connection:
            B, H,W, C = x.get_shape().as_list()
            res = tf.reshape(res,[B,H,W,C])
            res = self.conv_layer(res)
            x = x + res
            x = tf.reshape(x,[B,H*W,C])
        x = self.norm(x)
        # x = self.blocks(x,training)
        # print(x.get_shape())
        for block in self.blocks:
            x = block(x,training)
            # print(x.get_shape())
        x = tf.reshape(x, [B,H,W,C])
        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__(name='patch_embed')
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size,
                           strides=patch_size, name='proj')
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5, name='norm')
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = tf.reshape(
            x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x

class SwinTransformerEncoder(tf.keras.Model):
    def __init__(self, model_name='swin_tiny_patch4_window7_224', include_top=False,
                 img_size=(224, 224), patch_size=(4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False,sep_encode=False,no_map=False,flow_sep=False,use_flow=False,large_input=False, **kwargs):
        super().__init__(name=model_name)
        """
        Encoder of SwinTransformer

        Input:
        x : [batch, 256, 256, 10, 2] # 10s OGM of both vehicle(0) and cyclist_pedestrian(1)
        map_img : [batch, 256, 256, 3] # BEV map image

        Output:
        res_list: a list of results of SwinT Layer output:
        H = W = 255 , C = emb_dim
        [0-3]: 
        [H/4,W/4,C],
        [H/8,W/8,2C],
        [H/16,W/16,4C],
        [H/32,W/32,8C]
        """

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.flow_sep = flow_sep
        self.no_map=no_map

        self.use_flow = use_flow
        self.large_input = large_input

        # split image into non-overlapping patches
        self.patch_embed_vecicle = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
            
        self.sep_encode=sep_encode

        num_patches = self.patch_embed_vecicle.num_patches
        patches_resolution = self.patch_embed_vecicle.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        if sep_encode:
            
            if self.use_flow:
                self.patch_embed_flow = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
                
                if self.flow_sep:
                    self.flow_norm = norm_layer(epsilon=1e-5,name='all_norm')
                    self.flow_layer = BasicLayer(dim=int(embed_dim * (2 ** 0)),
                                                    input_resolution=(patches_resolution[0] // (2 ** 0),
                                                                    patches_resolution[1] // (2 ** 0)),
                                                    depth=depths[0],
                                                    num_heads=num_heads[0],
                                                    window_size=window_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path_prob=dpr[sum(depths[:0]):sum(
                                                        depths[:0 + 1])],
                                                    norm_layer=norm_layer,
                                                    downsample=PatchMerging if (
                                                        0 < self.num_layers - 1) else None,# No downsample of the last layer
                                                    use_checkpoint=use_checkpoint,
                                                    prefix=f'flow_layers{0}')      
            if not self.no_map:
                self.patch_embed_map = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
        # build layers
        self.basic_layers = [BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,# No downsample of the last layer
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)]
        
        self.final_resolution = patches_resolution[0] // (2 ** (self.num_layers - 1)), patches_resolution[1] // (2 ** (self.num_layers - 1))
        self.all_patch_norm = norm_layer(epsilon=1e-5,name='all_norm')
        
        dummy_ogm = tf.zeros([1,img_size[0],img_size[1],11,2])
        if self.large_input:
           dummy_map = tf.zeros([1,img_size[0]//2,img_size[1]//2,3]) 
        else:
            dummy_map = tf.zeros([1,img_size[0],img_size[1],3])

        dummy_flow =tf.zeros((1,)+(img_size[0],img_size[1])+(2,))

        self(dummy_ogm,dummy_map,dummy_flow)
        self.summary()

    def forward_features(self,x,map_img,flow=None,training=True):
        if self.sep_encode:
            vec,ped_cyc = x[:,:,:,:,0],x[:,:,:,:,1]
            if self.no_map:
                x = self.patch_embed_vecicle(vec)
            elif self.flow_sep and self.use_flow:
                flow = self.patch_embed_flow(flow)
                flow = self.flow_norm(flow)
                flow_x,flow_res = self.flow_layer(flow,training)
                if not self.large_input:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img)
                else:
                    maps = self.patch_embed_map(map_img)
                    maps = tf.reshape(maps,[-1,64,64,self.embed_dim])
                    maps = tf.pad(maps,tf.constant([[0,0],[32,32],[32,32],[0,0]]))
                    maps = tf.reshape(maps,[-1,128*128,self.embed_dim])
                    x = self.patch_embed_vecicle(vec)
                    x = x + maps
            else:
                if self.use_flow:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img) + self.patch_embed_flow(flow)
                else:
                    x = self.patch_embed_vecicle(vec) +  self.patch_embed_map(map_img)
        else:
            x = tf.reshape(x,[-1,256,256,11*2])
            if not self.no_map and self.use_flow:
                x = tf.concat([x,map_img,flow], axis=-1)
            elif not self.use_flow:
                x = tf.concat([x,map_img], axis=-1)
            x = self.patch_embed_vecicle(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.all_patch_norm(x)
        res_list=[]
        for i,st_layer in enumerate(self.basic_layers):
            x,res = st_layer(x,training)
            if i==self.num_layers - 1:
                H, W = self.final_resolution
                B, L, C = x.get_shape().as_list()
                assert L == H * W, "input feature has wrong size"
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
                res = tf.reshape(res, shape=[-1, H, W, C])
            if i==0 and self.flow_sep and self.use_flow:
                x = x + flow_x
                if self.large_input:
                    flow_res = tf.reshape(tf.reshape(flow_res,[-1,128,128,self.embed_dim])[:,32:32+64,32:32+64,:],[-1,64*64,96])
                res_list.append(flow_res)
            if self.large_input:
                init_res = 128 // (2**i)
                dim = self.embed_dim * (2**i)
                crop = init_res // 2
                c_b,c_e = int(init_res*0.25),int(init_res*0.75)
                res = tf.reshape(tf.reshape(res,[-1,init_res,init_res,dim])[:,c_b:c_e,c_b:c_e,:],[-1,crop*crop,dim])
            res_list.append(res)
        return res_list

    def call(self, x,map_img,flow,training=True):
        x = self.forward_features(x,map_img,flow,training)
        return x

class Pyramid3DDecoder(tf.keras.Model):
    def __init__(self,config,img_size,use_pyramid=False,model_name='PyrDecoder',split_pred=False,
        timestep_split=False,double_decode=False,stp_grad=False,shallow_decode=0,flow_sep_decode=False,
        conv_cnn=False,sep_conv=False,rep_res=True,fg_sep=False):
        super().__init__(name=model_name)
        decode_inds = [4, 3, 2, 1, 0][shallow_decode:]
        decoder_channels = [48, 96, 128, 192, 384]

        self.stp_grad = stp_grad
        self.rep_res = rep_res

        #traj-rrc

        conv2d_kwargs = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }

        self.upsample = [
            tf.keras.layers.UpSampling3D(size=(1,2,2),name=f'upsample_{i}') for i in decode_inds
        ]
        if conv_cnn:
            self.upconv_0s = [
                tf.keras.layers.ConvLSTM2D(
                filters=decoder_channels[decode_inds[0]],
                activation='elu',
                name=f'uplstmconv_{0}_0',
                return_sequences=True,
                **conv2d_kwargs)] + [
                        tf.keras.layers.Conv2D(
                    filters=decoder_channels[i],
                    activation='elu',
                    name=f'upconv_{i}_0',
                    **conv2d_kwargs) for i in decode_inds[1:]
                ]
        else:
            self.upconv_0s = [
                tf.keras.layers.Conv2D(
                filters=decoder_channels[i],
                activation='elu',
                name=f'upconv_{i}_0',
                **conv2d_kwargs) for i in decode_inds
            ]
        self.flow_sep_decode = flow_sep_decode

        if flow_sep_decode:
            self.upsample_f = [
            tf.keras.layers.UpSampling3D(size=(1,2,2),name=f'upsamplef_{i}') for i in decode_inds[-2:]
            ]
            if sep_conv:
                self.upconv_f = [
                tf.keras.layers.ConvLSTM2D(filters=96,activation='elu',name=f'upconvf_{1}_0',return_sequences=True,**conv2d_kwargs),
                tf.keras.layers.Conv2D(filters=48,activation='elu',name=f'upconvf_{0}_0',**conv2d_kwargs)
                ]
            else:
                self.upconv_f = [
                    tf.keras.layers.Conv2D(
                    filters=decoder_channels[i],
                    activation='elu',
                    name=f'upconvf_{i}_0',
                    **conv2d_kwargs) for i in decode_inds[-2:]
                ]
            self.res_f = tf.keras.layers.Conv3D(
                filters=128,
                activation='elu',
                name='resconv_f',
                kernel_size=(8,1,1),
                strides=1,
                padding='same')

            self.output_layer_f = tf.keras.layers.Conv2D(
                filters=2,
                activation=None,
                name=f'outconv',
                **conv2d_kwargs)
        
        self.use_pyramid = use_pyramid
        if use_pyramid:
            self.res_layer = [
            tf.keras.layers.Conv3D(
            filters=decoder_channels[i],
            activation='elu',
            name=f'resconv_{i}',
            kernel_size=(8,1,1),
            strides=1,
            padding='same') for i in decode_inds[:3-shallow_decode]
            ]
            self.ind_list=[2,1,0][shallow_decode:]
            self.reshape_dim = [16,32,64][shallow_decode:]

        if flow_sep_decode:
            out_dim=2
        else:
            out_dim=4

        self.output_layer = tf.keras.layers.Conv2D(
            filters=out_dim,
            activation=None,
            name=f'outconv',
            **conv2d_kwargs)
    
    def get_flow_output(self,x):
        for upsample,uconv_0 in zip(self.upsample_f,self.upconv_f):
            x = upsample(x)
            x = uconv_0(x)
        x = self.output_layer_f(x)
        return x
    
    def call(self,x,training=True,res_list=None):
        if self.stp_grad:
            x = tf.stop_gradient(x)
        i = 0
        if self.flow_sep_decode:
            flow_res = res_list[0]
            res_list = res_list[1:]
        for upsample,uconv_0 in zip(self.upsample,self.upconv_0s):
            x = upsample(x)
            x = uconv_0(x)

            if self.use_pyramid and i<=len(self.ind_list)-1:
                if self.rep_res:
                    res_flat  = tf.repeat(res_list[self.ind_list[i]][:,tf.newaxis],repeats=8,axis=1)
                else:
                    res_flat = res_list[self.ind_list[i]]

                if self.stp_grad:
                    res_flat = tf.stop_gradient(res_flat)
                h = res_flat.get_shape().as_list()[-1]
                res_flat  = tf.reshape(res_flat,[-1,8,self.reshape_dim[i],self.reshape_dim[i],h])
                x = x + self.res_layer[i](res_flat)

            if i==len(self.ind_list)-1 and self.flow_sep_decode:
                flow_res = tf.reshape(flow_res,[-1,64,64,96])
                flow_res = tf.repeat(flow_res[:,tf.newaxis],repeats=8,axis=1)
                flow_x = x + self.res_f(flow_res)
            i+=1  
        x = self.output_layer(x)      
        if self.flow_sep_decode:
            flow_x = self.get_flow_output(flow_x)
            x = tf.concat([x,flow_x],axis=-1)

        return x

from trajNet import TrajNetCrossAttention
from FG_MSA import FGMSA

class STrajNet(tf.keras.Model):
    def __init__(self,cfg,model_name='STrajNet',use_pyramid=True,actor_only=True,sep_actors=False,
        fg_msa=False,use_last_ref=False,fg=False,large_ogm=True):
        super().__init__(name=model_name)

        self.encoder = SwinTransformerEncoder(include_top=True,img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads'],
            sep_encode=True,flow_sep=True,use_flow=True,drop_rate=0.0, attn_drop_rate=0.0,drop_path_rate=0.1,
            large_input=large_ogm)
        
        if sep_actors:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=True)
        else:
            traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=False)
        
        resolution=[8,16,32]
        hw = resolution[4-len(cfg['depths'][:])]
        self.trajnet_attn = TrajNetCrossAttention(traj_cfg,actor_only=actor_only,pic_size=(hw,hw),pic_dim=768//(2**(4-len(cfg['depths'][:])))
        ,multi_modal=True,sep_actors=sep_actors)
        self.fg_msa = fg_msa
        self.fg = fg
        if fg_msa:
            self.fg_msa_layer = FGMSA(q_size=(16,16), kv_size=(16,16),n_heads=8,n_head_channels=48,n_groups=8,out_dim=384,use_last_ref=False,fg=fg)
        self.decoder = Pyramid3DDecoder(config=None,img_size=cfg['input_size'],use_pyramid=use_pyramid,timestep_split=True,
        shallow_decode=(4-len(cfg['depths'][:])),flow_sep_decode=True,conv_cnn=False)

        dummy_ogm =tf.zeros((1,)+cfg['input_size']+(11,2,))
        dummy_map =tf.zeros((1,)+(256,256)+(3,))

        dummy_obs_actors = tf.zeros([1,48,11,8])
        dummy_occ_actors = tf.zeros([1,16,11,8])
        dummy_ccl = tf.zeros([1,256,10,7])
        dummy_flow =tf.zeros((1,)+cfg['input_size']+(2,))
        self.ref_res = None

        self(dummy_ogm,dummy_map,obs=dummy_obs_actors,occ=dummy_occ_actors,mapt=dummy_ccl,flow=dummy_flow)
        self.summary()
    
    def call(self,ogm,map_img,training=True,obs=None,occ=None,mapt=None,flow=None,dense_vec=None,dense_map=None):

        #visual encoder:
        res_list = self.encoder(ogm,map_img,flow,training)
        q = res_list[-1]

        if self.fg_msa:
            q = tf.reshape(q,[-1,16,16,384])
            #fg-msa:
            res,pos,ref = self.fg_msa_layer(q,training=training)
            q = res + q
            q = tf.reshape(q,[-1,16*16,384])
        query = tf.repeat(tf.expand_dims(q, axis=1),repeats=8,axis=1)
        if self.fg:
            # added Projected flow-features to each timestep
            ref = tf.reshape(ref,[-1,8,256,384])
            query = ref + query
        
        #time-sep-cross attention and vector encoders:
        obs_value = self.trajnet_attn(query,obs,occ,mapt,training)

        #fpn decoding:
        y = self.decoder(obs_value,training,res_list)
        y = tf.reshape(tf.transpose(y, [0,2,3,1,4]),[-1,256,256,32])
        return y


def test_SwinT():
    gpus = tf.config.list_physical_devices('GPU')[2:]
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU(s),", len(logical_gpus), "Logical GPU(s)")
    cfg = dict(input_size=(256, 256), window_size=8, embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12])
    model = STrajNet(cfg=cfg,fg_msa=True,fg=True)

def SwinTransformer(model_name='swin_tiny_224', num_classes=1000, include_top=True, pretrained=True, use_tpu=False, cfgs=CFGS):
    cfg = cfgs[model_name]
    net = SwinTransformerModel(
        model_name=model_name, include_top=include_top, num_classes=num_classes, img_size=cfg['input_size'], window_size=cfg[
            'window_size'], embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads']
    )
    net(tf.keras.Input(shape=(cfg['input_size'][0], cfg['input_size'][1], 3)))
    if pretrained is True:
        url = f'https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/{model_name}.tgz'
        pretrained_ckpt = tf.keras.utils.get_file(
            model_name, url, untar=True)
    else:
        pretrained_ckpt = pretrained

    if pretrained_ckpt:
        if tf.io.gfile.isdir(pretrained_ckpt):
            pretrained_ckpt = f'{pretrained_ckpt}/{model_name}.ckpt'

        if use_tpu:
            load_locally = tf.saved_model.LoadOptions(
                experimental_io_device='/job:localhost')
            net.load_weights(pretrained_ckpt, options=load_locally)
        else:
            net.load_weights(pretrained_ckpt)

    return net

if __name__=='__main__':
    test_SwinT()
