
import tensorflow as tf
from occu_metric import sample
import numpy as np
layers = tf.keras.layers

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

class FGMSA(tf.keras.Model):
    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups=6,
        attn_drop=0., proj_drop=0., stride=1, 
        offset_range_factor=2, use_pe=True, dwc_pe=False,
        no_off=False, fixed_pe=False, stage_idx=3,use_last_ref=False,
        out_dim=384,fg=False,in_dim=384
    ):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.use_last_ref = use_last_ref
        self.fg = fg

        self.ref_res = None
        
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset_0 = layers.Conv2D(self.nc, kernel_size=kk, strides=stride, padding="same", groups=self.n_groups)
        self.conv_norm = layers.LayerNormalization()
        # self.out_norm = layers.LayerNormalization(1e-5)
        self.conv_offset_proj = layers.Conv2D(2, kernel_size=1,strides=1, use_bias=False)
        if self.fg:
            self.conv_offset_proj2 = layers.Conv2D(out_dim, kernel_size=1,strides=1)

        self.proj_q = layers.Conv2D(self.nc,kernel_size=1, strides=1)

        self.proj_k = layers.Conv2D(self.nc,kernel_size=1, strides=1)

        self.proj_v = layers.Conv2D(self.nc,kernel_size=1, strides=1)

        self.proj_out = layers.Conv2D(out_dim,kernel_size=1, strides=1)

        self.proj_drop = layers.Dropout(proj_drop)
        self.attn_drop = layers.Dropout(attn_drop)

        if self.use_pe:
            self.rpe_table = self.add_weight(name='warp_attn_rel_table',
                                            shape=(self.kv_h * 2 - 1, self.kv_w * 2 - 1,self.n_heads),
                                            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01, seed=None), 
                                            trainable=True)
        else:
            self.rpe_table = None
        
        dummy_x = tf.zeros((1,self.q_h,self.q_w,in_dim))
        self.in_dim = in_dim
        self.out_dim = out_dim
        ref = tf.zeros((1*self.n_groups,self.q_h,self.q_w,2))
        self(dummy_x,last_reference=ref)
        self.summary()
    
    def _get_offset(self,x):
        x = self.conv_offset_0(x)
        x = tf.reshape(x,[-1,self.q_h*self.q_w,self.nc])
        x = self.conv_norm(x)
        x = tf.reshape(x,[-1,self.q_h,self.q_w,self.nc])
        x = Gelu(x)
        x = tf.reshape(tf.transpose(tf.reshape(x,[-1,self.q_h,self.q_w,self.n_groups,self.n_group_channels]),[0,3,1,2,4]),[-1,self.q_h,self.q_w,self.n_group_channels])
        x = self.conv_offset_proj(x)
        return x
        
    
    def _get_ref_points(self, H_key, W_key, B):
        ref_y, ref_x = tf.meshgrid(
            tf.range(H_key), 
            tf.range(W_key)
        )
        ref = tf.stack((ref_y, ref_x), -1)
        ref = tf.cast(ref,tf.float32)
        ref = tf.repeat(ref[tf.newaxis,...],repeats=B*self.n_groups,axis=0)
        ref = tf.stop_gradient(ref)
        return ref

    def call(self, x,training=True,last_reference=None):

        B, H, W,C = x.get_shape().as_list()
        q = self.proj_q(x)
        offset = self._get_offset(q)
        _,Hk,Wk,_ = offset.get_shape().as_list()
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = tf.reshape(tf.constant([Hk/2,Wk/2]),(1, 1, 1, 2))
            offset = tf.nn.tanh(offset)
            offset = tf.multiply(offset, offset_range)
            self.ref_res = tf.reshape(offset,(B,self.n_groups, Hk, Wk, 2))
        
        if self.fg:
            time_offset = tf.reshape(offset,(B,self.n_groups, Hk, Wk, 2))
            flow_hidden = self.conv_offset_proj2(time_offset)
            flow_hidden = tf.reshape(flow_hidden,(B,self.n_groups, Hk,Wk, self.out_dim))
        
        if self.use_last_ref:
            reference = tf.reshape(last_reference,(B*self.n_groups, Hk, Wk, 2))
        else:
            reference = self._get_ref_points(Hk, Wk, B)
            
        if self.no_off:
            offset = tf.zeros_like(offset)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = tf.nn.tanh(offset + reference)
        
        x = tf.reshape(tf.transpose(tf.reshape(x,[B, H, W,self.n_groups,self.n_group_channels]),[0,3,1,2,4]),[B*self.n_groups, H, W,self.n_group_channels])
        
        warp = tf.concat([pos[...,1][...,tf.newaxis],pos[...,0][...,tf.newaxis]],axis=-1)
        x_sampled = sample(image=x, warp=warp,pixel_type=0)
        x_sampled = tf.reshape(tf.transpose(tf.reshape(x, [B,self.n_groups, H, W,self.n_group_channels]),[0,2,3,1,4]),[B,n_sample,1,C])
            
        q = tf.reshape(tf.transpose(tf.reshape(q,(B, H * W,self.n_heads, self.n_head_channels)),[0,2,1,3]),[B*self.n_heads,H * W,self.n_head_channels])
        k = tf.reshape(tf.transpose(tf.reshape(self.proj_k(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)),[0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        v = tf.reshape(tf.transpose(tf.reshape(self.proj_v(x_sampled),(B, n_sample,self.n_heads, self.n_head_channels)),[0,2,1,3]),[B*self.n_heads,n_sample,self.n_head_channels])
        attn = tf.einsum('bqc, bkc-> bqk', q, k)
        attn = attn * self.scale
        
        if self.use_pe:
            rpe_table = self.rpe_table
            # rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
            rpe_bias = tf.repeat(rpe_table[tf.newaxis,...],repeats=B,axis=0)
            
            q_grid = self._get_ref_points(H, W, B)
            
            displacement = tf.expand_dims(tf.reshape(q_grid,(B * self.n_groups, H * W, 2)),axis=2) - tf.expand_dims(tf.reshape(pos,(B * self.n_groups, n_sample, 2)),axis=1)

            rpe_bias = tf.transpose(tf.reshape(rpe_bias, (B,2 * H - 1, 2 * W - 1,self.n_groups, self.n_group_heads)),[0,3,1,2,4])
            displacement = tf.concat([displacement[...,1][...,tf.newaxis],displacement[...,0][...,tf.newaxis]],axis=-1)
            
            attn_bias = sample(
                image=tf.reshape(rpe_bias,[B * self.n_groups,2 * H - 1, 2 * W - 1,self.n_group_heads]),
                warp=displacement,
                pixel_type=0
            )

            attn_bias = tf.transpose(tf.reshape(attn_bias, [B*self.n_groups,H*W,n_sample,self.n_group_heads]),[0,3,1,2])
            
            attn_bias = tf.reshape(attn_bias,[B*self.n_heads,H*W,n_sample] )
            
            attn = attn + attn_bias

        attn = tf.nn.softmax(attn, axis=2)
        attn = self.attn_drop(attn,training=training)
        out = tf.einsum('bkv, bvc -> bck', attn, v)
        out = tf.transpose(tf.reshape(out,(B,C,H,W)),[0,2,3,1])
        y = self.proj_drop(self.proj_out(out),training=training)

        if self.fg:
            return y,tf.reshape(pos,(B, self.n_groups, Hk, Wk, 2)),flow_hidden
        
        return y, tf.reshape(pos,(B, self.n_groups, Hk, Wk, 2)), tf.reshape(reference,(B, self.n_groups, Hk, Wk, 2))


if __name__=='__main__':
    FGMSA(q_size=(32,32), kv_size=(32,32), n_heads=8, n_head_channels=24,n_groups=8,in_dim=192,out_dim=192)