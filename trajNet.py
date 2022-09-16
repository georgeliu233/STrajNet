import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import tensorflow_addons as tfa

class MapEncoder(tf.keras.layers.Layer):
    def __init__(self,num_heads=4,out_dim=256):
        super(MapEncoder, self).__init__()
        self.node_feature = tf.keras.layers.Conv1D(64, 1,activation='elu')
        self.node_attention = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=64, dropout=0.1, output_size=64*4)
        self.flatten = tf.keras.layers.GlobalMaxPooling1D()
        self.vector_feature = tf.keras.layers.Dense(64,use_bias=False)
        self.sublayer = tf.keras.layers.Dense(out_dim, activation='elu')

    def call(self, inputs, mask,training=True):
        mask = tf.cast(mask, tf.int32)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :4])
        nodes = self.node_attention(inputs=[nodes, nodes,nodes], mask=mask,training=training)
        nodes = self.flatten(nodes)
        vector = self.vector_feature(inputs[:, 0, 4:])
        out = tf.concat([nodes, vector], axis=1)
        polyline_feature = self.sublayer(out)

        return polyline_feature


class TrajEncoder(tf.keras.layers.Layer):
    def __init__(self,num_heads=4,out_dim=256):
        super(TrajEncoder, self).__init__()
        self.node_feature = tf.keras.layers.Conv1D(64, 1,activation='elu')
        self.node_attention = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=64, dropout=0.1, output_size=64*5)
        self.flatten = tf.keras.layers.GlobalMaxPooling1D()
        self.vector_feature = tf.keras.layers.Dense(64,use_bias=False)
        self.sublayer = tf.keras.layers.Dense(out_dim, activation='elu')

    def call(self, inputs, mask,training=True):
        mask = tf.cast(mask, tf.int32)
        mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        nodes = self.node_feature(inputs[:, :, :5])
        nodes = self.node_attention(inputs=[nodes, nodes,nodes], mask=mask,training=training)
        nodes = self.flatten(nodes)
        vector = self.vector_feature(inputs[:, 0, 5:])
        out = tf.concat([nodes, vector], axis=1)
        polyline_feature = self.sublayer(out)

        return polyline_feature

class TrajEncoderLSTM(tf.keras.layers.Layer):
    def __init__(self,out_dim=256):
        super(TrajEncoderLSTM, self).__init__() 
        self.embed = tf.keras.layers.Conv1D(64, 1, activation='elu')
        self.extract = tf.keras.layers.LSTM(out_dim)
        # self.vector_feature = tf.keras.layers.Dense(64,use_bias=False)
        
    
    def call(self, inputs,mask=None,training=None):
        x = inputs
        x = self.embed(x)
        x = self.extract(x)
        return x

    
class Cross_Attention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim,conv_attn=False):
        super(Cross_Attention, self).__init__()
        if conv_attn:
            self.mha = ConvMultiHeadAttention(dim=key_dim, num_heads=num_heads, output_dim=key_dim,dropout=0.1)
        else:
            self.mha = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim, dropout=0.1)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.FFN1 = tf.keras.layers.Dense(4*key_dim, activation='elu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.FFN2 = tf.keras.layers.Dense(key_dim)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, query, key, mask=None, training=True):
        value= self.mha(inputs=[query, key],mask=mask,training=training)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value, training=training)
        value = self.FFN2(value)
        value = self.dropout2(value, training=training)
        value = self.norm2(value)
        return value



class TrajNet(tf.keras.Model):
    def __init__(self,cfg,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,no_attn=False,
        double_net=False):
        super(TrajNet, self).__init__()

        self.actor_only = actor_only
        self.obs_actors=obs_actors
        self.occ_actors=occ_actors
        self.double_net = double_net
        
        self.traj_encoder = TrajEncoder(num_heads=cfg['traj_heads'],out_dim=cfg['out_dim'])
        # self.traj_encoder  = TrajEncoderLSTM(cfg['out_dim'])
        self.no_attn = no_attn
        if not no_attn:
            if double_net:
                self.cross_attention = [Cross_AttentionT(num_heads=cfg['att_heads'],key_dim=192,output_dim=cfg['out_dim']) for _ in range(2)]
            else:
                self.cross_attention = Cross_Attention(num_heads=cfg['att_heads'], key_dim=cfg['out_dim'])

        self.obs_norm = tf.keras.layers.LayerNormalization()
        self.occ_norm = tf.keras.layers.LayerNormalization()
        # self.obs_drop = tf.keras.layers.Dropout(0.1)
        # self.occ_drop = tf.keras.layers.Dropout(0.1)

        dummy_obs_actors = tf.zeros([1,obs_actors,past_to_current_steps,8])
        dummy_occ_actors = tf.zeros([1,occ_actors,past_to_current_steps,8])
        # dummy_ccl = tf.zeros([1,256,10,7])

        self.bi_embed = tf.cast(tf.repeat([[1,0],[0,1]], repeats=[obs_actors,occ_actors],axis=0),tf.float32)
        self.seg_embed = tf.keras.layers.Dense(cfg['out_dim'],use_bias=False)

        self(dummy_obs_actors,dummy_occ_actors)
        self.summary()
    
    def call(self,obs_traj,occ_traj,map_traj=None,training=True):

        obs_mask = tf.not_equal(obs_traj, 0)[:,:,:,0]
        obs = [self.traj_encoder(obs_traj[:, i],obs_mask[:,i],training) for i in range(self.obs_actors)]
        obs = tf.stack(obs,axis=1)

        occ_mask = tf.not_equal(occ_traj, 0)[:,:,:,0]
        occ = [self.traj_encoder(occ_traj[:, i],occ_mask[:,i],training) for i in range(self.occ_actors)]
        occ = tf.stack(occ,axis=1)

        embed = tf.repeat([self.bi_embed], repeats=occ.get_shape().as_list()[0],axis=0)
        embed = self.seg_embed(embed)

        c_attn_mask = tf.not_equal(tf.reduce_sum(tf.cast(tf.concat([obs_mask,occ_mask], axis=1),tf.int32),axis=-1),0) #[batch,64] (last step denote the current)
        c_attn_mask = tf.cast(c_attn_mask, tf.int32)

        if self.no_attn:
            if self.double_net:
                concat_actors = tf.concat([obs,occ], axis=1)
                obs = self.obs_norm(concat_actors+embed)
                occ = self.occ_norm(concat_actors+embed)
                return obs,occ,c_attn_mask
            else:
                return self.obs_norm(obs + embed[:,:self.obs_actors,:]),self.occ_norm(occ + embed[:,self.obs_actors:,:]),c_attn_mask

        # interactions given seg_embedding
        concat_actors = tf.concat([obs,occ], axis=1)
        concat_actors = tf.multiply(tf.cast(c_attn_mask[:, :, tf.newaxis],tf.float32), concat_actors)
        query = concat_actors + embed

        attn_mask = tf.matmul(c_attn_mask[:, :, tf.newaxis], c_attn_mask[:, tf.newaxis, :]) #[batch,64,64]

        if self.double_net:
            value = self.cross_attention[0](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs,val_occ = value[:,:self.obs_actors,:] , value[:,self.obs_actors:,:]

            value_flow = self.cross_attention[1](query=query, key=concat_actors, mask=attn_mask, training=training)
            val_obs_f,val_occ_f = value_flow[:,:self.obs_actors,:] , value_flow[:,self.obs_actors:,:]

            obs = obs + val_obs
            occ = occ + val_occ

            ogm = tf.concat([obs,occ], axis=1) + embed

            obs_f = obs + val_obs_f
            occ_f = occ + val_occ_f

            flow = tf.concat([obs_f,occ_f], axis=1) + embed

            return self.obs_norm(ogm) , self.occ_norm(flow) , c_attn_mask
        
        value = self.cross_attention(query=query, key=concat_actors, mask=attn_mask, training=training)
        val_obs,val_occ = value[:,:self.obs_actors,:] , value[:,self.obs_actors:,:]

        obs = obs + val_obs
        occ = occ + val_occ

        concat_actors = tf.concat([obs,occ], axis=1)

        obs = self.obs_norm(obs + embed[:,:self.obs_actors,:])
        occ = self.occ_norm(occ + embed[:,self.obs_actors:,:])

        return obs,occ,c_attn_mask

class Cross_AttentionT(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim,output_dim,conv_attn=False,sep_actors=False):
        super(Cross_AttentionT, self).__init__()
        if conv_attn:
            self.mha = ConvMultiHeadAttention(dim=key_dim, num_heads=num_heads, output_dim=key_dim,dropout=0.1)
        else:
            self.mha = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim,dropout=0.1)
        self.sep_actors = sep_actors
        if sep_actors:
            self.actor_mha = tfa.layers.MultiHeadAttention(num_heads=num_heads, head_size=key_dim//num_heads,output_size=key_dim,dropout=0.1)
            self.actor_norm = tf.keras.layers.LayerNormalization()
            self.actor_norm2 = tf.keras.layers.LayerNormalization()
            self.aFFN1 = tf.keras.layers.Dense(4*key_dim, activation='elu')
            self.adropout1 = tf.keras.layers.Dropout(0.1)
            self.aFFN2 = tf.keras.layers.Dense(output_dim)
            self.adropout2 = tf.keras.layers.Dropout(0.1)
            
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.FFN1 = tf.keras.layers.Dense(4*key_dim, activation='elu')
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.FFN2 = tf.keras.layers.Dense(output_dim)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.conv_attn = conv_attn

    def call(self, query, key, mask, training=True,actor_mask=None):
        if self.sep_actors:
            org = key
            key = self.actor_mha(inputs=[key, key],mask=actor_mask,training=training)
            key = self.actor_norm(key)
            key = self.aFFN1(key)
            key = self.adropout1(key,training=training)
            key = self.aFFN2(key)
            key = self.adropout2(key,training=training)
            key = self.actor_norm2(key + org)
        if not self.conv_attn:
            value= self.mha(inputs=[query, key],mask=mask,training=training)
        else:
            value= self.mha(query, key,mask=mask,training=training)
        value = self.norm1(value)
        value = self.FFN1(value)
        value = self.dropout1(value, training=training)
        value = self.FFN2(value)
        value = self.dropout2(value, training=training)
        value = self.norm2(value)
        return value

class TrajNetCrossAttention(tf.keras.Model):
    def __init__(self,traj_cfg,pic_size=(8,8),pic_dim=768,past_to_current_steps=11,obs_actors=48,occ_actors=16,actor_only=True,
        multi_modal=True,sep_actors=False):
        super(TrajNetCrossAttention, self).__init__()

        self.traj_net = TrajNet(traj_cfg,no_attn=traj_cfg['no_attn'],
            past_to_current_steps=past_to_current_steps,obs_actors=obs_actors,occ_actors=occ_actors,actor_only=actor_only,
            double_net=False)

        self.obs_actors = obs_actors
        self.H, self.W = pic_size
        self.pic_dim = pic_dim

        self.multi_modal = multi_modal
        self.actor_only = actor_only
        self.sep_actors = sep_actors
  
        if actor_only==False:
            self.map_encoder = MapEncoder(num_heads=traj_cfg['traj_heads'],out_dim=traj_cfg['out_dim'])
            self.map_norm = tf.keras.layers.LayerNormalization()
            self.map_cross_attn = [Cross_AttentionT(num_heads=3, output_dim=pic_dim,key_dim=128,sep_actors=sep_actors) for _ in range(8)]
        self.cross_attn_obs = [Cross_AttentionT(num_heads=3, output_dim=pic_dim,key_dim=128,sep_actors=sep_actors) for _ in range(8)]

        dummy_obs_actors = tf.zeros([1,obs_actors,past_to_current_steps,8])
        dummy_occ_actors = tf.zeros([1,occ_actors,past_to_current_steps,8])
        dummy_ccl = tf.zeros([1,256,10,7])
        dummy_pic_encode = tf.zeros((1,) + pic_size + (pic_dim,))
        
        flow_pic_encode = tf.zeros((1,) + pic_size + (pic_dim,))
        if multi_modal:
            dummy_pic_encode = tf.zeros((1,8,) + pic_size + (pic_dim,))

        self(dummy_pic_encode,dummy_obs_actors,dummy_occ_actors,dummy_ccl,flow_pic_encode=flow_pic_encode)
        self.summary()
    
    def map_encode(self,map_traj,training):

        segs = map_traj.get_shape()[1]
        map_mask = tf.not_equal(map_traj[:,:,:,0], 0) #[batch,256,10]
        amap_mask = tf.reshape(map_mask,[-1,10])
        map_traj = tf.reshape(map_traj, [-1,10,7])
        map_enc = self.map_encoder(map_traj,amap_mask,training)
        map_enc = tf.reshape(map_enc,[-1,256,map_enc.get_shape()[-1]])

        map_mask = tf.cast(map_mask[:,:,0],tf.int32)#[batch,256]
        return map_enc,map_mask


    def call(self,pic_encode,obs_traj,occ_traj,map_traj=None,training=True,flow_pic_encode=None):

        obs,occ,traj_mask = self.traj_net(obs_traj,occ_traj,map_traj,training)

        if self.sep_actors:
            actor_mask = tf.matmul(traj_mask[:, :, tf.newaxis], traj_mask[:, tf.newaxis, :])
        
        flat_encode = tf.reshape(pic_encode, shape=[-1,8,self.H*self.W,self.pic_dim])
        pic_mask = tf.ones_like(flat_encode[:,0,:,0],tf.int32)


        if not self.actor_only:
            map_enc,map_mask = self.map_encode(map_traj,training)
            map_enc = self.map_norm(map_enc)
            map_attn_mask = tf.matmul(pic_mask[:, :, tf.newaxis], map_mask[:, tf.newaxis, :])

        obs_attn_mask = tf.matmul(pic_mask[:, :, tf.newaxis], traj_mask[:, tf.newaxis, :])

        query = flat_encode
        key = tf.concat([obs,occ], axis=1)
        res_list = []
        for i in range(8):
            if self.sep_actors:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training,actor_mask)
            else:
                o = self.cross_attn_obs[i](query[:,i],key,obs_attn_mask,training)
            v = o + query[:,i]
            if not self.actor_only:
                v = self.map_cross_attn[i](o,map_enc,map_attn_mask,training)
                v = v + o + query[:,i]
            res_list.append(v)
            
        obs_value = tf.stack(res_list,axis=1)
        obs_value = tf.reshape(obs_value, shape=[-1,8,self.H,self.W,self.pic_dim])

        return obs_value

def test_emb():
    seg = tf.repeat([[1,0],[0,1]], repeats=[48,16],axis=0)
    seg = tf.repeat([seg], repeats=16,axis=0)
    seg = tf.cast(seg,tf.float32)
    print(seg.get_shape())

def test_trajnet():
    cfg = dict(traj_heads=4,att_heads=4,out_dim=256)
    model = TrajNet(cfg)

def testTrajC():
    cfg = dict(traj_heads=4,att_heads=4,out_dim=256,no_attn=False)
    model = TrajNetCrossAttention(cfg)

if __name__=='__main__':
    traj_cfg = dict(traj_heads=4,att_heads=6,out_dim=384,no_attn=False)
    testTrajC()





        
