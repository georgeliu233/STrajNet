import tensorflow as tf
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from loss import OGMFlow_loss
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.utils import occupancy_flow_grids
import occu_metric as occupancy_flow_metrics

from google.protobuf import text_format
import csv 
import argparse
from metrics import OGMFlowMetrics,print_metrics

layer = tf.keras.layers

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus, 'GPU')
print(len(gpus), "Physical GPU(s),")

REPLICA = len(gpus)

#configuration
config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
config_text = """
num_past_steps: 10
num_future_steps: 80
num_waypoints: 8
cumulative_waypoints: false
normalize_sdc_yaw: true
grid_height_cells: 256
grid_width_cells: 256
sdc_y_in_grid: 192
sdc_x_in_grid: 128
pixels_per_meter: 3.2
agent_points_per_side_length: 48
agent_points_per_side_width: 16
"""
text_format.Parse(config_text, config)

print(config)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--save_dir', type=str, help='saving directory',default="")
parser.add_argument('--file_dir', type=str, help='Training Val Dataset directory',default="./Waymo_Dataset/preprocessed_data")
parser.add_argument('--model_path', type=str, help='loaded weight path',default=None)
parser.add_argument('--batch_size', type=int, help='batch_size',default=16)
parser.add_argument('--epochs', type=int, help='training eps',default=15)
parser.add_argument('--lr', type=float, help='initial learning rate',default=1e-4)
args = parser.parse_args()

# Hyper parameters
NUM_PRED_CHANNELS = 4
BATCH_SIZE = args.batch_size
EPOCH = args.epochs
LR = args.lr
SAVE_DIR = args.save_dir

import os
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

from time import time

strategy = tf.distribute.MirroredStrategy()

feature = {
    'centerlines': tf.io.FixedLenFeature([], tf.string),
    'actors': tf.io.FixedLenFeature([], tf.string),
    'occl_actors': tf.io.FixedLenFeature([], tf.string),
    'ogm': tf.io.FixedLenFeature([], tf.string),
    'map_image': tf.io.FixedLenFeature([], tf.string),
    'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
    'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
    'gt_flow': tf.io.FixedLenFeature([], tf.string),
    'origin_flow': tf.io.FixedLenFeature([], tf.string),
    'vec_flow':tf.io.FixedLenFeature([], tf.string),
    # 'byc_flow':tf.io.FixedLenFeature([], tf.string)
}

def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  new_dict = {}
  d =  tf.io.parse_single_example(example_proto, feature)
  new_dict['centerlines'] = tf.cast(tf.reshape(tf.io.decode_raw(d['centerlines'],tf.float64),[256,10,7]),tf.float32)
  new_dict['actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['actors'],tf.float64),[48,11,8]),tf.float32)
  new_dict['occl_actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['occl_actors'],tf.float64),[16,11,8]),tf.float32)

  new_dict['gt_flow'] = tf.reshape(tf.io.decode_raw(d['gt_flow'],tf.float32),[8,512,512,2])[:,128:128+256,128:128+256,:]
  new_dict['origin_flow'] = tf.reshape(tf.io.decode_raw(d['origin_flow'],tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]

  new_dict['ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['ogm'],tf.bool),tf.float32),[512,512,11,2])

  new_dict['gt_obs_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_obs_ogm'],tf.bool),tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]
  new_dict['gt_occ_ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['gt_occ_ogm'],tf.bool),tf.float32),[8,512,512,1])[:,128:128+256,128:128+256,:]

  new_dict['map_image'] = tf.cast(tf.reshape(tf.io.decode_raw(d['map_image'],tf.int8),[256,256,3]),tf.float32) / 256
  new_dict['vec_flow'] = tf.reshape(tf.io.decode_raw(d['vec_flow'],tf.float32),[512,512,2])
  return new_dict

def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy)
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy)
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits


def _warpped_gt(
    gt_ogm: tf.Tensor,
    gt_occ: tf.Tensor,
    gt_flow: tf.Tensor,
    origin_flow: tf.Tensor,) -> occupancy_flow_grids.WaypointGrids:

    true_waypoints = occupancy_flow_grids.WaypointGrids()

    for k in range(8):
        true_waypoints.vehicles.observed_occupancy.append(gt_ogm[:,k])
        true_waypoints.vehicles.occluded_occupancy.append(gt_occ[:,k])
        true_waypoints.vehicles.flow.append(gt_flow[:,k])
        true_waypoints.vehicles.flow_origin_occupancy.append(origin_flow[:,k])
    
    return true_waypoints

def _apply_sigmoid_to_occupancy_logits(
    pred_waypoint_logits: occupancy_flow_grids.WaypointGrids
) -> occupancy_flow_grids.WaypointGrids:
  """Converts occupancy logits with probabilities."""
  pred_waypoints = occupancy_flow_grids.WaypointGrids()
  pred_waypoints.vehicles.observed_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.observed_occupancy
  ]
  pred_waypoints.vehicles.occluded_occupancy = [
      tf.sigmoid(x) for x in pred_waypoint_logits.vehicles.occluded_occupancy
  ]
  pred_waypoints.vehicles.flow = pred_waypoint_logits.vehicles.flow
  return pred_waypoints

no_warp=False

with strategy.scope():
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss_occ = tf.keras.metrics.Mean(name='train_loss_occ')
    train_loss_flow = tf.keras.metrics.Mean(name='train_loss_flow')
    train_loss_warp = tf.keras.metrics.Mean(name='train_loss_wp')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_loss_occ = tf.keras.metrics.Mean(name='valid_loss_occ')
    valid_loss_flow = tf.keras.metrics.Mean(name='valid_loss_flow')
    valid_loss_warp = tf.keras.metrics.Mean(name='valid_loss_wp')

    train_metrics = OGMFlowMetrics(preflix='train')
    valid_metrics = OGMFlowMetrics(preflix='val')
    def val_metric_func(config,true_waypoints,pred_waypoints):
        return occupancy_flow_metrics.compute_occupancy_flow_metrics(
        config=config,
        true_waypoints=true_waypoints,
        pred_waypoints=pred_waypoints,
        no_warp=False
        )


print('load_model...')

from modules import STrajNet
cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
from lr_schedule import CustomSchedule,CosineDecayRestarts
schedule = CosineDecayRestarts(initial_learning_rate=LR,
    first_decay_steps=int(30438*1.5),t_mul=1.25,m_mul=0.99,alpha=0)

ogm_weight = 1000.0
occ_weight = 1000.0
flow_origin_weight = 1000.0
flow_weight = 1.0

with strategy.scope():
    model = STrajNet(cfg,actor_only=True,sep_actors=False)
    loss_fn = OGMFlow_loss(config,replica=REPLICA,no_use_warp=False,use_pred=False,use_gt=True,
    ogm_weight=ogm_weight, occ_weight=occ_weight,flow_origin_weight=flow_origin_weight,flow_weight=flow_weight,use_focal_loss=False)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=LR) 

@tf.function
def train_step(data):

    map_img = data['map_image']
    centerlines = data['centerlines']
    actors = data['actors']
    occl_actors = data['occl_actors']

    ogm = data['ogm']
    gt_obs_ogm = data['gt_obs_ogm']
    gt_occ_ogm = data['gt_occ_ogm']
    gt_flow = data['gt_flow']
    origin_flow = data['origin_flow']

    flow = data['vec_flow']

    true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm,gt_occ=gt_occ_ogm,gt_flow=gt_flow,origin_flow=origin_flow)

    with tf.GradientTape() as tape:
        outputs = model(ogm,map_img,training=True,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
        logits = _get_pred_waypoint_logits(outputs)
        loss_dict = loss_fn(true_waypoints=true_waypoints,pred_waypoint_logits=logits,curr_ogm=ogm[:,:,:,-1,0])
        loss_value = tf.math.add_n(loss_dict.values())

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss.update_state(loss_dict['observed_xe']*REPLICA)
    train_loss_occ.update_state(loss_dict['occluded_xe']*REPLICA)
    train_loss_flow.update_state(loss_dict['flow']*REPLICA)
    train_loss_warp.update_state(loss_dict['flow_warp_xe']*REPLICA)
    return outputs

def train_metric_function(data,outputs):

    gt_obs_ogm = data['gt_obs_ogm']
    gt_occ_ogm = data['gt_occ_ogm']
    gt_flow = data['gt_flow']
    origin_flow = data['origin_flow']

    true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm,gt_occ=gt_occ_ogm,gt_flow=gt_flow,origin_flow=origin_flow)
    logits = _get_pred_waypoint_logits(outputs)
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
    config=config,
    true_waypoints=true_waypoints,
    pred_waypoints=pred_waypoints,
    no_warp=no_warp
    )
    train_metrics.update_state(metrics)

# @tf.function
def val_step(data):

    map_img = data['map_image']
    centerlines = data['centerlines']
    actors = data['actors']
    occl_actors = data['occl_actors']

    ogm = data['ogm']
    gt_obs_ogm = data['gt_obs_ogm']
    gt_occ_ogm = data['gt_occ_ogm']
    gt_flow = data['gt_flow']
    origin_flow = data['origin_flow']

    flow = data['vec_flow']

    true_waypoints = _warpped_gt(gt_ogm=gt_obs_ogm,gt_occ=gt_occ_ogm,gt_flow=gt_flow,origin_flow=origin_flow)

    outputs = model(ogm,map_img,training=False,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
    logits = _get_pred_waypoint_logits(outputs)
    
    loss_dict = loss_fn(true_waypoints=true_waypoints,pred_waypoint_logits=logits,curr_ogm=ogm[:,:,:,-1,0])
    loss_value = tf.math.add_n(loss_dict.values())

    valid_loss.update_state(loss_dict['observed_xe']*REPLICA)
    valid_loss_occ.update_state(loss_dict['occluded_xe']*REPLICA)
    valid_loss_flow.update_state(loss_dict['flow']*REPLICA)
    valid_loss_warp.update_state(loss_dict['flow_warp_xe']*REPLICA)
    
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)
    metrics = val_metric_func(config,true_waypoints,pred_waypoints)
    valid_metrics.update_state(metrics)

def val_metric_function(true_waypoints,pred_waypoints):
    metrics = occupancy_flow_metrics.compute_occupancy_flow_metrics(
    config=config,
    true_waypoints=true_waypoints,
    pred_waypoints=pred_waypoints,
    no_warp=no_warp
    )
    valid_metrics.update_state(metrics)


def model_training(train_dataset, valid_dataset, epochs,continue_ep=0):
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
    training_samples = None
    val_samples = None
    training_losses = []
    validation_losses = []
    lr_cnt = 0.
    lr_cnt = lr_cnt
    
    for epoch in range(epochs):
        if epoch<continue_ep:
            print("\nskip epoch {}/{}".format(epoch+1, epochs))
            lr_cnt += 30438
            continue
        
        print("\nepoch {}/{}".format(epoch+1, epochs))
        
        progBar = tf.keras.utils.Progbar(training_samples, stateful_metrics=['obs_loss','occ_loss','flow_loss','warp_loss'], unit_name='sample')
        vprogBar = tf.keras.utils.Progbar(val_samples, stateful_metrics=['obs_loss','occ_loss','flow_loss','warp_loss',
        'epe','obs_auc','occ_auc','flowogm_auc'], unit_name='sample')

        # Iterate over the batches of the training dataset.
        for step, batch in enumerate(train_dataset):
            training_samples = (step+1) * BATCH_SIZE
            outputs = strategy.run(train_step,args=(batch,))
            progBar.update((step+1) * BATCH_SIZE, values=[('obs_loss', train_loss.result()/ogm_weight),('occ_loss', train_loss_occ.result()/occ_weight),
            ('flow_loss', train_loss_flow.result()/flow_weight),('warp_loss', train_loss_warp.result()/flow_origin_weight)])

        # Iterate over the batches of the validation dataset. 
        if valid_dataset is not None:
            for step, batch in enumerate(valid_dataset):
                val_samples = (step+1) * BATCH_SIZE
                strategy.run(val_step,args=(batch,))
                vprogBar.update((step+1) * BATCH_SIZE, values=[
                    ('obs_loss', valid_loss.result()/ogm_weight),('occ_loss',valid_loss_occ.result()/occ_weight),
                ('flow_loss', valid_loss_flow.result()/flow_weight),('warp_loss', valid_loss_warp.result()/flow_origin_weight),('flowogm_auc',valid_metrics.flow_ogm_auc.result())
                ])

            # Display metrics at the end of testing.
            val_res_dict = valid_metrics.get_result()
            print_metrics(val_res_dict,'val')

            # Log training results every epoch
            training_losses.append(train_loss.result().numpy())
            validation_losses.append(valid_loss.result().numpy())

        log = {'epoch': epoch+1, 'loss': train_loss.result().numpy(), 'val_loss': valid_loss.result().numpy(), 'lr': optimizer.lr.numpy()}

        # log.update(train_res_dict)

        if valid_dataset is not None:
            log.update(val_res_dict)

        if epoch == 0:
            with open(f'{SAVE_DIR}/train_log.csv','w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'{SAVE_DIR}/train_log.csv','a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())
        
        model.save_weights('{}/model_{}_{:.4f}_{:.4f}.tf'.format(SAVE_DIR,epoch+1, train_loss.result(), valid_loss.result()))
        
        # Clear metrics
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metrics.reset_states()
        valid_metrics.reset_states()
    
    model.save_weights(f'{SAVE_DIR}/final_model.tf')

if __name__ == "__main__":
    import glob
    weight_path = args.model_path
    if weight_path is not None:
        model.load_weights(weight_path)
        continue_ep = int(weight_path.split('/')[-1].split('_')[1])
        print(f'Continue_training...ep:{continue_ep+1}')
    else:
        continue_ep = 0

    filenames = tf.io.matching_files(f'{args.file_dir}/train/*.tfrecords')
    print(f'{len(filenames)} found, start loading dataset')
    train_dataset = tf.data.TFRecordDataset(filenames, compression_type='')
    train_dataset = train_dataset.shuffle(64,reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(_parse_image_function_test)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    v_filenames = tf.io.matching_files(f'{args.file_dir}/val/*.tfrecords')
    print(f'{len(v_filenames)} found, start loading dataset')
    valid_dataset = tf.data.TFRecordDataset(v_filenames, compression_type='')
    valid_dataset = valid_dataset.map(_parse_image_function)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)

    print('file loaded! start training...')
    model_training(train_dataset, valid_dataset, EPOCH,continue_ep)
