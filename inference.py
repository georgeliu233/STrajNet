import tensorflow as tf
import numpy as np

from tqdm import tqdm
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from swinT import SwinTransformerEncoder , SwinTransformerDecoder,CFGS

from loss import OGMFlow_loss , OGMFlow_loss2

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2

from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
# from waymo_open_dataset.utils import occupancy_flow_metrics
import occu_metric as occupancy_flow_metrics

from google.protobuf import text_format

import csv 

import pathlib
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
import uuid
import zlib

from metrics import OGMFlowMetrics,print_metrics

layer = tf.keras.layers


gpus = tf.config.list_physical_devices('GPU')[0:1]
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus, 'GPU')


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
import os
# Hyper parameters
NUM_PRED_CHANNELS = 4

from time import time

TEST =True

feature = {
    'centerlines': tf.io.FixedLenFeature([], tf.string),
    'actors': tf.io.FixedLenFeature([], tf.string),
    'occl_actors': tf.io.FixedLenFeature([], tf.string),
    'ogm': tf.io.FixedLenFeature([], tf.string),
    'map_image': tf.io.FixedLenFeature([], tf.string),
    'scenario/id':tf.io.FixedLenFeature([], tf.string),
    'vec_flow':tf.io.FixedLenFeature([], tf.string),
    # 'byc_flow':tf.io.FixedLenFeature([], tf.string)
}
if not TEST:
    feature.update({'gt_flow': tf.io.FixedLenFeature([], tf.string),
                    'origin_flow': tf.io.FixedLenFeature([], tf.string),
                    'gt_obs_ogm': tf.io.FixedLenFeature([], tf.string),
                    'gt_occ_ogm': tf.io.FixedLenFeature([], tf.string),
                    })

def _parse_image_function_test(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  new_dict = {}
  d =  tf.io.parse_single_example(example_proto, feature)
  new_dict['centerlines'] = tf.cast(tf.reshape(tf.io.decode_raw(d['centerlines'],tf.float64),[256,10,7]),tf.float32)
  new_dict['actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['actors'],tf.float64),[48,11,8]),tf.float32)
  new_dict['occl_actors'] = tf.cast(tf.reshape(tf.io.decode_raw(d['occl_actors'],tf.float64),[16,11,8]),tf.float32)
  new_dict['ogm'] = tf.reshape(tf.cast(tf.io.decode_raw(d['ogm'],tf.bool),tf.float32),[512,512,11,2])

  new_dict['map_image'] = tf.cast(tf.reshape(tf.io.decode_raw(d['map_image'],tf.int8),[256,256,3]),tf.float32) / 256
  new_dict['vec_flow'] = tf.reshape(tf.io.decode_raw(d['vec_flow'],tf.float32),[512,512,2])
  new_dict['scenario/id'] = d['scenario/id']
  return new_dict


def _get_pred_waypoint_logits(
    model_outputs: tf.Tensor,
    mode_flow_outputs:tf.Tensor=None) -> occupancy_flow_grids.WaypointGrids:
  """Slices model predictions into occupancy and flow grids."""
  pred_waypoint_logits = occupancy_flow_grids.WaypointGrids()

  # Slice channels into output predictions.
  for k in range(config.num_waypoints):
    index = k * NUM_PRED_CHANNELS
    if mode_flow_outputs is not None:
        waypoint_channels_flow = mode_flow_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    waypoint_channels = model_outputs[:, :, :, index:index + NUM_PRED_CHANNELS]
    pred_observed_occupancy = waypoint_channels[:, :, :, :1]
    pred_occluded_occupancy = waypoint_channels[:, :, :, 1:2]
    pred_flow = waypoint_channels[:, :, :, 2:]
    if mode_flow_outputs is not None:
        pred_flow = waypoint_channels_flow[:, :, :, 2:]
    pred_waypoint_logits.vehicles.observed_occupancy.append(
        pred_observed_occupancy)
    pred_waypoint_logits.vehicles.occluded_occupancy.append(
        pred_occluded_occupancy)
    pred_waypoint_logits.vehicles.flow.append(pred_flow)

  return pred_waypoint_logits

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


print('load_model...')

from swinT import STrajNet
cfg=dict(input_size=(512,512), window_size=8, embed_dim=96, depths=[2,2,2], num_heads=[3,6,12])
model = STrajNet(cfg,sep_actors=False,defattn=True,fg=True)

def test_step(data):
    map_img = data['map_image']
    centerlines = data['centerlines']
    actors = data['actors']
    occl_actors = data['occl_actors']
    ogm = data['ogm']
    flow = data['vec_flow']

    outputs = model(ogm,map_img,training=False,obs=actors,occ=occl_actors,mapt=centerlines,flow=flow)
    logits = _get_pred_waypoint_logits(outputs)
    
    pred_waypoints = _apply_sigmoid_to_occupancy_logits(logits)

    return pred_waypoints

def _add_waypoints_to_scenario_prediction(
    pred_waypoints: occupancy_flow_grids.WaypointGrids,
    scenario_prediction: occupancy_flow_submission_pb2.ScenarioPrediction,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Add predictions for all waypoints to scenario_prediction message."""
  for k in range(config.num_waypoints):
    waypoint_message = scenario_prediction.waypoints.add()
    # Observed occupancy.
    obs_occupancy = pred_waypoints.vehicles.observed_occupancy[k].numpy()
    obs_occupancy_quantized = np.round(obs_occupancy * 255).astype(np.uint8)
    obs_occupancy_bytes = zlib.compress(obs_occupancy_quantized.tobytes())
    waypoint_message.observed_vehicles_occupancy = obs_occupancy_bytes
    # Occluded occupancy.
    occ_occupancy = pred_waypoints.vehicles.occluded_occupancy[k].numpy()
    occ_occupancy_quantized = np.round(occ_occupancy * 255).astype(np.uint8)
    occ_occupancy_bytes = zlib.compress(occ_occupancy_quantized.tobytes())
    waypoint_message.occluded_vehicles_occupancy = occ_occupancy_bytes
    # Flow.
    flow = pred_waypoints.vehicles.flow[k].numpy()
    flow_quantized = np.clip(np.round(flow), -128, 127).astype(np.int8)
    flow_bytes = zlib.compress(flow_quantized.tobytes())
    waypoint_message.all_vehicles_flow = flow_bytes

from tqdm import tqdm

def model_testing(test_shard_path,ids):
    file_name = test_shard_path.split('/')[-1]
    print(f'Creating submission for test shard {file_name}...')
    test_dataset = _make_test_dataset(test_shard_path=test_shard_path)
    submission = _make_submission_proto()

    cnt_sample = 0
    for batch in tqdm(test_dataset):
        pred_waypoints = test_step(batch)

        scenario_prediction = submission.scenario_predictions.add()
        sc_id = batch['scenario/id'].numpy()[0]
        if isinstance(sc_id, bytes):
            sc_id=str(sc_id, encoding = "utf-8") 
        scenario_prediction.scenario_id = sc_id

        assert sc_id in ids, (sc_id)

        # Add all waypoints.
        _add_waypoints_to_scenario_prediction(
            pred_waypoints=pred_waypoints,
            scenario_prediction=scenario_prediction,
            config=config)

        cnt_sample += 1
        
    _save_submission_to_file(submission,test_shard_path)

    return cnt_sample
        
def _make_submission_proto(
) -> occupancy_flow_submission_pb2.ChallengeSubmission:
    """Makes a submission proto to store predictions for one shard."""
    submission = occupancy_flow_submission_pb2.ChallengeSubmission()
    submission.account_name = ''
    submission.unique_method_name = ''
    # submission.authors.extend([''])
    submission.authors.extend([''])
    submission.description = ''
    submission.method_link = ''
    return submission

def _save_submission_to_file(
    submission: occupancy_flow_submission_pb2.ChallengeSubmission,
    test_shard_path: str,
) -> None:
    """Save predictions for one test shard as a binary protobuf."""
    # save_folder = os.path.join(pathlib.Path.home(),
    #                             'occupancy_flow_challenge/testing')
    # save_folder = os.path.join(SAVE_DIR,
    #                             '/test6')
    save_folder = args.save_dir
     
    os.makedirs(save_folder, exist_ok=True)
    basename = os.path.basename(test_shard_path)
    if 'new.tfrecords' not in basename:
        raise ValueError('Cannot determine file path for saving submission.')
    num = basename[:5]
    submission_basename = 'occupancy_flow_submission.binproto' + '-' + num + '-of-00150'

    submission_shard_file_path = os.path.join(save_folder, submission_basename)
    num_scenario_predictions = len(submission.scenario_predictions)
    print(f'Saving {num_scenario_predictions} scenario predictions to '
        f'{submission_shard_file_path}...\n')
    f = open(submission_shard_file_path, 'wb')
    f.write(submission.SerializeToString())
    f.close()

def _make_test_dataset(test_shard_path: str) -> tf.data.Dataset:
  """Makes a dataset for one shard in the test set."""
  test_dataset = tf.data.TFRecordDataset(test_shard_path)
  test_dataset = test_dataset.map(_parse_image_function_test)
  test_dataset = test_dataset.batch(1)
  return test_dataset

def id_checking(test=True):
    if val:
            path = f'{args.ids_dir}/validation_scenario_ids.txt'
    else:
        path = f'{args.ids_dir}/testing_scenario_ids.txt'

    with tf.io.gfile.GFile(path) as f:
        test_scenario_ids = f.readlines()
        test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
        print(f'original ids num:{len(test_scenario_ids)}')
        test_scenario_ids = set(test_scenario_ids)
    return test_scenario_ids

if __name__ == "__main__":
    import glob
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="./Waymo_Dataset/occupancy_flow_challenge/")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="./Waymo_Dataset/inference/")
    parser.add_argument('--file_dir', type=str, help='Test Dataset directory',default="./Waymo_Dataset/preprocessed_data/test/")
    parser.add_argument('--weight_path', type=str, help='Model weights directory',default="")
    args = parser.parse_args()
    model.load_weights(args.weight_path)

    v_filenames = tf.io.gfile.glob(args.file_dir+'/*.tfrecords')
    print(f'{len(v_filenames)} found, start loading dataset')
    test_scenario_ids = id_checking(test=TEST)
    cnt = 0
    for filename in v_filenames:
        num = model_testing(test_shard_path=filename,ids=test_scenario_ids)
        cnt += num
    print(cnt)

