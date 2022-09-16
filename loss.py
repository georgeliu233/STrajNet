import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
# from waymo_open_dataset.utils import occupancy_flow_metrics
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from occu_metric import sample,_compute_occupancy_auc

import tensorflow_addons as tfa


NUM_PRED_CHANNELS = 4

class OGMFlow_loss(tf.keras.losses.Loss):

    def __init__(self, config, ogm_weight=1000.0,occ_weight=1000.0,flow_weight=1.0,replica=1.0,flow_origin_weight=1000.0,no_use_warp=False,use_pred=False,
    use_focal_loss=True,use_gt=False):

        self.config = config
        self.ogm_weight = ogm_weight
        self.flow_weight = flow_weight
        self.occ_weight = occ_weight
        self.replica = replica
        self.focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        self.occlude_focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
        self.no_use_warp = no_use_warp
        self.use_focal_loss = use_focal_loss
        self.use_pred = use_pred
        self.flow_origin_weight = flow_origin_weight
        self.flow_focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.NONE)
        self.use_gt = use_gt
        self.auc =  tf.keras.metrics.AUC(
                num_thresholds=100,
                summation_method='interpolation',
                curve='PR',
            )
    
    def warping_preparation(self):
        pass

    def __call__(self,
        pred_waypoint_logits: occupancy_flow_grids.WaypointGrids,
        true_waypoints:occupancy_flow_grids.WaypointGrids,
        curr_ogm:tf.Tensor,
        # gt_ogm: tf.Tensor,
        # gt_occ: tf.Tensor,
        # gt_flow: tf.Tensor,
    ) -> Dict[str, tf.Tensor]:
        """Loss function.

        Args:
            config: OccupancyFlowTaskConfig proto message.
            true_waypoints: Ground truth labels.
            pred_waypoint_logits: Predicted occupancy logits and flows.

        Returns:
            A dict containing different loss tensors:
            observed_xe: Observed occupancy cross-entropy loss.
            occluded_xe: Occluded occupancy cross-entropy loss.
            flow: Flow loss.
        """
        loss_dict = {}
        # Store loss tensors for each waypoint and average at the end.
        loss_dict['observed_xe'] = []
        loss_dict['occluded_xe'] = []
        loss_dict['flow'] = []
        loss_dict['flow_warp_xe'] = []

        # loss_dict['flow_2'] = []

        #Preparation for flow warping:
        h = tf.range(self.config.grid_height_cells, dtype=tf.float32)
        w = tf.range(self.config.grid_width_cells, dtype=tf.float32)
        h_idx, w_idx = tf.meshgrid(h, w)
        # These indices map each (x, y) location to (x, y).
        # [height, width, 2] but storing x, y coordinates.
        identity_indices = tf.stack(
            (
                tf.transpose(w_idx),
                tf.transpose(h_idx),
            ),axis=-1)
        identity_indices = tf.stop_gradient(identity_indices)

        # Iterate over waypoints.
        # flow_origin_occupancy = curr_ogm[:,128:128+256,128:128+256,tf.newaxis]
        n_waypoints = self.config.num_waypoints
        has_true_observed_occupancy = {-1: True}
        has_true_occluded_occupancy = {-1: True}
        true_obs_cnt,true_occ_cnt,true_flow_cnt = [],[],[]
        f_c = []
        for k in range(n_waypoints):
            # Occupancy cross-entropy loss.
            pred_observed_occupancy_logit = (
                pred_waypoint_logits.vehicles.observed_occupancy[k])
            pred_occluded_occupancy_logit = (
                pred_waypoint_logits.vehicles.occluded_occupancy[k])
            pred_flow = pred_waypoint_logits.vehicles.flow[k]
          
            true_observed_occupancy = true_waypoints.vehicles.observed_occupancy[k]
            true_occluded_occupancy = true_waypoints.vehicles.occluded_occupancy[k]
        
            true_flow = true_waypoints.vehicles.flow[k]
  
            # Accumulate over waypoints.
            loss_dict['observed_xe'].append(
                self._sigmoid_xe_loss(
                    true_occupancy=true_observed_occupancy,
                    pred_occupancy=pred_observed_occupancy_logit,
                    loss_weight=self.ogm_weight)) 
            loss_dict['occluded_xe'].append(
                self._sigmoid_occ_loss(
                    true_occupancy=true_occluded_occupancy,
                    pred_occupancy=pred_occluded_occupancy_logit,
                    loss_weight=self.occ_weight))
            
            true_all_occupancy = tf.clip_by_value(true_observed_occupancy + true_occluded_occupancy, 0, 1)
            flow_origin_occupancy = true_waypoints.vehicles.flow_origin_occupancy[k]
            if self.use_gt:
                warped_indices = identity_indices + true_flow
                wp_org = sample(
                        image=flow_origin_occupancy,
                        warp=warped_indices,
                        pixel_type=0,
                    )
                self.auc.update_state(true_all_occupancy,wp_org*true_all_occupancy)
                res = self.auc.result()
                self.auc.reset_states()
                res = tf.cast(1 - res<1.0,tf.float32)
            else:
                res = 1.0
            f_c.append(res)
            loss_dict['flow'].append(res*self._flow_loss(true_flow,pred_flow))

            # flow warp_loss:
            if not self.no_use_warp:
                warped_indices = identity_indices + pred_flow
                wp_origin = sample(
                    image=flow_origin_occupancy,
                    warp=warped_indices,
                    pixel_type=0,
                )
                if self.use_pred:
                    loss_dict['flow_warp_xe'].append(res*self._sigmoid_xe_warp_loss_pred(true_all_occupancy,
                pred_observed_occupancy_logit, pred_occluded_occupancy_logit, wp_origin,
                loss_weight=self.flow_origin_weight))
                else:
                    loss_dict['flow_warp_xe'].append(res*self._sigmoid_xe_warp_loss(true_all_occupancy,
                    true_observed_occupancy, true_occluded_occupancy, wp_origin,
                    loss_weight=self.flow_origin_weight))
            
        # Mean over waypoints.
        n_dict = {}
        n_dict['observed_xe'] = tf.math.add_n(loss_dict['observed_xe']) / n_waypoints
        n_dict['occluded_xe'] = tf.math.add_n(loss_dict['occluded_xe']) / n_waypoints
        n_dict['flow'] = tf.math.add_n(loss_dict['flow']) / tf.math.add_n(f_c)

        if not self.no_use_warp:
            n_dict['flow_warp_xe'] = tf.math.add_n(loss_dict['flow_warp_xe']) / tf.math.add_n(f_c)
        else:
            n_dict['flow_warp_xe'] = 0.0
        return n_dict


    def _sigmoid_xe_loss(
        self,
        true_occupancy: tf.Tensor,
        pred_occupancy: tf.Tensor,
        loss_weight: float = 1000,
    ) -> tf.Tensor:
        """Computes sigmoid cross-entropy loss over all grid cells."""
        # Since the mean over per-pixel cross-entropy values can get very small,
        # we compute the sum and multiply it by the loss weight before computing
        # the mean.
        if self.use_focal_loss:
            xe_sum = tf.reduce_sum(
                self.focal_loss(
                    y_true=self._batch_flatten(true_occupancy),
                    y_pred=self._batch_flatten(pred_occupancy)
                )) + tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        else:
            xe_sum = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        # Return mean.
        return loss_weight * xe_sum / (tf.size(pred_occupancy, out_type=tf.float32)*self.replica)
    
    def _sigmoid_occ_loss(
        self,
        true_occupancy: tf.Tensor,
        pred_occupancy: tf.Tensor,
        loss_weight: float = 1000,
    ) -> tf.Tensor:
        """Computes sigmoid cross-entropy loss over all grid cells."""
        # Since the mean over per-pixel cross-entropy values can get very small,
        # we compute the sum and multiply it by the loss weight before computing
        # the mean.
        if self.use_focal_loss:
            xe_sum = tf.reduce_sum(
                self.occlude_focal_loss(
                    y_true=self._batch_flatten(true_occupancy),
                    y_pred=self._batch_flatten(pred_occupancy)
                )) +tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        else:
            xe_sum = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._batch_flatten(true_occupancy),
                logits=self._batch_flatten(pred_occupancy),
            ))
        # Return mean.
        return loss_weight * xe_sum / (tf.size(pred_occupancy, out_type=tf.float32)*self.replica)
    
    def _sigmoid_xe_warp_loss(
        self,
        true_occupancy: tf.Tensor,
        pred_occupancy_obs: tf.Tensor,
        pred_occupancy_occ: tf.Tensor,
        warped_origin: tf.Tensor,
        loss_weight: float = 1000,
    ) -> tf.Tensor:
        labels=self._batch_flatten(true_occupancy)
        sig_logits = self._batch_flatten(tf.sigmoid(pred_occupancy_obs)+ tf.sigmoid(pred_occupancy_occ))
        sig_logits = tf.clip_by_value(sig_logits,0,1)
        joint_flow_occ_logits = sig_logits * self._batch_flatten(warped_origin)
        # joint_flow_occ_logits = tf.clip_by_value(joint_flow_occ_logits,0,1)
        if self.use_focal_loss:
            xe_sum = tf.reduce_sum(self.flow_focal_loss(labels,joint_flow_occ_logits)) + tf.reduce_sum(self.bce(labels,joint_flow_occ_logits))
        else:
            xe_sum =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels,joint_flow_occ_logits))

        # Return mean.
        return loss_weight * xe_sum / (tf.size(true_occupancy, out_type=tf.float32)*self.replica)
    
    def _sigmoid_xe_warp_loss_pred(
        self,
        true_occupancy: tf.Tensor,
        pred_occupancy_obs: tf.Tensor,
        pred_occupancy_occ: tf.Tensor,
        warped_origin: tf.Tensor,
        loss_weight: float = 1000,
    ) -> tf.Tensor:
        labels=self._batch_flatten(true_occupancy)
        sig_logits = self._batch_flatten(tf.sigmoid(pred_occupancy_obs)+tf.sigmoid(pred_occupancy_occ))
        sig_logits = tf.clip_by_value(sig_logits,0,1)
        joint_flow_occ_logits =  self._batch_flatten(warped_origin)*sig_logits
        if self.use_focal_loss:
            xe_sum = tf.reduce_sum(self.flow_focal_loss(labels,joint_flow_occ_logits)) + tf.reduce_sum(self.bce(labels,joint_flow_occ_logits))
        else:
            xe_sum =tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels,joint_flow_occ_logits))
        xe_sum = tf.reduce_sum( self.bce(labels,joint_flow_occ_logits) )

        # Return mean.
        return loss_weight * xe_sum / (tf.size(true_occupancy, out_type=tf.float32)*self.replica)

    def _flow_loss(
        self,
        true_flow: tf.Tensor,
        pred_flow: tf.Tensor,
        loss_weight: float = 1,
    ) -> tf.Tensor:
        """Computes L1 flow loss."""
        diff = true_flow - pred_flow
        # Ignore predictions in areas where ground-truth flow is zero.
        # [batch_size, height, width, 1], [batch_size, height, width, 1]
        true_flow_dx, true_flow_dy = tf.split(true_flow, 2, axis=-1)
        # [batch_size, height, width, 1]
        flow_exists = tf.logical_or(
            tf.not_equal(true_flow_dx, 0.0),
            tf.not_equal(true_flow_dy, 0.0),
        )
        flow_exists = tf.cast(flow_exists, tf.float32)
        diff = diff * flow_exists
        diff_norm = tf.linalg.norm(diff, ord=1, axis=-1)  # L1 norm.
        mean_diff = tf.math.divide_no_nan(
            tf.reduce_sum(diff_norm),
            (tf.reduce_sum(flow_exists)*self.replica / 2))  # / 2 since (dx, dy) is counted twice.
        return loss_weight * mean_diff

    def _batch_flatten(self,input_tensor: tf.Tensor) -> tf.Tensor:
        """Flatten tensor to a shape [batch_size, -1]."""
        image_shape = tf.shape(input_tensor)
        return tf.reshape(input_tensor, tf.concat([image_shape[0:1], [-1]], 0))