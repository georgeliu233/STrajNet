import tensorflow as tf
import numpy as np

class OGMFlowMetrics(object):
    def __init__(self,preflix='train',no_warp=False):
        # super().__init__()
        self.observed_auc = tf.keras.metrics.Mean(name='observed_auc')
        self.occluded_auc = tf.keras.metrics.Mean(name='occluded_auc')

        self.observed_iou = tf.keras.metrics.Mean(name='observed_iou')
        self.occluded_iou = tf.keras.metrics.Mean(name='occluded_iou')
        
        self.flow_epe = tf.keras.metrics.Mean(name='flow_epe')
        self.no_warp = no_warp

        if not no_warp:
            self.flow_ogm_auc = tf.keras.metrics.Mean(name='flow_ogm_auc')
            self.flow_ogm_iou = tf.keras.metrics.Mean(name='flow_ogm_iou')

        self.preflix = preflix
    
    def reset_states(self):
        self.observed_auc.reset_states()
        self.occluded_auc.reset_states()

        self.observed_iou.reset_states()
        self.occluded_iou.reset_states()

        self.flow_epe.reset_states()
        if not self.no_warp:
            self.flow_ogm_auc.reset_states()
            self.flow_ogm_iou.reset_states()
    
    def update_state(self,metrics):
        self.observed_auc.update_state(metrics.vehicles_observed_auc)
        self.occluded_auc.update_state(metrics.vehicles_occluded_auc)

        self.observed_iou.update_state(metrics.vehicles_observed_iou)
        self.occluded_iou.update_state(metrics.vehicles_occluded_iou)

        self.flow_epe.update_state(metrics.vehicles_flow_epe)
        if not self.no_warp:
            self.flow_ogm_auc.update_state(metrics.vehicles_flow_warped_occupancy_auc)
            self.flow_ogm_iou.update_state(metrics.vehicles_flow_warped_occupancy_iou)
    
    def get_result(self):
        res_dict={}
        res_dict[f'{self.preflix}_observed_auc'] = self.observed_auc.result().numpy()
        res_dict[f'{self.preflix}_occluded_auc'] = self.occluded_auc.result().numpy()

        res_dict[f'{self.preflix}_observed_iou'] = self.observed_iou.result().numpy()
        res_dict[f'{self.preflix}_occluded_iou'] = self.occluded_iou.result().numpy()

        res_dict[f'{self.preflix}_flow_epe'] = self.flow_epe.result().numpy()
        if not self.no_warp:
            res_dict[f'{self.preflix}_flow_ogm_auc'] = self.flow_ogm_auc.result().numpy()
            res_dict[f'{self.preflix}_flow_ogm_iou'] = self.flow_ogm_iou.result().numpy()

        return res_dict
    
def print_metrics(res_dict,preflix='train',no_warp=False):
    # print(f'\n |obs-AUC: {res_dict['observed_auc']}')
    if no_warp:
       print(f"""\n |obs-AUC: {res_dict[f'{preflix}_observed_auc']}|occ-AUC: {res_dict[f'{preflix}_occluded_auc']}
            |obs-IOU: {res_dict[f'{preflix}_observed_iou']}|occ-IOU: {res_dict[f'{preflix}_occluded_iou']}
            | Flow-EPE: {res_dict[f'{preflix}_flow_epe']}|""")
    else: 
        print(f"""\n |obs-AUC: {res_dict[f'{preflix}_observed_auc']}|occ-AUC: {res_dict[f'{preflix}_occluded_auc']}
                |obs-IOU: {res_dict[f'{preflix}_observed_iou']}|occ-IOU: {res_dict[f'{preflix}_occluded_iou']}
                | Flow-EPE: {res_dict[f'{preflix}_flow_epe']}
                |FlowOGM_AUC: {res_dict[f'{preflix}_flow_ogm_auc']} |FlowOGM_IOU: {res_dict[f'{preflix}_flow_ogm_iou']} |""")
        