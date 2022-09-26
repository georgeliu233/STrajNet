import os 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import matplotlib.pyplot as plt  
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import occupancy_flow_submission_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils import occupancy_flow_grids
from waymo_open_dataset.utils import occupancy_flow_renderer
from waymo_open_dataset.utils import occupancy_flow_vis

from tqdm import tqdm
from PIL import Image as Image
from time import time as time

from data_utils import road_label,road_line_map,light_label,light_state_map
from grid_utils import create_all_grids,rotate_all_from_inputs,add_sdc_fields
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import argparse
import os
mpl.use('Agg')

def extract_lines(xy, id, typ):
    line = [] # a list of points  
    lines = [] # a list of lines
    length = xy.shape[0]
    for i, p in enumerate(xy):
        line.append(p)
        next_id = id[i+1] if i < length-1 else id[i]
        current_id = id[i]
        if next_id != current_id or i == length-1:
            if typ in [18, 19]:
                line.append(line[0])
            lines.append(line)
            line = []
    return lines

class Processor(object):

    def __init__(self, area_size, max_actors,max_occu, radius,rasterisation_size=256,save_dir='.',ids_dir=''):
        # parameters
        self.img_size = rasterisation_size # size = pixels * pixels
        self.area_size = area_size # size = [vehicle, pedestrian, cyclist] meters * meters
        self.max_actors = max_actors
        self.max_occu = max_occu
        self.radius = radius
        self.save_dir = save_dir
        self.ids_dir = ids_dir

        self.get_config()

    def load_data(self, filename):
        self.filename = filename
        dataset = tf.data.TFRecordDataset(filename, compression_type='')
        self.dataset_length = len(list(dataset.as_numpy_iterator()))
        dataset = dataset.map(occupancy_flow_data.parse_tf_example)
        self.datalist = dataset.batch(1)
        # self.datalist = list(dataset.as_numpy_iterator())
    
    def get_config(self):
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

        self.config = config

        ogm_config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        oconfig_text = """
        num_past_steps: 10
        num_future_steps: 80
        num_waypoints: 8
        cumulative_waypoints: false
        normalize_sdc_yaw: true
        grid_height_cells: 512
        grid_width_cells: 512
        sdc_y_in_grid: 320
        sdc_x_in_grid: 256
        pixels_per_meter: 3.2
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """
        text_format.Parse(oconfig_text, ogm_config)
        self.ogm_config = ogm_config

    def read_data(self, parsed):
        
        map_traj,real_map_traj,map_valid,actor_traj,traj_mask,occu_mask,actor_valid = rotate_all_from_inputs(parsed, self.config)
        
        # actor traj
        self.actor_traj = actor_traj[0].numpy()
        self.traj_mask = traj_mask[0,:].numpy()
        self.occu_mask = occu_mask[0,:].numpy()
        #[batch,actor_num,11,1]
        self.actor_valid = actor_valid[0,:,:,0].numpy()
        self.actor_type = parsed['state/type'][0].numpy()

        # road map
        roadgraph_xyz = map_traj[0].numpy()
        real_map_traj = real_map_traj[0].numpy()
        roadgraph_dir = parsed['roadgraph_samples/dir'][0].numpy()
        roadgraph_type = parsed['roadgraph_samples/type'][0].numpy()
        roadgraph_id = parsed['roadgraph_samples/id'][0].numpy()
        roadgraph_valid = map_valid[0,:,0].numpy()
        # print(roadgraph_valid)
        v_mask = np.where(roadgraph_valid)
        self.roadgraph_xyz = roadgraph_xyz[v_mask]
        self.roadgraph_dir = roadgraph_dir[v_mask]
        self.roadgraph_type = roadgraph_type[v_mask]
        self.roadgraph_real_traj = real_map_traj[v_mask]
        self.roadgraph_id = roadgraph_id[v_mask]

        # print(len(self.roadgraph_id))
        self.roadgraph_uid = np.unique(self.roadgraph_id)
        # print(len(self.roadgraph_uid))
        self.roadgraph_types = np.unique(self.roadgraph_type)
        # print(self.roadgraph_types)

        # traffic lights
        traffic_light_state = parsed['traffic_light_state/current/state'][0].numpy()
        traffic_light_x = parsed['traffic_light_state/current/x'][0].numpy()
        traffic_light_y = parsed['traffic_light_state/current/y'][0].numpy()
        traffic_light_valid = parsed['traffic_light_state/current/valid'].numpy()
        self.traffic_light_x = traffic_light_x[0, np.where(traffic_light_valid)[1]]
        self.traffic_light_y = traffic_light_y[0, np.where(traffic_light_valid)[1]]
        self.traffic_light_state = traffic_light_state[0, np.where(traffic_light_valid)[1]]
    
    def actor_traj_process(self):
        emb = np.eye(3)
        traj_m = np.where(self.traj_mask)
        valid_actor = self.actor_traj[traj_m]
        valid_mask = self.actor_valid[traj_m]
        valid_type = self.actor_type[traj_m]
        dist=[]
        curr_buf=[]

        for i in range(valid_actor.shape[0]):
            w = np.where(valid_mask[i])[0]
            if w.shape[0]==0:
                continue
            n = w[-1]
            last_pos = valid_actor[i,n,:]
            dist.append(last_pos[:2])

        dist = np.argsort(np.linalg.norm(dist,axis=-1))[:self.max_actors]
        # current_state = [curr_buf[d] for d in dist]
        actor_type = []
        for d in dist:
            ind = int(valid_type[d])
            if ind in set([1,2,3]):
                actor_type.append(emb[ind-1])
            else:
                actor_type.append([0,0,0])

        output_actors = np.zeros((self.max_actors,11,5+3))
        for i,d in enumerate(dist):
            output_actors[i] =  np.concatenate((valid_actor[d],np.tile(actor_type[i],(11,1))),axis=-1)
        
        #process the possible occulde traj:
        occ_m = np.where(self.occu_mask)
        occu_actor = self.actor_traj[occ_m]
        occu_valid = self.actor_valid[occ_m]
        occu_type = self.actor_type[occ_m]

        dist=[]
        curr_buf=[]
        occu_traj = []
        o_type = []
        for i in range(occu_actor.shape[0]):
            w = np.where(occu_valid[i])[0]
            if w.shape[0]==0:
                continue
            b,e = w[0] , w[-1]
            begin_pos,last_pos = occu_actor[i,b,:2],occu_actor[i,e,:2]
            begin_dist,last_dist = np.linalg.norm(begin_pos) , np.linalg.norm(last_pos)
            if begin_dist<=last_dist:
                continue
            dist.append(last_dist)
            # curr_buf.append(occu_actor[i,e,:])
            occu_traj.append(occu_actor[i])
            o_type.append(occu_type[i])
        
        dist = np.argsort(dist)[:self.max_occu]
        out_occu_type = []
        for d in dist:
            ind = int(o_type[d])
            if ind in set([1,2,3]):
                out_occu_type.append(emb[ind-1])
            else:
                out_occu_type.append([0,0,0])

        output_occu_actors = np.zeros((self.max_occu,11,5+3))
        for i,d in enumerate(dist):
            output_occu_actors[i] = np.concatenate((occu_traj[d] ,np.tile(out_occu_type[i],(11,1))),axis=-1)
        
        return output_actors , output_occu_actors #, np.array(current_state)
    
    def seg_traj(self,traj,emb,seg_length=10):
        # np = self.np
        traj = np.array(traj)
        traj_length = traj.shape[0]
        pad_length = seg_length - traj_length % seg_length
        embs = np.tile(emb,(traj_length,1))
        traj = np.concatenate((traj,embs),axis=-1)
        traj = np.concatenate((traj,np.zeros((pad_length,4+3))),axis=0).reshape((-1,seg_length,4+3))
        return traj

    def map_traj_process(self):
        ##segment all valid center traj in the ogm map##   
        # np = self.np 
        num_segs = 256   
        type_set = set(self.roadgraph_types)
        # self.centerlines = []
        seg_length = 10
        line_cnt = 0
        res_traj = []
        # emb = np.eye(3)
        if 1 in type_set or 2 in type_set or 3 in type_set or 18 in type_set:
            for uid in self.roadgraph_uid:
                mask = np.where(self.roadgraph_id==uid)[0]
                way_type = int(self.roadgraph_type[mask][0])
                if way_type not in set([1,2,3,18]):
                    continue
                if way_type in set([1,2]):
                    emb_type = [1,0,0]
                elif way_type==3:
                    emb_type = [0,1,0]
                else:
                    emb_type = [0,0,1]
                traj = self.roadgraph_real_traj[mask]
                seg_traj = self.seg_traj(traj,seg_length=seg_length,emb=emb_type)
                seg_traj_len = seg_traj.shape[0]

                line_cnt += seg_traj_len
                res_traj.append(seg_traj)
                if line_cnt>num_segs:
                    break
            res_traj = np.concatenate(res_traj,axis=0)[:num_segs]
            if res_traj.shape[0]<num_segs:
                res_traj = np.concatenate((res_traj, np.zeros((num_segs-res_traj.shape[0],10,4+3))),axis=0)
            return res_traj
        else:
            return np.zeros((num_segs,10,4+3))

    def ogm_process(self,inputs):
        timestep_grids = occupancy_flow_grids.create_ground_truth_timestep_grids(inputs, self.ogm_config)
        gt_v_ogm = tf.concat([timestep_grids.vehicles.past_occupancy,timestep_grids.vehicles.current_occupancy],axis=-1)
        gt_o_ogm = tf.concat([tf.clip_by_value(
                timestep_grids.pedestrians.past_occupancy +
                timestep_grids.cyclists.past_occupancy, 0, 1),
            tf.clip_by_value(
                timestep_grids.pedestrians.current_occupancy +
                timestep_grids.cyclists.current_occupancy, 0, 1)
                ],axis=-1)
        ogm = tf.stack([gt_v_ogm,gt_o_ogm],axis=-1)
        return ogm[0].numpy().astype(np.bool_),timestep_grids
    
    def image_process(self,show_image=False,num=0):

        fig, ax = plt.subplots()
        dpi = 1
        size_inches = self.img_size / dpi

        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)
        fig.set_facecolor('k')
        ax.set_facecolor('k') #
        ax.grid(False)
        ax.margins(0)
        ax.axis('off')
        
        # plot static roadmap
        big=80
        for t in self.roadgraph_types:
            road_points = self.roadgraph_xyz[np.where(self.roadgraph_type==t)[0]]
            road_points = road_points[:, :2]
            point_id = self.roadgraph_id[np.where(self.roadgraph_type==t)[0]]
            if t in set([1, 2, 3]):
                lines = extract_lines(road_points, point_id, t)
                for line in lines:
                    ax.plot([point[0] for point in line], [point[1] for point in line], 
                             color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big, alpha=1, zorder=1)
            elif t == 17: # plot stop signs
                ax.plot(road_points.T[0, :], road_points.T[1, :], road_line_map[t][1], color=road_line_map[t][0], markersize=road_line_map[t][2]*big)
            elif t in set([18, 19]): # plot crosswalk and speed bump
                rects = extract_lines(road_points, point_id, t)
                for rect in rects:
                     area = plt.fill([point[0] for point in rect], [point[1] for point in rect], color=road_line_map[t][0], alpha=0.7, zorder=2)
            else: # plot other elements
                lines = extract_lines(road_points, point_id, t)
                for line in lines:
                    ax.plot([point[0] for point in line], [point[1] for point in line], 
                            color=road_line_map[t][0], linestyle=road_line_map[t][1], linewidth=road_line_map[t][2]*big)

        # plot traffic lights
        for lx, ly, ls in zip(self.traffic_light_x, self.traffic_light_y, self.traffic_light_state):
            light_circle = plt.Circle((lx, ly), 1.5*big, color=light_state_map[ls], zorder=2)
            ax.add_artist(light_circle)

        pixels_per_meter = 1#self.config.pixels_per_meter
        range_x = self.config.sdc_x_in_grid
        range_y = self.config.sdc_y_in_grid

        ax.axis([0,256,0,256])
        ax.set_aspect('equal')

        # convert plot to numpy array
        fig.canvas.draw()
        array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        array = array.reshape(fig.canvas.get_width_height() + (3,))[::-1,:,:]

        plt.close('all')

        # visualize the image                   
        if show_image:
            img = self.Image.fromarray(array, 'RGB')
            time.sleep(30)
            plt.imshow(array)
        return array

    def gt_process(self,timestep_grids,flow_only=False):
        true_waypoints = occupancy_flow_grids.create_ground_truth_waypoint_grids(timestep_grids=timestep_grids, config=self.config)
        if flow_only:
            gt_origin_flow = tf.concat(true_waypoints.vehicles.flow_origin_occupancy,axis=0).numpy()
            return gt_origin_flow
        gt_obs_ogm = tf.concat(true_waypoints.vehicles.observed_occupancy,axis=0).numpy().astype(np.bool_)
        gt_occ_ogm = tf.concat(true_waypoints.vehicles.occluded_occupancy,axis=0).numpy().astype(np.bool_)
        gt_flow = tf.concat(true_waypoints.vehicles.flow,axis=0).numpy()
        gt_origin_flow = tf.concat(true_waypoints.vehicles.flow_origin_occupancy,axis=0).numpy()
        return gt_obs_ogm,gt_occ_ogm,gt_flow,gt_origin_flow
    
    def get_ids(self,val=True):
        if val:
            path = f'{self.ids_dir}/validation_scenario_ids.txt'
        else:
            path = f'{self.ids_dir}/testing_scenario_ids.txt'
        with tf.io.gfile.GFile(path) as f:
            test_scenario_ids = f.readlines()
            test_scenario_ids = [id.rstrip() for id in test_scenario_ids]
            self.test_scenario_ids = set(test_scenario_ids)
    
    def flow_process(self,timestep_grids):
        vec_hist_flow = timestep_grids.vehicles.all_flow[:,:,:,0,:]
        ped_byc_hist_flow = timestep_grids.pedestrians.all_flow[:,:,:,0,:] + timestep_grids.cyclists.all_flow[:,:,:,0,:]
        return vec_hist_flow[0].numpy(),ped_byc_hist_flow[0].numpy()
    
    def build_saving_tfrecords(self,pred,val,num):
        if pred:
            self.get_ids(val=False)
            if not os.path.exists(f'{self.save_dir}/test/'):
                os.makedirs(f'{self.save_dir}/test/')
            writer = tf.io.TFRecordWriter(f'{self.save_dir}/test/'+f'{num}'+'new.tfrecords')
        if val:
            self.get_ids(val=True)
            if not os.path.exists(f'{self.save_dir}/val/'):
                os.makedirs(f'{self.save_dir}/val/')
            writer = tf.io.TFRecordWriter(f'{self.save_dir}/val/'+f'{num}'+'new.tfrecords')
        
        if not (pred or val):
            if not os.path.exists(f'{self.save_dir}/train/'):
                os.makedirs(f'{self.save_dir}/train/')
            writer = tf.io.TFRecordWriter(f'{self.save_dir}/train/'+f'{num}'+'new.tfrecords')
        return writer
        
    def workflow(self,pred=False,val=False):
        i = 0
        self.pbar = tqdm(total=self.dataset_length)
        num = self.filename.split('-')[1]
        writer = self.build_saving_tfrecords(pred, val,num)
        
        for dataframe in self.datalist:
            if pred or val:
                sc_id = dataframe['scenario/id'].numpy()[0]
                if isinstance(sc_id, bytes):
                    sc_id=str(sc_id, encoding = "utf-8") 
                if sc_id not in self.test_scenario_ids:
                    self.pbar.update(1)
                    continue

            dataframe = add_sdc_fields(dataframe)
            self.read_data(dataframe)

            ogm,timestep_grids = self.ogm_process(dataframe)
            output_actors,occu_actors = self.actor_traj_process()
            map_trajs = self.map_traj_process()

            image = self.image_process(show_image=False,num=i)
            image = image.tobytes()
            ogm = ogm.tobytes()

            map_trajs = map_trajs.tobytes()
            output_actors = output_actors.tobytes()
            occu_actors = occu_actors.tobytes()

            vec_flow,byc_flow = self.flow_process(timestep_grids)
            vec_flow = vec_flow.tobytes()
            byc_flow = byc_flow.tobytes()
            
            feature = {
                'centerlines': tf.train.Feature(bytes_list=tf.train.BytesList(value=[map_trajs])),
                'actors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[output_actors])),
                'occl_actors': tf.train.Feature(bytes_list=tf.train.BytesList(value=[occu_actors])),
                'ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[ogm])),
                'map_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'gt_obs_ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                'gt_occ_ogm': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                'gt_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                'origin_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[])),
                'vec_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vec_flow])),
                'byc_flow': tf.train.Feature(bytes_list=tf.train.BytesList(value=[byc_flow]))
            }
            if pred or val:
                feature['scenario/id'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[sc_id.encode('utf-8')]))
            if not pred:
                gt_obs_ogm,gt_occ_ogm,gt_flow,gt_origin_flow=self.gt_process(timestep_grids)
                feature['gt_obs_ogm'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_obs_ogm.tobytes()]))
                feature['gt_occ_ogm'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_occ_ogm.tobytes()]))
                feature['gt_flow'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_flow.tobytes()]))
                feature['origin_flow'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[gt_origin_flow.tobytes()]))

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            self.pbar.update(1)
            i+=1
            # if i>=64:
            #     break

        writer.close()
        self.pbar.close()
        print('collect:',i)


def process_training_data(filename):
    print('Working on',filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow()
    print(filename, 'done!')

def process_val_data(filename):
    print('Working on', filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow(val=True)
    print(filename, 'done!')

def process_test_data(filename):
    print('Working on', filename)
    processor = Processor(rasterisation_size=256, area_size=[70, 35, 50], max_occu=16,max_actors=48, radius=30,
    save_dir=args.save_dir, ids_dir=args.ids_dir)
    processor.load_data(filename)
    processor.workflow(pred=True)
    print(filename, 'done!')

if __name__=="__main__":
    from multiprocessing import Pool
    from glob import glob

    parser = argparse.ArgumentParser(description='Data-preprocessing')
    parser.add_argument('--ids_dir', type=str, help='ids.txt downloads from Waymos', default="./Waymo_Dataset/occupancy_flow_challenge/")
    parser.add_argument('--save_dir', type=str, help='saving directory',default="./Waymo_Dataset/preprocessed_data/")
    parser.add_argument('--file_dir', type=str, help='Dataset directory',default="./Waymo_Dataset/tf_example")
    parser.add_argument('--pool', type=int, help='num of pooling multi-processes in preprocessing',default=2)
    args = parser.parse_args()

    NUM_POOLS = args.pool

    # train_files = glob(f'{args.file_dir}/training/*')
    # print(f'Processing training data...{len(train_files)} found!')
    # print('Starting processing pooling...')
    # with Pool(NUM_POOLS) as p:
    #     p.map(process_training_data, train_files[:1])
    
    val_files = glob(f'{args.file_dir}/validation/*')
    print(f'Processing validation data...{len(val_files)} found!')
    print('Starting processing pooling...')
    with Pool(NUM_POOLS) as p:
        p.map(process_val_data, val_files[:1])
    
    # test_files = glob(f'{args.file_dir}/testing/*')
    # print(f'Processing validation data...{len(test_files)} found!')
    # print('Starting processing pooling...')
    # with Pool(NUM_POOLS) as p:
    #     p.map(process_test_data, test_files[:1])