from waymo_open_dataset.utils.occupancy_flow_renderer import _sample_and_filter_agent_points,rotate_points_around_origin,_stack_field
from data_utils import *
import tensorflow as tf
import numpy as np

import dataclasses
import math
from typing import List, Mapping, Sequence, Tuple,Dict

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.utils.occupancy_flow_grids import *
_ObjectType = scenario_pb2.Track.ObjectType

import tensorflow as tf

def _transform_to_image_coordinates(
    points_x: tf.Tensor,
    points_y: tf.Tensor,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    larger_box:bool=False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Returns transformed points and a mask indicating whether point is in image.

  Args:
    points_x: Tensor of any shape containing x values in world coordinates
      centered on the autonomous vehicle (see translate_sdc_to_origin).
    points_y: Tensor with same shape as points_x containing y values in world
      coordinates centered on the autonomous vehicle.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Tuple containing the following tensors:
      - Transformed points_x.
      - Transformed points_y.
      - tf.bool tensor with same shape as points_x indicating which points are
        inside the FOV of the image after transformation.
  """
  pixels_per_meter = config.pixels_per_meter
  points_x = tf.round(points_x * pixels_per_meter) + config.sdc_x_in_grid
  points_y = tf.round(-points_y * pixels_per_meter) + config.sdc_y_in_grid

  # Filter out points that are located outside the FOV of topdown map.
  if not larger_box:
    point_is_in_fov = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(points_x, 0), tf.greater_equal(points_y, 0)),
        tf.logical_and(
            tf.less(points_x, config.grid_width_cells),
            tf.less(points_y, config.grid_height_cells)))
  else:
    point_is_in_fov = tf.logical_and(
        tf.logical_and(
            tf.greater_equal(points_x, 0-64), tf.greater_equal(points_y, 0-64)),
        tf.logical_and(
            tf.less(points_x, config.grid_width_cells+64),
            tf.less(points_y, config.grid_height_cells+64)))

  return points_x, points_y, point_is_in_fov


def add_sdc_fields(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
  """Extracts current x, y, z of the autonomous vehicle as specific fields."""
  # [batch_size, 2]
  sdc_indices = tf.where(tf.equal(inputs['state/is_sdc'], 1))
  # [batch_size, 1]
  inputs['sdc/current/x'] = tf.gather_nd(inputs['state/current/x'], sdc_indices)
  inputs['sdc/current/y'] = tf.gather_nd(inputs['state/current/y'], sdc_indices)
  inputs['sdc/current/z'] = tf.gather_nd(inputs['state/current/z'], sdc_indices)

  inputs['sdc/current/velocity_x'] = tf.gather_nd(inputs['state/current/velocity_x'], sdc_indices)
  inputs['sdc/current/velocity_y'] = tf.gather_nd(inputs['state/current/velocity_y'], sdc_indices)

  inputs['sdc/current/bbox_yaw'] = tf.gather_nd(
      inputs['state/current/bbox_yaw'], sdc_indices)
  return inputs

def create_all_grids(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> TimestepGrids:
  """Renders topdown views of agents over past/current/future time frames.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    TimestepGrids object holding topdown renders of agents.
  """
  timestep_grids = TimestepGrids()

  # Occupancy grids.
  render_func = functools.partial(
      occupancy_flow_renderer.render_occupancy_from_inputs,
      inputs=inputs,
      config=config)

  current_occupancy = render_func(
      times=['current'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.current_occupancy = current_occupancy.vehicles
  timestep_grids.pedestrians.current_occupancy = current_occupancy.pedestrians
  timestep_grids.cyclists.current_occupancy = current_occupancy.cyclists

  past_occupancy = render_func(
      times=['past'],
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.past_occupancy = past_occupancy.vehicles
  timestep_grids.pedestrians.past_occupancy = past_occupancy.pedestrians
  timestep_grids.cyclists.past_occupancy = past_occupancy.cyclists

  # Flow.
  # NOTE: Since the future flow depends on the current and past timesteps, we
  # need to compute it from [past + current + future] sparse points.
  all_flow = render_flow_from_inputs_temp(
      inputs=inputs,
      times=['past', 'current'],
      config=config,
      include_observed=True,
      include_occluded=True,
  )
  timestep_grids.vehicles.all_flow = all_flow.vehicles
  timestep_grids.pedestrians.all_flow = all_flow.pedestrians
  timestep_grids.cyclists.all_flow = all_flow.cyclists

  return timestep_grids

def render_occupancy_from_inputs_temp(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> occupancy_flow_data.AgentGrids:
  """Creates topdown renders of agents grouped by agent class.

  Renders agent boxes by densely sampling points from their boxes.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size,time ,height, width, steps] float32 in [0, 1].
      pedestrians: [batch_size, time,height, width, steps] float32 in [0, 1].
      cyclists: [batch_size,time, height, width, steps] float32 in [0, 1].
      where steps is the number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells, num_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  occupancies = {}
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)

    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_steps, points_per_agent]),
    ])

    topdowns= []

    for t in range(num_steps):
        # Scatter points across topdown maps for each timestep.  The tensor
        # `point_indices` holds the indices where `should_render_point` is True.
        # It is a 2-D tensor with shape [n, 4], where n is the number of valid
        # agent points inside FOV.  Each row in this tensor contains indices over
        # the following 4 dimensions: (batch, agent, timestep, point).

        # [num_points_to_render, 4]
        point_indices = tf.cast(tf.where(should_render_point[:,:,t,:]), tf.int32)
        # print(point_indices)
        # [num_points_to_render, 1]
        x_img_coord = tf.gather_nd(agent_x[:,:,t,:], point_indices)[..., tf.newaxis]
        y_img_coord = tf.gather_nd(agent_y[:,:,t,:], point_indices)[..., tf.newaxis]

        num_points_to_render = point_indices.shape.as_list()[0]
        assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                    (y_img_coord, [num_points_to_render, 1])])

        # [num_points_to_render, 4]
        xy_img_coord = tf.concat(
            [
                point_indices[:, :1],
                tf.cast(y_img_coord, tf.int32),
                tf.cast(x_img_coord, tf.int32),
                # point_indices[:, 2:3],
            ],
            axis=1,
        )
        # [num_points_to_render]
        gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

        # [batch_size, grid_height_cells, grid_width_cells, num_steps]
        topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
        assert_shapes([(topdown, topdown_shape)])

        # scatter_nd() accumulates values if there are repeated indices.  Since
        # we sample densely, this happens all the time.  Clip the final values.
        topdown = tf.clip_by_value(topdown, 0.0, 1.0)
        topdowns.append(topdown)

    occupancies[object_type] = tf.stack(topdowns,axis=1)

  return occupancy_flow_data.AgentGrids(
      vehicles=occupancies[_ObjectType.TYPE_VEHICLE],
      pedestrians=occupancies[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=occupancies[_ObjectType.TYPE_CYCLIST],
  )


def render_flow_from_inputs_temp(
    inputs: Mapping[str, tf.Tensor],
    times: Sequence[str],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    include_observed: bool,
    include_occluded: bool,
) -> occupancy_flow_data.AgentGrids:
  """Compute top-down flow between timesteps `waypoint_size` apart.

  Returns (dx, dy) for each timestep.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    times: List containing any subset of ['past', 'current', 'future'].
    config: OccupancyFlowTaskConfig proto message.
    include_observed: Whether to include currently-observed agents.
    include_occluded: Whether to include currently-occluded agents.

  Returns:
    An AgentGrids object containing:
      vehicles: [batch_size, height, width, num_flow_steps, 2] float32
      pedestrians: [batch_size, height, width, num_flow_steps, 2] float32
      cyclists: [batch_size, height, width, num_flow_steps, 2] float32
      where num_flow_steps = num_steps - waypoint_size, and num_steps is the
      number of timesteps covered in `times`.
  """
  sampled_points = _sample_and_filter_agent_points(
      inputs=inputs,
      times=times,
      config=config,
      include_observed=include_observed,
      include_occluded=include_occluded,
  )

  agent_x = sampled_points.vx
  agent_y = sampled_points.vy
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  # The timestep distance between flow steps.
#   waypoint_size = config.num_future_steps // config.num_waypoints
  num_flow_steps = num_steps# - waypoint_size
  topdown_shape = [
      batch_size, config.grid_height_cells, config.grid_width_cells,
      num_flow_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )
  assert_shapes([(point_is_in_fov,
                  [batch_size, num_agents, num_steps, points_per_agent])])

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)

  # Backward Flow.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  dx =  agent_x[:, :, :, :]
  dy =  agent_y[:, :, :, :]
  assert_shapes([
      (dx, [batch_size, num_agents, num_flow_steps, points_per_agent]),
      (dy, [batch_size, num_agents, num_flow_steps, points_per_agent]),
  ])

  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  flows = {}
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)
    assert_shapes([
        (should_render_point,
         [batch_size, num_agents, num_flow_steps, points_per_agent]),
    ])

    # [batch_size, height, width, num_flow_steps, 2]
    flow = _render_flow_points_for_one_agent_type(
        agent_x=agent_x,
        agent_y=agent_y,
        dx=dx,
        dy=dy,
        should_render_point=should_render_point,
        topdown_shape=topdown_shape,
    )
    flows[object_type] = flow

  return occupancy_flow_data.AgentGrids(
      vehicles=flows[_ObjectType.TYPE_VEHICLE],
      pedestrians=flows[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=flows[_ObjectType.TYPE_CYCLIST],
  )

def _render_flow_points_for_one_agent_type(
    agent_x: tf.Tensor,
    agent_y: tf.Tensor,
    dx: tf.Tensor,
    dy: tf.Tensor,
    should_render_point: tf.Tensor,
    topdown_shape: List[int],
) -> tf.Tensor:
  """Renders topdown (dx, dy) flow for given agent points.

  Args:
    agent_x: [batch_size, num_agents, num_steps, points_per_agent].
    agent_y: [batch_size, num_agents, num_steps, points_per_agent].
    dx: [batch_size, num_agents, num_steps, points_per_agent].
    dy: [batch_size, num_agents, num_steps, points_per_agent].
    should_render_point: [batch_size, num_agents, num_steps, points_per_agent].
    topdown_shape: Shape of the output flow field.

  Returns:
    Rendered flow as [batch_size, height, width, num_flow_steps, 2] float32
      tensor.
  """
  assert_shapes = tf.debugging.assert_shapes

  # Scatter points across topdown maps for each timestep.  The tensor
  # `point_indices` holds the indices where `should_render_point` is True.
  # It is a 2-D tensor with shape [n, 4], where n is the number of valid
  # agent points inside FOV.  Each row in this tensor contains indices over
  # the following 4 dimensions: (batch, agent, timestep, point).

  # [num_points_to_render, 4]
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 4]
  xy_img_coord = tf.concat(
      [
          point_indices[:, :1],
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
          point_indices[:, 2:3],
      ],
      axis=1,
  )
  # [num_points_to_render]
  gt_values_dx = tf.gather_nd(dx, point_indices)
  gt_values_dy = tf.gather_nd(dy, point_indices)

  # tf.scatter_nd() accumulates values when there are repeated indices.
  # Keep track of number of indices writing to the same pixel so we can
  # account for accumulated values.
  # [num_points_to_render]
  gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
  flow_x = tf.scatter_nd(xy_img_coord, gt_values_dx, topdown_shape)
  flow_y = tf.scatter_nd(xy_img_coord, gt_values_dy, topdown_shape)
  num_values_per_pixel = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
  assert_shapes([
      (flow_x, topdown_shape),
      (flow_y, topdown_shape),
      (num_values_per_pixel, topdown_shape),
  ])

  # Undo the accumulation effect of tf.scatter_nd() for repeated indices.
  flow_x = tf.math.divide_no_nan(flow_x, num_values_per_pixel)
  flow_y = tf.math.divide_no_nan(flow_y, num_values_per_pixel)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps, 2]
  flow = tf.stack([flow_x, flow_y], axis=-1)
  assert_shapes([(flow, topdown_shape + [2])])
  return flow


def rotate_all_from_inputs(
    inputs: Mapping[str, tf.Tensor],
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> tf.Tensor:
  """Creates a topdown render of roadgraph points.

  This function is mostly useful for visualization.

  Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    Rendered roadgraph as [batch_size, height, width, 1] float32 tensor
      containing zeros and ones.
  """
  grid_height_cells = config.grid_height_cells
  grid_width_cells = config.grid_width_cells


  # FOR ROADGRAPH #
  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  batch_size, num_rg_points, _ = (
      inputs['roadgraph_samples/xyz'].shape.as_list())
  # print(inputs['roadgraph_samples/xyz'].shape.as_list())
  topdown_shape = [batch_size, grid_height_cells, grid_width_cells, 1]

  # Translate the roadgraph points so that the autonomous vehicle is at the
  # origin.
  sdc_xyz = tf.concat(
      [
          inputs['sdc/current/x'],
          inputs['sdc/current/y'],
          inputs['sdc/current/z'],
      ],
      axis=1,
  )
  # [batch_size, 1, 3]
  sdc_xyz = sdc_xyz[:, tf.newaxis, :]
  # [batch_size, num_rg_points, 3]
  rg_points = inputs['roadgraph_samples/xyz'] - sdc_xyz

  # [batch_size, num_rg_points, 1]
  rg_valid = inputs['roadgraph_samples/valid']
  assert_shapes([(rg_points, [batch_size, num_rg_points, 3]),
                 (rg_valid, [batch_size, num_rg_points, 1])])
  # [batch_size, num_rg_points]
  rg_x, rg_y, _ = tf.unstack(rg_points, axis=-1)
  assert_shapes([(rg_x, [batch_size, num_rg_points]),
                 (rg_y, [batch_size, num_rg_points])])

  rg_dx, rg_dy, _ = tf.unstack(inputs['roadgraph_samples/dir'], axis=-1)
  if config.normalize_sdc_yaw:
    angle = math.pi / 2 - inputs['sdc/current/bbox_yaw']

    rg_x, rg_y = rotate_points_around_origin(rg_x, rg_y, angle)
    # rg_dx, rg_dy = rotate_points_around_origin(rg_dx, rg_dy, angle)

  # print(inputs['roadgraph_samples/dir'][0])
  nrg_x,nrg_y= rg_x[..., tf.newaxis] , rg_y[..., tf.newaxis] 
  
  points_x,points_y,map_mask = _transform_to_image_coordinates(rg_x, rg_y,config)
  map_mask = map_mask[...,tf.newaxis]
  map_mask = tf.logical_and(map_mask,tf.cast(rg_valid,tf.bool))
  # print(map_mask.get_shape())

  im_rg_x, im_rg_y = points_x[..., tf.newaxis] , points_y[..., tf.newaxis]
  xy_val = tf.concat([im_rg_x, im_rg_y], axis=-1)

  map_traj = tf.concat([rg_x[..., tf.newaxis], rg_y[..., tf.newaxis],
  rg_dx[..., tf.newaxis], rg_dy[..., tf.newaxis]], axis=-1)


  # FOR TRAJECOTRIES #
  times = ['past','current']
  x = _stack_field(inputs, times, 'x')
  y = _stack_field(inputs, times, 'y')
  z = _stack_field(inputs, times, 'z')

  vx = _stack_field(inputs, times, 'velocity_x')
  vy = _stack_field(inputs, times, 'velocity_y')

  bbox_yaw = _stack_field(inputs, times, 'bbox_yaw')

  length = _stack_field(inputs, times, 'length')
  width = _stack_field(inputs, times, 'width')

  valid = _stack_field(inputs, times, 'valid')
  valid_indices = tf.cast(tf.equal(valid, 1),tf.float32)
 
  shape = ['batch_size', 'num_agents', 'num_steps', 1]
  tf.debugging.assert_shapes([
      (x, shape),
      (y, shape),
      (vx, shape),
      (vy, shape),
      (z, shape),
      (bbox_yaw, shape)
  ])

  # Translate all agent coordinates such that the autonomous vehicle is at the
  # origin.
  sdc_x = inputs['sdc/current/x'][:, tf.newaxis, tf.newaxis, :]
  sdc_y = inputs['sdc/current/y'][:, tf.newaxis, tf.newaxis, :]
  sdc_z = inputs['sdc/current/z'][:, tf.newaxis, tf.newaxis, :]

  sdc_vx = inputs['sdc/current/velocity_x'][:, tf.newaxis, tf.newaxis, :]
  sdc_vy = inputs['sdc/current/velocity_y'][:, tf.newaxis, tf.newaxis, :]

  x = x - sdc_x
  y = y - sdc_y
  z = z - sdc_z

  # vx = vx - sdc_vx
  # vy = vy - sdc_vy

  angle = math.pi / 2 - inputs['sdc/current/bbox_yaw'][:, tf.newaxis,
                                                        tf.newaxis, :]
  
  x, y = rotate_points_around_origin(x, y, angle)

  _,_,psudo_occu_mask = _transform_to_image_coordinates(x[:,:,-1,:], y[:,:,-1,:], config,larger_box=True)

  ul_x,ul_y, ur_x,ur_y, ll_x,ll_y, lr_x,lr_y = _rotate_box(x,y,length,width,bbox_yaw+angle)

  _,_,in_box_lu = _transform_to_image_coordinates(ul_x,ul_y,config)
  _,_,in_box_ru = _transform_to_image_coordinates(ur_x,ur_y,config)
  _,_,in_box_ld = _transform_to_image_coordinates(ll_x,ll_y,config)
  _,_,in_box_rd = _transform_to_image_coordinates(lr_x,lr_y,config)

  in_box = tf.logical_or(
    tf.logical_or(in_box_lu,in_box_ru), 
    tf.logical_or(in_box_ld,in_box_rd)
    )

  # print(in_box.get_shape())
  in_box_mask = tf.not_equal(tf.reduce_sum(tf.cast(in_box,tf.int32)[:,:,:,0],axis=-1),0)

  occu_mask = tf.logical_and(psudo_occu_mask[:,:,0],tf.logical_not(in_box_mask))
  # print(in_box_mask.get_shape())
  vx, vy = rotate_points_around_origin(vx, vy, angle)
  bbox_yaw = bbox_yaw #+ angle

  actor_traj = tf.multiply(valid_indices,tf.concat([x,y,vx,vy,bbox_yaw], axis=-1))

  return xy_val,map_traj,map_mask, actor_traj,in_box_mask,occu_mask,valid


def _rotate_box(x,y,length,width,bbox_yaw):
  sin_yaw = tf.sin(bbox_yaw)
  cos_yaw = tf.cos(bbox_yaw)

  #upper-left
  ul_x = cos_yaw * length * 0.5 - sin_yaw * width * (-0.5) + x
  ul_y = sin_yaw * length * 0.5 + cos_yaw * width * (-0.5) + y
  
  #upper-right
  ur_x = cos_yaw * length * 0.5 - sin_yaw * width * (0.5) + x
  ur_y = sin_yaw * length * 0.5 + cos_yaw * width * (0.5) + y

  #lower-left
  ll_x = cos_yaw * length * (-0.5) - sin_yaw * width * (-0.5) + x
  ll_y = sin_yaw * length * (-0.5) + cos_yaw * width * (-0.5) + y

  #lower-right
  lr_x = cos_yaw * length * (-0.5) - sin_yaw * width * (0.5) + x
  lr_y = sin_yaw * length * (-0.5) + cos_yaw * width * (0.5) + y

  return ul_x,ul_y, ur_x,ur_y, ll_x,ll_y, lr_x,lr_y
