import math
import os
import time
import datetime
import torch
import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
import warp.optim
from enum import Enum

import utils
from object_loader import ObjectLoader
from integrator_euler_fem import FEMIntegrator
from tendon_model import TendonModelBuilder, TendonRenderer, TendonHolder
from init_pose import InitializeFingers

@wp.kernel
def update_materials(
    # frame_id: wp.int32,
    log_K: wp.array(dtype=wp.float32),
    opt_v: wp.array(dtype=wp.float32),
    block_ids: wp.array(dtype=wp.int32),
    tet_materials: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    this_log_k = log_K[block_ids[tid]]
    K = wp.exp(this_log_k)
    v = opt_v[0]
    k_mu = K / (2.0 * (1.0 + v))
    k_lambda = K * v / ((1.0 + v) * (1.0 - 2.0 * v))

    tet_materials[tid, 0] = k_mu
    tet_materials[tid, 1] = k_lambda

@wp.kernel
def sample_logk(
    kernel_seed: wp.int32,
    min_val: wp.float32, max_val: wp.float32,
    log_K: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    state = wp.rand_init(kernel_seed, tid)
    log_K[tid] = wp.randf(state, min_val, max_val)

@wp.kernel
def apply_gravity(
    gravity: wp.vec3,
    body_mass: wp.array(dtype=wp.float32),
    body_f: wp.array(dtype=wp.spatial_vector)
    ):
    tid = wp.tid()
    wp.atomic_add(body_f, tid, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0*gravity[1] * body_mass[tid], 0.0))

class FEMTendon:
    def __init__(self, 
                 stage_path="sim", num_frames=30, 
                 verbose=True, save_log=False,
                 train_iters=100,
                 log_prefix="", 
                 is_render=True,
                 use_graph=False,
                 kernel_seed=42,
                 object_rot=wp.quat_identity(),
                 ycb_object_name='',
                 object_density=1e1,
                 finger_len=9, finger_rot=np.pi/9, finger_width=0.08, scale=4.0, finger_transform=None,
                 init_finger=None):
        self.verbose = verbose
        self.save_log = save_log
        fps = 4000
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames

        self.sim_substeps = 100
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.render_time = 0.0
        self.iter = 0
        self.train_iters = train_iters
        self.requires_grad = False

        self.obj_loader = ObjectLoader()
        self.finger_len = finger_len # need to be odd number
        self.finger_num = 2
        self.finger_rot = finger_rot
        self.finger_width = finger_width
        self.finger_transform = finger_transform
        self.init_finger = init_finger
        self.scale = scale
        self.obj_name = 'ycb'
        self.ycb_object_name = ycb_object_name
        self.object_density = object_density

        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.stage_path = self.curr_dir + "/../output/" + stage_path + f"{ycb_object_name}_{log_prefix}_frame{num_frames}" + ".usd"

        save_dir = self.curr_dir + "/../data_gen/" + f"{ycb_object_name}_frame{num_frames}/rand/"
        if save_log and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created.")
        self.save_dir = save_dir

        self.builder = TendonModelBuilder()
        utils.load_object(self.builder, self.obj_loader,
                          object=self.obj_name,
                          ycb_object_name=self.ycb_object_name,
                          obj_rot=object_rot,
                          scale=self.scale,
                          use_simple_mesh=False,
                          is_fix=False,
                          density=object_density,
                          )
        self.builder.init_builder_tendon_variables(self.finger_num, self.finger_len, self.scale, self.requires_grad)
        self.builder.build_fem_model(
            finger_width=self.finger_width,
            finger_rot=self.finger_rot,
            obj_loader=self.obj_loader,
            finger_transform=self.finger_transform,
            is_triangle=True
            )
        self.model = self.builder.model
        self.control = self.model.control(requires_grad=self.requires_grad)

        self.tendon_holder = TendonHolder(self.model, self.control)
        self.integrator = FEMIntegrator()

        self.log_K_warp = None
        self.kernel_seed = kernel_seed

        self.init_tendons()
        self.init_materials()

        # allocate sim states
        self.states = []
        for i in range(self.sim_substeps * self.num_frames + 1):
            self.states.append(self.model.state(requires_grad=self.requires_grad))
        self.init_particle_q = self.states[0].particle_q.numpy()[0, :]
        self.init_body_q = self.states[0].body_q.numpy()[0, :]
        self.object_body_f = self.states[0].body_f
        self.object_q = self.states[0].body_q
        self.object_grav = np.zeros(6)
        self.log_K_list = []
        self.save_list = []
        self.mass_list = []
        self.run_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.file_name = self.save_dir + f"{self.run_name}.npz"

        if self.stage_path and is_render:
            self.renderer = TendonRenderer(self.model, self.stage_path, scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = False
        if use_graph:
            self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            print("Creating CUDA graph...")
            with wp.ScopedCapture() as capture:
                self.forward()
            self.graph = capture.graph

    def init_tendons(self):
        finger_waypoint_num = [len(self.builder.waypoints[i]) for i in range(self.finger_num)]
        self.tendon_holder.finger_len = self.finger_len
        self.tendon_holder.finger_num = self.finger_num
        
        # waypoints related
        self.tendon_holder.finger_waypoint_num= wp.array(
            np.hstack([np.ones(finger_waypoint_num[i])*i for i in range(self.finger_num)]), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.waypoints = wp.array(
            np.array(self.builder.waypoint_ids).flatten(), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.waypoint_pair_ids = wp.array(
            np.array(self.builder.waypoint_pair_ids).flatten(), 
            dtype=wp.int32, requires_grad=False)
        self.tendon_holder.tendon_tri_indices = wp.array(
            np.array(self.builder.waypoints_tri_indices).flatten(), 
            dtype=wp.int32, requires_grad=False)
        
        # force related
        self.tendon_holder.surface_normals = wp.array(
            np.zeros([len(self.tendon_holder.waypoints) * self.finger_num, 3]), 
            dtype=wp.vec3, requires_grad=self.requires_grad)
        self.success_flag = wp.array([1], dtype=wp.int32, requires_grad=False)

        # control related
        self.control.finger_num = self.finger_num
        self.control.waypoint_forces = wp.array(np.zeros([len(self.tendon_holder.waypoints), 3]), dtype=wp.vec3, requires_grad=self.requires_grad)
        self.control.waypoint_ids = self.tendon_holder.waypoints
        # target vel related
        self.control.vel_values = wp.array(
            np.zeros([1, 3]),
            dtype=wp.vec3, requires_grad=self.requires_grad)
        self.tendon_forces = wp.array([100.0]*self.finger_num, dtype=wp.float32, requires_grad=self.requires_grad)

        self.tendon_holder.init_tendon_variables(requires_grad=self.requires_grad)
    
    def init_materials(self):
        # print("init_K:", self.builder.init_K)
        
        self.tet_block_ids = wp.array(self.builder.tet_block_ids, dtype=wp.int32, requires_grad=False)
        self.block_num = np.max(np.array(self.builder.tet_block_ids)) + 1
        self.finger_back_ids = wp.array(self.builder.finger_back_ids, dtype=wp.int32, requires_grad=False)
        self.all_ids = wp.array(np.arange(len(self.model.particle_q)), dtype=wp.int32, requires_grad=False)

        # self.log_K_warp = wp.from_numpy(np.zeros(len(self.model.tet_materials)) + np.log(self.builder.init_K), dtype=wp.float32, requires_grad=self.requires_grad)
        self.log_K_warp = wp.from_numpy(np.zeros(self.block_num) + np.log(13.5), dtype=wp.float32, requires_grad=self.requires_grad)
        self.v = wp.array([self.builder.init_v], dtype=wp.float32, requires_grad=self.requires_grad)

        wp.launch(update_materials,
                dim=len(self.model.tet_materials),
                inputs=[self.log_K_warp, self.v, self.tet_block_ids],
                outputs=[self.model.tet_materials])

    def forward(self):
        wp.launch(update_materials,
                dim=len(self.model.tet_materials),
                inputs=[self.log_K_warp, self.v,
                        self.tet_block_ids],
                outputs=[self.model.tet_materials])

        for frame in range(self.num_frames):

            for i in range(self.sim_substeps):
                index = i + frame * self.sim_substeps
                self.states[index].clear_forces()
                self.tendon_holder.reset()
                if i % 20 == 0:
                    wp.sim.collide(self.model, self.states[index])
                
                self.control.update_target_vel(frame)

                force = self.tendon_forces

                self.tendon_holder.apply_force(force, self.states[index].particle_q, self.success_flag)
                self.integrator.simulate(self.model, self.states[index], self.states[index+1], self.sim_dt, self.control)

                self.object_body_f = self.states[index].body_f

        self.object_q = self.states[-1].body_q
    
    def simulate(self):
        # sample random stiffness
        wp.launch(sample_logk,
                dim=len(self.log_K_warp),
                inputs=[self.kernel_seed + self.iter, 13.5, 17.0],
                outputs=[self.log_K_warp])
        print(f"iter:{self.iter}, sampled log_K:", self.log_K_warp.numpy()[:10])

        # sample density
        new_density = np.random.uniform(1e0, 1e1)
        utils.update_object_density(self.model, 0, new_density, self.object_density)
        self.object_density = new_density
        print("object density:", self.object_density)
        print("object mass:", self.model.body_mass.numpy())

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.forward()

        end_particle_q = self.states[-1].particle_q.numpy()[0, :]
        particle_trans_q = end_particle_q[0:3] - self.init_particle_q[0:3]
        object_trans_q = self.object_q.numpy()[0, 0:3] - self.init_body_q[0:3]

        diff_q = object_trans_q - particle_trans_q
        wp.sim.collide(self.model, self.states[-1])
        collide_num = self.model.rigid_contact_count.numpy()
        print("collision:", collide_num)

        if np.linalg.norm(diff_q) > 1e2:
            print(f'Error in diff_q {self.ycb_object_name}')
            return
        object_body_f = self.object_body_f.numpy().flatten()
        object_grav = -9.8 * self.model.body_mass.numpy()[0]
        self.object_grav[4] = object_grav

        if self.success_flag.numpy()[0] == 0:
            print(f'Error in simulation {self.ycb_object_name}')
            collide_num += 1
            diff_q[1] -= 0.1
            object_body_f += self.object_grav 
        if np.linalg.norm(object_body_f) < 1e-5:
            print(f'Zero in object body force {self.ycb_object_name}')
            collide_num += 1

        object_body_f += self.object_grav 
        print(f'{self.ycb_object_name} {self.iter} body_f:', object_body_f)
        print(f'{self.ycb_object_name} {self.iter} body_q:', diff_q)

        this_output = object_body_f.tolist()
        this_output += (diff_q).flatten().tolist()
        this_output += collide_num.flatten().tolist()

        self.save_list.append(this_output)
        self.log_K_list.append(self.log_K_warp.numpy().flatten().tolist())
        self.mass_list.append(self.model.body_mass.numpy().flatten().tolist())
        self.density_list.append([self.object_density])

        if self.save_log:
            if (self.iter+1) % 10 == 0 or self.iter == self.train_iters - 1:
                self.save_data()
        self.sim_time += self.frame_dt
        self.iter += 1
    
    def reset_states(self, finger_mesh):
        for i in range(self.finger_num):
            wp.launch(utils.update_particles_input,
                      dim=len(finger_mesh[i]),
                      inputs=[self.states[0].particle_q, 
                              finger_mesh[i],
                              i*len(finger_mesh[i]),]
                      )
        print("reset body_q:", self.states[0].body_q.numpy())
        self.states[0].body_qd.zero_()
        self.states[0].particle_qd.zero_()
        self.success_flag = wp.array([1], dtype=wp.int32, requires_grad=False)
        self.object_mass = self.model.body_mass.numpy()[0]
        self.log_K_list = []
        self.save_list = []
        self.mass_list = []
        self.density_list = []

        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        self.file_name = self.save_dir + f"{self.run_name}.npz"

    def save_data(self):
        np.savez(self.file_name,
                 log_k=np.array(self.log_K_list),
                 output=np.array(self.save_list),
                 init_transform=self.finger_transform,
                 object_mass=np.array(self.mass_list),
                 density_list=np.array(self.density_list),)
        print("Saved data to:", self.file_name)

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            for i in range(self.num_frames + 1):
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(
                    self.states[i * self.sim_substeps], 
                    np.array(self.builder.waypoint_ids).flatten(), 
                    force_scale=0.1)
                self.renderer.end_frame()
                self.render_time += self.frame_dt
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="sim",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=850, help="Total number of frames per training iteration.")
    parser.add_argument("--stiff_iters", type=int, default=3, help="Total number of sampling stiffness iterations.")
    parser.add_argument("--pose_iters", type=int, default=3, help="Total number of pose iterations.")
    parser.add_argument("--object_name", type=str, default="006_mustard_bottle", help="Name of the object to load.")
    parser.add_argument("--object_density", type=float, default=2e0, help="Density of the object.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--no_init", action="store_true", help="Automatically initialize the fingers.")
    parser.add_argument("--use_graph", action="store_true", help="Use CUDA graph for forward pass.")
    parser.add_argument("--save_log", action="store_true", help="Save the logs.")
    parser.add_argument("--log_prefix", type=str, default="", help="Prefix for the log file.")
    parser.add_argument("--pose_id", type=int, default=0, help="Initial pose id from anygrasp")
    parser.add_argument("--random", action="store_true", help="Add random noise to the initial position.")

    args = parser.parse_known_args()[0]

    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)
    is_triangle = True

    with wp.ScopedDevice(args.device):
        finger_transform = None
        init_finger = InitializeFingers( 
                            finger_len=finger_len, 
                            finger_rot=finger_rot,
                            finger_width=finger_width,
                            stop_margin=0.0005,
                            scale=scale,
                            num_envs=1,
                            ycb_object_name=args.object_name,
                            object_rot=object_rot,
                            num_frames=30, 
                            iterations=15000,
                            is_render=False,
                            is_triangle=is_triangle,
                            pose_id=args.pose_id,
                            add_random=args.random,)
            
        tendon = FEMTendon(
            stage_path=args.stage_path, 
            num_frames=args.num_frames, 
            verbose=args.verbose, 
            save_log=args.save_log,
            log_prefix=args.log_prefix,
            is_render=True,
            kernel_seed=np.random.randint(0, 10000),
            train_iters=args.pose_iters,
            object_rot=object_rot,
            object_density=args.object_density,
            ycb_object_name=args.object_name,
            use_graph=args.use_graph,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform,
            init_finger=init_finger)

        for i in range(args.pose_iters):
            init_finger.load_grasp_pose(-1)
            init_finger.reset_states()
            finger_transform, jq = init_finger.get_initial_position()
            if finger_transform is None or jq is None:
                print("Failed to initialize fingers.")
                continue

            tendon.finger_transform = finger_transform
            tendon.reset_states(init_finger.curr_finger_mesh)
            for j in range(args.stiff_iters):
                tendon.simulate()
            tendon.save_data()

        tendon.render()
        tendon.renderer.save()


