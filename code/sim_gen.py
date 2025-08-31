import math
import os
import time
import datetime
import numpy as np
import warp as wp

import csv
import utils
from object_loader import ObjectLoader
from integrator_euler_fem import FEMIntegrator
from tendon_model import TendonModel, TendonModelBuilder, TendonRenderer, TendonHolder
from init_pose import InitializeFingers
from evaluation import Evaluation
from scipy.spatial.transform import Rotation as R

@wp.kernel
def update_materials(
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
                 train_iters=1,
                 log_prefix="", 
                 is_render=True,
                 use_graph=False,
                 kernel_seed=42,
                 object_rot=wp.quat_identity(),
                 ycb_object_name='',
                 object_density=2e0,
                 finger_len=9, finger_rot=np.pi/9, finger_width=0.08, scale=4.0, finger_transform=None,
                 init_finger=None,
                 is_ood=False):
        self.verbose = verbose
        self.save_log = save_log
        fps = 5000
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

        save_dir = self.curr_dir + "/../data_gen/" + f"{ycb_object_name}_frame{num_frames}/fix/"
        if save_log and (not os.path.exists(save_dir)):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created.")
        self.save_dir = save_dir
        self.result_file_name = None # change if need to save result
        # self.result_file_name = self.curr_dir + f"/../experiments/density{self.object_density}_{log_prefix}.csv"

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
        object_grav = -9.8 * self.model.body_mass.numpy()[0]
        self.object_grav = np.zeros(6)
        self.object_grav[4] = object_grav

        self.log_K_list = []
        self.save_list = []
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
        # print("finger waypoint num:", finger_waypoint_num)
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
        self.tet_block_ids = wp.array(self.builder.tet_block_ids, dtype=wp.int32, requires_grad=False)
        self.block_num = np.max(np.array(self.builder.tet_block_ids)) + 1
        self.finger_back_ids = wp.array(self.builder.finger_back_ids, dtype=wp.int32, requires_grad=False)
        self.all_ids = wp.array(np.arange(len(self.model.particle_q)), dtype=wp.int32, requires_grad=False)

        # self.log_K_warp = wp.from_numpy(np.zeros(self.block_num) + 14.5, dtype=wp.float32, requires_grad=self.requires_grad)
        self.log_K_warp = wp.from_numpy(np.array(
            [16.0477, 15.1574, 14.6199, 14.8867, 15.3114, 14.6197, 15.8461, 14.6456, 15.2328, 15.3423, 15.2193, 
            15.5395, 14.5256, 14.2653, 14.8140, 14.9995, 15.9662, 14.5387, 15.1426, 14.9531, 15.3118, 15.4673
        ]), dtype=wp.float32, requires_grad=self.requires_grad)
        self.v = wp.array([self.builder.init_v], dtype=wp.float32, requires_grad=True)

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
                if i % 10 == 0:
                    wp.sim.collide(self.model, self.states[index])
                
                self.control.update_target_vel(frame)

                force = self.tendon_forces

                self.tendon_holder.apply_force(force, self.states[index].particle_q, self.success_flag)
                self.integrator.simulate(self.model, self.states[index], self.states[index+1], self.sim_dt, self.control)

                self.object_body_f = self.states[index].body_f

            if frame % 100 == 0:
                print(f"frame {frame} / {self.num_frames}: body_f:", self.object_body_f.numpy().flatten())

        self.object_q = self.states[-1].body_q
    
    def simulate(self):
        # sample random stiffness
        # wp.launch(sample_logk,
        #         dim=len(self.log_K_warp),
        #         inputs=[self.kernel_seed + self.iter, 13.0, 17.0],
        #         outputs=[self.log_K_warp])
        print(f"iter:{self.iter}, log_K:", self.log_K_warp.numpy()[:10])

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
            print(f'Error in simulation {self.ycb_object_name}: success_flag false')
            collide_num += 1
        object_body_f += self.object_grav 
        print(f'{self.ycb_object_name} {self.iter} body_f:', object_body_f)
        print(f'{self.ycb_object_name} {self.iter} body_q:', diff_q)

        if self.result_file_name:
            with open(self.result_file_name, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f'{args.object_name}'])
                writer.writerow([f'collision:{collide_num.flatten().tolist()}'])
                writer.writerow([f'body_f norm: {np.linalg.norm(object_body_f)}, body_q norm: {np.linalg.norm(diff_q)}'])

        this_output = object_body_f.tolist()
        this_output += (diff_q).flatten().tolist()
        this_output += collide_num.flatten().tolist()
        self.save_list = this_output
        self.log_K_list = self.log_K_warp.numpy().flatten().tolist()

        if self.save_log:
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

        self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        self.file_name = self.save_dir + f"{self.run_name}.npz"

    def save_data(self):
        np.savez(self.file_name,
                 log_k=np.array(self.log_K_list),
                 output=np.array(self.save_list),
                 init_transform=self.finger_transform)
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
                    # self.control.waypoint_forces,
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
        default="femtendon_sim.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames per training iteration.")
    parser.add_argument("--train_iters", type=int, default=1, help="Total number of training iterations.")
    parser.add_argument("--object_name", type=str, default="006_mustard_bottle", help="Name of the object to load.")
    parser.add_argument("--object_density", type=float, default=2e0, help="Density of the object.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--no_init", action="store_true", help="Automatically initialize the fingers.")
    parser.add_argument("--use_graph", action="store_true", help="Use CUDA graph for forward pass.")
    parser.add_argument("--save_log", action="store_true", help="Save the logs.")
    parser.add_argument("--log_prefix", type=str, default="", help="Prefix for the log file.")
    parser.add_argument("--pose_id", type=int, default=0, help="Initial pose id from anygrasp")
    parser.add_argument("--render", action="store_true", help="Render the simulation.")
    
    # experiment related setup
    parser.add_argument("--model_name", type=str, default="", help="Path to the model file.")
    parser.add_argument("--load_optim", action="store_true", help="Load the optimized stiffness.")
    parser.add_argument("--load_best_pose", action="store_true", help="Load the best pose.")
    parser.add_argument("--pick_best_pose", action="store_true", help="Pick the best pose.")
    parser.add_argument("--log_k", type=float, default=0.0, help="Log stiffness value.")
    parser.add_argument("--ood", action="store_true", help="Use out-of-distribution data.")
    parser.add_argument("--evaluation", action="store_true", help="Evaluate the simulation.")

    args = parser.parse_known_args()[0]
    np.set_printoptions(suppress=True)

    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    scale = 5.0
    object_rot = wp.quat_rpy(-math.pi/2, 0.0, 0.0)
    is_triangle = True

    with wp.ScopedDevice(args.device):
        finger_transform = None
        pose_id = args.pose_id
        log_K = None
        if args.log_k > 0.0:
            log_K = np.zeros(22) + args.log_k
            args.log_prefix += f"logk{args.log_k}"
            print("Using input log_k:", log_K)

        if args.pick_best_pose:
            print("Picking best pose...")
            log_K = np.array([
                16.0477, 15.1574, 14.6199, 14.8867, 15.3114, 14.6197, 15.8461, 14.6456, 15.2328, 15.3423, 15.2193, 
                15.5395, 14.5256, 14.2653, 14.8140, 14.9995, 15.9662, 14.5387, 15.1426, 14.9531, 15.3118, 15.4673])
            assert args.model_name != "", "Please provide the model name to load."
            best_loss = 1e10
            best_pose_id = -1

            from optimize import StiffnessOptimizer
            optimizer = StiffnessOptimizer(
                model_name=args.model_name + ".pth",
                                       density=args.object_density,
                                       use_density=True,)
            if args.log_k > 0.0:
                log_K = np.zeros(22) + args.log_k
                args.log_prefix += f"logk{args.log_k}"
                print("Using input log_k:", log_K)
            for i in range(10):
                loss = 1e10
                loss = optimizer.get_loss(args.object_name, 
                                        i,
                                        log_K,
                                        args.object_density,)
                if loss is None: continue
                if loss < best_loss:
                    best_loss = loss
                    best_pose_id = i
            assert best_pose_id != -1, "No valid pose found."
            print("Best pose id:", best_pose_id, "with loss:", best_loss)
            pose_id = best_pose_id

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if args.load_best_pose:
            assert args.model_name != "", "Please provide the model name to load."
            assert args.load_optim, "Need to load the optimized stiffness."
            best_loss = 1e10
            best_pose_id = -1

            # get all files starting with the object_name
            for file in os.listdir(curr_dir + f"/../stiff_results/{args.model_name}"):
                if not file.startswith(args.object_name):
                    continue
                if not file.endswith(".npz"): continue
                this_density = float(file.split("density")[1].split(".")[0])
                if this_density != args.object_density:
                    continue
                this_pose_id = int(file.split("_pose")[1].split("density")[0])
                file_name = curr_dir + f"/../stiff_results/{args.model_name}/{file}"
                data = np.load(file_name, allow_pickle=True)
                loss = data['loss']
                if loss < best_loss:
                    best_loss = loss
                    best_pose_id = this_pose_id
            assert best_pose_id != -1, "No valid pose found."
            pose_id = best_pose_id
            print("Best pose id:", best_pose_id, "with loss:", best_loss)

        # try load from the processed dir
        pose_file = curr_dir + f"/../pose_info/init_opt/{args.object_name}.npz"
        try:
            data = np.load(pose_file, allow_pickle=True)
            finger_transform_list = data['finger_transform']
            this_trans = finger_transform_list[pose_id]
            if np.linalg.norm(this_trans) > 0.0:
                finger_transform = [
                    wp.transform(this_trans[0, :3], this_trans[0, 3:]),
                    wp.transform(this_trans[1, :3], this_trans[1, 3:])]
        except Exception as e:
            print(f"Error loading pose file: {e}")
            finger_transform = None

        init_finger = None
        if finger_transform is None:
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
                                    is_render=args.render,
                                    is_triangle=is_triangle,
                                    pose_id=pose_id,
                                    is_ood=args.ood,)
            finger_transform, jq = init_finger.get_initial_position()

            if init_finger.renderer:
                init_finger.renderer.save()

        if args.load_optim:
            assert args.model_name != "", "Please provide the model name to load."
            file_name = curr_dir + f"/../stiff_results/{args.model_name}/{args.object_name}_pose{pose_id}density{args.object_density}.npz"
            data = np.load(file_name, allow_pickle=True)
            log_K = data['stiffness']
            log_K = np.clip(log_K, 13.5, 17.0)
            args.log_prefix += f"{args.model_name}density{args.object_density}"
            print("Loaded stiffness from file:", log_K)

        tendon = FEMTendon(
            stage_path=args.stage_path, 
            num_frames=args.num_frames, 
            verbose=args.verbose, 
            save_log=args.save_log,
            log_prefix=args.log_prefix,
            train_iters=args.train_iters,
            object_rot=object_rot,
            object_density=args.object_density,
            ycb_object_name=args.object_name,
            kernel_seed=np.random.randint(0, 10000),
            use_graph=args.use_graph,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            is_render=args.render,
            scale=scale,
            finger_transform=finger_transform,
            init_finger=init_finger,
            is_ood=args.ood,)
        
        if log_K is not None:
            tendon.log_K_warp = wp.from_numpy(log_K, dtype=wp.float32, requires_grad=True)
        
        for i in range(args.train_iters):
            tendon.simulate()

        if args.render:
            print("Rendering...")
            tendon.render()

        if tendon.renderer:
            tendon.renderer.save()

        if args.evaluation:
            dist_f = -500.0
            eval_render = False
            print("Evaluating with dist:", dist_f, ", fix_force:", True)
            eval = Evaluation(tendon, is_render=eval_render)
            val = eval.evaluate(wp.array([wp.spatial_vector([0.0, 0.0, 0.0, 0.0, dist_f, 0.0])], dtype=wp.spatial_vector),
                                fix_force=True,
                                record_path=tendon.result_file_name)

            if eval_render:
                eval.renderer.save()

            print("Evaluating with dist:", dist_f, ", fix_force:", False)
            eval = Evaluation(tendon, is_render=False)
            val = eval.evaluate(wp.array([wp.spatial_vector([0.0, 0.0, 0.0, 0.0, dist_f, 0.0])], dtype=wp.spatial_vector),
                                fix_force=False,
                                record_path=tendon.result_file_name)
