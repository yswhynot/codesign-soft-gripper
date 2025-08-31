import os
import numpy as np
import torch

import warp as wp
import utils

from tendon_model import TendonModel, TendonModelBuilder, TendonRenderer, TendonHolder
from object_loader import ObjectLoader

class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, transform_9d, 
                transform_2d,
                model,
                state_in,
                state_out,
                integrator,
                sim_dt,
                curr_finger_mesh,
                finger_mesh,
                finger_body_ids,
                object_com,
                distance_param,
                total_dis
                ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.state_in = state_in
        ctx.state_out = state_out
        ctx.intergrator = integrator
        ctx.sim_dt = sim_dt
        ctx.transform_9d = wp.from_torch(transform_9d)
        ctx.transform_2d = wp.from_torch(transform_2d)
        ctx.joint_q = model.joint_q

        ctx.curr_finger_mesh = curr_finger_mesh
        
        ctx.finger_mesh = finger_mesh
        ctx.total_dis = total_dis

        with ctx.tape:
            ctx.state_in.clear_forces()
            wp.launch(utils.transform_from11d, dim=1, 
                    inputs=[ctx.transform_9d, ctx.transform_2d],
                    outputs=[ctx.joint_q])
            
            wp.sim.eval_fk(ctx.model, ctx.joint_q, ctx.model.joint_qd, None, ctx.state_in)
            
            # wp.sim.collide(ctx.model, ctx.state_in)
            ctx.intergrator.simulate(ctx.model, ctx.state_in, ctx.state_out, ctx.sim_dt)
            for i in range(len(curr_finger_mesh)):
                wp.launch(utils.transform_points_out,
                        dim=len(ctx.curr_finger_mesh[i]),
                        inputs=[ctx.finger_mesh[i],
                                finger_body_ids[i],
                                ctx.state_in.body_q],
                        outputs=[ctx.curr_finger_mesh[i]])
            
                wp.launch(utils.mesh_dis, 
                        dim=len(ctx.curr_finger_mesh[i]),
                        inputs=[ctx.model.shape_geo, 
                                0, 0,
                                ctx.state_out.body_q,
                                ctx.curr_finger_mesh[i],
                                # ctx.model.rigid_contact_margin*1.0,
                                ctx.model.object_contact_margin*1.0,
                                1e-1,
                                1e3, 0],
                        outputs=[ctx.total_dis])
        return (wp.to_torch(ctx.total_dis))
    
    @staticmethod
    def backward(ctx, adj_total_dis):
        max_grad_trans = 1.0
        max_grad_rot = 1e-8
        ctx.total_dis.grad = wp.from_torch(adj_total_dis)
        ctx.tape.backward()

        trans9d_grad = wp.to_torch(ctx.tape.gradients[ctx.transform_9d]).clone()
        trans2d_grad = wp.to_torch(ctx.tape.gradients[ctx.transform_2d]).clone()
        utils.remove_nan(trans9d_grad)
        utils.remove_nan(trans2d_grad)
        trans9d_grad.clamp_(-max_grad_trans, max_grad_trans)
        trans2d_grad.clamp_(-max_grad_trans, max_grad_trans)
        trans9d_grad[0].zero_()
        trans9d_grad[2].zero_()
        trans9d_grad[3:].zero_()

        ctx.tape.zero()
     
        return tuple([trans9d_grad] + [trans2d_grad] + [None]*11) # 11

class InitializeFingers:
    def __init__(self, stage_path="femtendon_sim.usd", 
                 finger_len=9, finger_rot=0.01,
                 finger_width=0.08,
                 stop_margin=0.01,
                 num_frames=30, 
                 iterations=10000,
                 scale=5.0,
                 num_envs=1,
                 ycb_object_name='',
                 object_rot=wp.quat_identity(),
                 is_render=True,
                 verbose=False,
                 is_triangle=False,
                 pose_id=0,
                 add_random=False,
                 init_height_offset=0.0,
                 post_height_offset=0.0, 
                 is_ood=False,):
        self.pose_id = pose_id
        self.verbose = verbose
        self.is_render = is_render
        self.flag = True
        fps = 100
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames
        self.train_iter = iterations
        self.num_envs = num_envs
        self.is_triangle = is_triangle
        self.add_random = add_random

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.render_time = 0.0
        self.requires_grad = True 
        self.is_ood = is_ood

        self.torch_device = wp.device_to_torch(wp.get_device())

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        if stage_path:
            self.stage_path = curr_dir + "/../output/" + stage_path

        self.obj_loader = ObjectLoader()
        self.obj_name = 'ycb'
        self.ycb_object_name = ycb_object_name
        if finger_len % 2 == 0:
            raise ValueError("finger_len should be odd number")
        if finger_rot < 0.0 or finger_rot > np.pi/4:
            raise ValueError("finger_rot should be in [0, pi/4]")
        self.finger_num = 2
        self.finger_len = finger_len # need to be odd number
        self.finger_rot = finger_rot
        self.finger_width = finger_width
        self.stop_margin = stop_margin
        self.scale = scale
        self.object_com = None
        self.init_height_offset = init_height_offset
        self.post_height_offset = post_height_offset
        
        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.build_rigid_model(object_rot)
        self.object_com = wp.to_torch(self.object_com, requires_grad=True)
        self.joint_q = wp.array(self.model.joint_q.numpy(), dtype=wp.float32, requires_grad=True)
        self.transform_9d_wp= wp.zeros(9, dtype=wp.float32, requires_grad=True)
        self.transform_2d_wp= wp.zeros(2, dtype=wp.float32, requires_grad=True)

        wp.launch(utils.transform_to11d, dim=1,
                  inputs=[self.joint_q],
                  outputs=[self.transform_9d_wp, self.transform_2d_wp])
        
        self.state0 = self.model.state(requires_grad=True)
        self.state1 = self.model.state(requires_grad=True)
        self.total_dis = wp.array([0.0], dtype=wp.float32, requires_grad=True)
        
        # things to optimize in torch
        self.transform_9d = wp.to_torch(self.transform_9d_wp, requires_grad=True)
        self.transform_2d = wp.to_torch(self.transform_2d_wp, requires_grad=True)

        self.optimizer = torch.optim.SGD([
            {'params': self.transform_9d, 'lr': 1e-1, 'weight_decay': 1e-5},
            {'params': self.transform_2d, 'lr': 1e-2}, 
        ])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.05)

        # loss
        self.loss = 0
        self.init_count = 0
        
        if stage_path and is_render:
            self.renderer = TendonRenderer(self.model, self.stage_path, scaling=1.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        self.use_cuda_graph = False
    
    def build_rigid_model(self, object_rot):
        soft_builder = TendonModelBuilder()
        soft_builder.init_builder_tendon_variables(self.finger_num, self.finger_len, self.scale, self.requires_grad)
        self.init_transforms, finger_actual_len, finger_THK = soft_builder.build_fem_model(
            finger_width=self.finger_width,
            finger_rot=self.finger_rot,
            obj_loader=None,
            h_dis_init=0.2,
            is_triangle=self.is_triangle,)
        for init_trans in self.init_transforms:
            print("init transforms:", wp.transform_get_translation(init_trans), wp.transform_get_rotation(init_trans))

        # seperate two fingers into two meshs
        v_num, face_num = len(soft_builder.particle_q), len(soft_builder.tri_indices)
        vertices_0, vertices_1 = soft_builder.particle_q[:v_num // 2], soft_builder.particle_q[v_num // 2:]
        faces_0, faces_1 = soft_builder.tri_indices[:face_num // 2], soft_builder.tri_indices[:face_num // 2]
        finger_mesh = [wp.sim.Mesh(vertices_0, faces_0), wp.sim.Mesh(vertices_1, faces_1)]
        self.finger_mesh = [wp.array(vertices_0, dtype=wp.vec3, requires_grad=True),
                            wp.array(vertices_1, dtype=wp.vec3, requires_grad=True)]
        self.curr_finger_mesh = [wp.array(vertices_0, dtype=wp.vec3, requires_grad=True), 
                                 wp.array(vertices_1, dtype=wp.vec3, requires_grad=True)]
        
        self.builder = wp.sim.ModelBuilder()
        self.builder.add_articulation()

        if self.is_ood:
            curr_dir = os.path.dirname(os.path.realpath(__file__))
            self.obj_loader.data_dir = curr_dir + "/../models/ood/"
        
        self.object_com, self.obj_body_id, self.obj_geo_id = utils.load_object(
                        self.builder, self.obj_loader,
                          object=self.obj_name,
                          ycb_object_name=self.ycb_object_name,
                          obj_rot=object_rot,
                          scale=self.scale,
                          use_simple_mesh=False,
                          is_fix=True
                          )

        self.finger_body_idx = []
        for _ in range(self.finger_num):
            self.finger_body_idx.append(
                self.builder.add_body(origin=wp.transform_identity()))

        self.finger_shape_ids = []
        wrist_body = self.builder.add_body(origin=wp.transform_identity())
        self.builder.add_joint_free(parent=-1, child=wrist_body)
        limit_low, limit_upp = 1*finger_THK, 6*finger_THK 
        self.limit_low, self.limit_upp = limit_low, limit_upp

        for i in range(self.finger_num):
            finger_shape_id = self.builder.add_shape_mesh(
                body=self.finger_body_idx[i], 
                                        mesh=finger_mesh[i], 
                                        density=1e1,
                                        ke=1.0e2,
                                        kd=1.0e-5,
                                        kf=1e1,
                                        mu=1.0)
            self.finger_shape_ids.append(finger_shape_id)
        self.builder.add_joint_prismatic(parent=wrist_body, child=self.finger_body_idx[0], axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=-limit_upp, limit_upper=-limit_low)
        self.builder.add_joint_prismatic(parent=wrist_body, child=self.finger_body_idx[1], axis=wp.vec3(1.0, 0.0, 0.0), limit_lower=limit_low, limit_upper=limit_upp)
        self.limit_low, self.limit_upp = [limit_low, limit_upp]
        
        self.model = self.builder.finalize(requires_grad=True)
        self.load_grasp_pose(self.pose_id)
        
        self.model.object_contact_margin = self.stop_margin * self.scale

    def load_grasp_pose(self, pose_id):
        self.pose_id = pose_id
        # initialization convert
        T_kc, T_gk, T_cb, T_gb = np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        path = os.path.dirname(os.path.realpath(__file__)) + "/../pose_info/"
        T_path = path + str(self.ycb_object_name)+".npz"
        data = np.load(T_path)
        total_pose = data['T_pose'].shape[0]
        self.total_pose = total_pose
        if self.pose_id == -1:
            self.pose_id = np.random.randint(0, total_pose)
        pose_id = self.pose_id % total_pose
        print("Loading pose id:", pose_id, "out of total poses:", total_pose)
        T_kc = data['T_pose'][pose_id]
        if self.add_random:
            T_kc = utils.add_random_to_pose(
                T_kc, 
                t_std=3e-3*self.scale,
                r_std=5e-3)
        
        T_gk[:3, :3] = np.array([[ 0.0, -1.0, 0.0],
                                 [-1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0]])
        T_gk[:3, 3] = np.array([0.0, 0.0, 0.0])
        
        T_gc = np.matmul(T_kc, T_gk)
        
        T_cb[:3, :3] = np.array([[-1.0, 0.0, 0.0],
                                 [0.0, 0.0, -1.0],
                                 [0.0, -1.0, 0.0]])
        T_cb[:3, 3] = np.array([0.0, 0.5, 0.0])
        T_gb = np.matmul(T_cb, T_gc)
        
        R_gb = T_gb[:3, :3]
        t_gb = T_gb[:3, 3] * self.scale
        t_gb[1] += self.init_height_offset

        R_quat = utils.mat33_to_quat(R_gb)
        self.model.joint_q = wp.array(t_gb.tolist() + R_quat.flatten().tolist() + [-self.limit_upp, self.limit_upp], dtype=wp.float32, requires_grad=True)

    def reset_states(self):
        self.joint_q = wp.array(self.model.joint_q.numpy(), dtype=wp.float32, requires_grad=True)

        wp.launch(utils.transform_to11d, dim=1,
                  inputs=[self.joint_q],
                  outputs=[self.transform_9d_wp, self.transform_2d_wp])
        
        self.state0 = self.model.state(requires_grad=True)
        self.state1 = self.model.state(requires_grad=True)

        with torch.no_grad():
            self.transform_9d.copy_(wp.to_torch(self.transform_9d_wp))
            self.transform_2d.copy_(wp.to_torch(self.transform_2d_wp))

        for i in range(2):
            self.curr_finger_mesh[i] = wp.array(self.finger_mesh[i].numpy(), dtype=wp.vec3, requires_grad=True)
        self.sim_time = 0.0
        self.render_time = 0.0
        self.total_dis.zero_()

        self.optimizer = torch.optim.SGD([
            {'params': self.transform_9d, 'lr': 1e-1, 'weight_decay': 1e-5},
            {'params': self.transform_2d, 'lr': 1e-2}, 
        ])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=50, factor=0.05)
        
    def forward(self, distance_param, use_com):
        total_dis_torch = ForwardKinematics.apply(
            self.transform_9d, 
            self.transform_2d,
            self.model,
            self.state0,
            self.state1,
            self.integrator,
            self.sim_dt,
            self.curr_finger_mesh,
            self.finger_mesh,
            self.finger_body_idx,
            self.object_com,
            distance_param,
            self.total_dis)
        return total_dis_torch

    def compute_loss(self, total_dis_torch):
        self.loss = torch.norm(total_dis_torch) ** 2.0

    def step(self, iter, distance_param=1.0, use_com=False):
        def closure():
            total_dis_torch = self.forward(
                                distance_param, 
                                use_com)
            self.compute_loss(total_dis_torch)
            self.loss.backward()
            return self.loss

        self.optimizer.step(closure)
        with torch.no_grad():
            self.transform_2d[0].clamp_(-self.limit_upp, -self.limit_low)
            self.transform_2d[1].clamp_(self.limit_low, self.limit_upp)
        if iter % 1000 == 0:
            print("iter:", iter, "loss:", self.loss)

        self.state0, self.state1 = self.state1, self.state0
        self.optimizer.zero_grad()
        self.total_dis.zero_()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(
                self.state1, 
            )
            
            self.renderer.end_frame()
            self.render_time += self.frame_dt
    
    def get_initial_position(self, init_trans=None):
        prev_loss = 1e10
        convergence_threshold = 1e-5
        loss_diff_threshold = 1e-8
        patiance = 50
        stagnant_epochs = 0

        distance_param = 1e-1
        use_com = True

        for i in range(self.train_iter):

            self.step(i, distance_param, use_com=use_com)
            self.scheduler.step(self.loss)
            
            if self.loss < convergence_threshold:
                stagnant_epochs += 1
                if stagnant_epochs > patiance:
                    print("Converged at iteration:", i, "loss:", self.loss)
                    break
            elif np.abs(self.loss.item() - prev_loss) < loss_diff_threshold:
                stagnant_epochs += 1
                if stagnant_epochs > patiance:
                    print("Converged at iteration:", i, "loss:", self.loss)
                    break
            else:
                stagnant_epochs = 0
            prev_loss = self.loss.item()

            if self.is_render: self.render()

        if self.loss.item() > 1.0:
            print("Warning: Loss did not converge properly. Current loss:", self.loss.item())
            self.init_count += 1
            return None, None

        jq = self.model.joint_q.numpy()
        joint_trans = wp.transform(wp.vec3(jq[0], jq[1], jq[2]), 
                                  wp.quat(jq[3], jq[4], jq[5], jq[6]))
        body_trans = []
        for i in range(self.finger_num):
            body_q = self.state1.body_q.numpy()[self.finger_body_idx[i]]
            body_trans.append(
                wp.transform(wp.vec3(body_q[0], body_q[1]+self.post_height_offset, body_q[2]),
                wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])))
         
        init_trans = [wp.transform_multiply(body_trans[i], self.init_transforms[i]) for i in range(self.finger_num)]
        
        return init_trans, jq
    
    def get_fixed_position(self, joint_q):
        assert joint_q.shape[0] == self.joint_q.shape[0], "joint_q shape mismatch"
        self.joint_q = wp.array(joint_q, dtype=wp.float32, requires_grad=True)
        wp.sim.eval_fk(self.model, self.joint_q, self.model.joint_qd, None, self.state0)
        
        self.integrator.simulate(self.model, self.state0, self.state1, self.sim_dt)
        
        if self.is_render: self.render()

        jq = self.model.joint_q.numpy()
        joint_trans = wp.transform(wp.vec3(jq[0], jq[1], jq[2]), 
                                  wp.quat(jq[3], jq[4], jq[5], jq[6]))
        body_trans = []
        for i in range(self.finger_num):
            body_q = self.state1.body_q.numpy()[self.finger_body_idx[i]]
            body_trans.append(
                wp.transform(wp.vec3(body_q[0], body_q[1]+self.post_height_offset, body_q[2]),
                wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])))
         
        init_trans = [wp.transform_multiply(body_trans[i], self.init_transforms[i]) for i in range(self.finger_num)]
        
        return init_trans, jq

    def print_transform(self, trans):
        print("translation:", wp.transform_get_translation(trans))
        print("rotation:", wp.transform_get_rotation(trans))

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
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument("--pose_id", type=int, default=0, help="Initial pose id from anygrasp")
    parser.add_argument("--num_frames", type=int, default=30, help="Total number of frames per training iteration.")
    parser.add_argument("--train_iters", type=int, default=15000, help="Total number of training iterations.")
    parser.add_argument("--object_name", type=str, default="013_apple", help="Name of the object to load.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument("--random", action="store_true", help="Add randomness to the loaded pose.")

    args = parser.parse_known_args()[0]
    finger_transform, end_state = None, None
    with wp.ScopedDevice(args.device):
        this_pose_id = 0
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = curr_dir + "/../pose_info/init_opt/" + str(args.object_name) + ".npz"
        transform_arr = []
        while True:
            tendon = InitializeFingers(stage_path=args.stage_path, 
                                    finger_len=11, finger_rot=np.pi/30,
                                    finger_width=0.08,
                                    stop_margin=0.0005,
                                    scale=5.0,
                                    num_envs=args.num_envs,
                                    is_render=False,
                                    ycb_object_name=args.object_name,
                                    object_rot=wp.quat_rpy(-np.pi/2, 0.0, 0.0),
                                    num_frames=args.num_frames, 
                                    iterations=args.train_iters,
                                    verbose=args.verbose,
                                    is_triangle=False,
                                    #    pose_id=args.pose_id,
                                    pose_id=this_pose_id,
                                    add_random=args.random)
            finger_transform, jq = tendon.get_initial_position()
            if finger_transform is None:
                finger_transform = np.zeros([2, 7])
            transform_arr.append(finger_transform)

            # break
            this_pose_id += 1
            if this_pose_id >= tendon.total_pose:
                break
        np.savez(file_name, finger_transform=transform_arr)
        print("saved to:", file_name)

        if tendon.renderer:
            tendon.renderer.save()