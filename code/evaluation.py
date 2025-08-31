import os
import numpy as np
import warp as wp
import csv
import utils
from tendon_model import TendonRenderer

@wp.kernel
def add_body_f(ext_body_f: wp.array(dtype=wp.spatial_vector), 
               body_f: wp.array(dtype=wp.spatial_vector)):
    tid = wp.tid()
    body_f[tid] += ext_body_f[tid]

class Evaluation:
    def __init__(self, fem_tendon, num_frames=100, is_render=False, stage_path="eval.usd"):
        self.fem_tendon = fem_tendon
        self.model = fem_tendon.model
        self.states = fem_tendon.states
        self.integrator = fem_tendon.integrator
        self.tendon_holder = fem_tendon.tendon_holder

        self.sim_dt = fem_tendon.sim_dt
        self.sim_substeps = fem_tendon.sim_substeps
        self.is_render = is_render

        self.num_frames = num_frames
        self.control = fem_tendon.control
        self.tendon_holder.reset()
        self.curr_tendon_len = wp.clone(self.tendon_holder.get_tendon_length(self.states[-1].particle_q))
        init_control = self.curr_tendon_len.numpy()
        self.init_q = self.states[-1].body_q.numpy()[0, :]
        self.control.target_positions = wp.array(np.vstack((init_control,) * self.num_frames), dtype=wp.vec2)

        self.state0 = utils.copy_state(self.states[-1])
        self.state1 = utils.copy_state(self.states[-1])

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.render_time = 0.0
        if stage_path:
            self.stage_path = curr_dir + "/../output/" + stage_path
        if stage_path and is_render:
            self.renderer = TendonRenderer(self.model, self.stage_path, scaling=1.0)
        else:
            self.renderer = None

    def disturb(self, frame, ext_body_f: wp.spatial_vector, fix_force=False):
        force_inds = [0]
        force_norm = 0.0
        success_flag = wp.array([1], dtype=wp.int32, requires_grad=False)
        force_default = wp.array([100.0, 100.0], dtype=wp.float32, requires_grad=False)
        for i in range(self.sim_substeps):
            index = i + frame * self.sim_substeps
            self.state0.clear_forces()
            self.tendon_holder.reset()
            if frame in force_inds:
                wp.launch(add_body_f, dim=1, inputs=[ext_body_f], outputs=[self.state0.body_f])
            
            if i % 20 == 0:
                wp.sim.collide(self.model, self.state0)

            tendon_len = self.tendon_holder.get_tendon_length(self.state0.particle_q)
            force = self.control.force_from_position(tendon_len, self.control.target_positions, frame)
            if fix_force:
                force = force_default

            self.tendon_holder.apply_force(force, self.state0.particle_q, success_flag)
            self.integrator.simulate(self.model, self.state0, self.state1, self.sim_dt, self.control)

            if i == 0 and frame == self.num_frames-1: 
                print("tendon_len: ", tendon_len.numpy(), "force: ", force.numpy())
                print("body_f: ", self.state1.body_f.numpy())
                force_norm = np.linalg.norm(self.state1.body_f.numpy())

            self.state0 = self.state1
        return force_norm, force.numpy()

    def evaluate(self, ext_body_f: wp.spatial_vector, fix_force=False, object_name=None, record_path=None):
        force_norm = 0.0
        gripper_force = [] 
        for frame in range(self.num_frames):
            force_norm, gripper_force = self.disturb(frame, ext_body_f, fix_force=fix_force)
            if self.is_render:
                self.render()
        if self.is_render:
            self.renderer.save()
        self.ending_q = self.state0.body_q.numpy()[0, :]
        qd_norm = np.linalg.norm(self.state0.body_qd.numpy())
        init_q_arg = self.init_q.tolist()
        ending_q_arg = self.ending_q.tolist()
        diff = wp.transform_multiply(wp.transform_inverse(wp.transform(*init_q_arg)), wp.transform(*ending_q_arg))
        diff_arr = wp.array(wp.transform_get_translation(diff)).numpy()
        diff_norm = np.linalg.norm(diff_arr)
        
        if record_path:
            with open(record_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([f'force_norm, qd_norm, diff_norm, gripper_force fix_force: {fix_force}'])
            with open(record_path, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([force_norm, qd_norm, diff_norm, gripper_force])
            print("force_norm: ", force_norm)
            print("qd_norm: ", qd_norm)
            print("diff_norm: ", diff_norm)
            print("gripper_force: ", gripper_force)

        return diff_norm

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(
            self.state0, 
            force_scale=0.1)
        self.renderer.end_frame()
        self.render_time += 1.0