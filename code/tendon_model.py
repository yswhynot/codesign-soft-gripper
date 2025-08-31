import numpy as np
import warp as wp
import warp.sim.render
from warp.sim.model import *
import utils

@wp.kernel
def calculate_surface_normals(
    particle_num: wp.int32,
    tri_indices: wp.array(dtype=wp.int32), # len=particle_num
    particle_q: wp.array(dtype=wp.vec3), # len=particle_num
    tri_particle_indices: wp.array2d(dtype=wp.int32), # len=tri_num*3
    surface_normals: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()

    if tid >= particle_num: return
    idx = tri_indices[tid]
    p0 = particle_q[tri_particle_indices[idx][0]]
    p1 = particle_q[tri_particle_indices[idx][1]]
    p2 = particle_q[tri_particle_indices[idx][2]]
    n = wp.cross(p1 - p0, p2 - p0)

    wp.atomic_add(surface_normals, tid, wp.normalize(n))  

@wp.kernel
def calculate_force(
    particle_indices: wp.array(dtype=wp.int32), # len=particle_num, all ids of particles in the waypoints, 1d array
    particle_num: wp.int32,
    finger_length: wp.int32, # the number of waypoints in each fingers
    waypoint_pair_ids: wp.array(dtype=wp.int32), # len=particle_num
    particle_q: wp.array(dtype=wp.vec3), # len=particle_num
    surface_normals: wp.array(dtype=wp.vec3), # len=particle_num
    finger_waypoint_num: wp.array(dtype=wp.int32), # finger id of each waypoint, len=len(waypoints)
    external_force: wp.array(dtype=wp.float32), # len=particle_num
    # control_particle_activations: wp.array(dtype=wp.vec3)):
    waypoint_activations: wp.array(dtype=wp.vec3),
    success: wp.array(dtype=wp.int32)):

    tid = wp.tid()

    if tid >= particle_num: return
    # potential bug if fingers are not of the same length
    finger_id = int(finger_waypoint_num[tid])
    # wp_id = int(waypoint_pair_ids[tid] + finger_id * finger_length)
    this_pid = int(particle_indices[tid])
    
    # the first contact force is 0
    if tid == (finger_id * finger_length):
        return
    # the last contact force goes inside
    if tid == ((finger_id+1) * finger_length - 1):
        f_dir = particle_q[particle_indices[tid-1]] - particle_q[this_pid]
        if wp.length(f_dir) < 1e-4:
            success[0] = 0
            return
        f_dir_norm = wp.normalize(f_dir)
        f = f_dir_norm * external_force[finger_id]
        wp.atomic_add(waypoint_activations, tid, f)
        return
    
    # find the two neighboring waypoints of this waypoint
    left_pid = int(particle_indices[tid-1])
    right_pid = int(particle_indices[tid+1])
    # find the uniform vector in both directions
    left_vec = particle_q[left_pid] - particle_q[this_pid]
    right_vec = particle_q[right_pid] - particle_q[this_pid]
    if wp.length(left_vec) < 1e-4 or wp.length(right_vec) < 1e-4:
        success[0] = 0
        return
    left_dir = wp.normalize(left_vec)
    right_dir = wp.normalize(right_vec)

    # consider the contact point is a pulley
    tendon_direction = left_dir + right_dir
    if wp.length(tendon_direction) < 1e-5:
        return

    proj_tendon = wp.dot(tendon_direction, surface_normals[tid]) * surface_normals[tid]
    tendon_f_dir = tendon_direction - proj_tendon
    
    threshold = 1e-4
    blend_factor = wp.smoothstep(0.0, threshold, wp.length(tendon_f_dir))
    f_dir = blend_factor * wp.normalize(tendon_f_dir) + (1.0 - blend_factor) * wp.vec3(0.0, 0.0, 0.0)

    f = np.dot(external_force[finger_id] * tendon_direction, f_dir) * f_dir
    wp.atomic_add(waypoint_activations, tid, f)

@wp.kernel
def compute_diff_length(
    particle_indices: wp.array(dtype=wp.int32),
    particle_num: wp.int32,
    finger_waypoint_num: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    tendon_length: wp.array(dtype=wp.float32)):
    
    tid = wp.tid()

    if tid >= (particle_num-1): return
    this_pid = particle_indices[tid]
    pair_pid = particle_indices[tid+1]
    this_finger_id = finger_waypoint_num[tid]
    pair_finger_id = finger_waypoint_num[tid+1]
    if this_finger_id != pair_finger_id:
        return

    diff = particle_q[pair_pid] - particle_q[this_pid]
    wp.atomic_add(tendon_length, this_finger_id, wp.length(diff))

@wp.kernel
def calculate_tendon(
    tendon_length: wp.array(dtype=wp.float32),
    init_tendon_length: wp.array(dtype=wp.float32),
    tendon_position: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if init_tendon_length[tid] < 0:
        init_tendon_length[tid] = tendon_length[tid]
    # tendon_position[tid] = init_tendon_length[tid] - tendon_length[tid]
    pos = init_tendon_length[tid] - tendon_length[tid]
    wp.atomic_add(tendon_position, tid, pos)

@wp.kernel
def force_pid(
    p: float,
    target_pos: wp.array(dtype=wp.vec2),
    curr_pos: wp.array(dtype=wp.float32),
    frame: wp.int32,
    force: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    max_force = 100.0

    f = 1e5 * (target_pos[frame][tid] - curr_pos[tid])
    force[tid] = f
    if force[tid] > max_force:
        force[tid] = max_force
    if force[tid] < 0.0:
        force[tid] = 0.0

@wp.kernel
def find_glue_index(
    this_particle: wp.vec3,
    glue_points: wp.array(dtype=wp.vec3),
    index: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if wp.length(this_particle - glue_points[tid]) < 1e-6:
        wp.atomic_add(index, 0, tid + 1)

@wp.kernel
def target_vel(
    frame_id: wp.int32,
    target_vels: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    target_vels[tid] = wp.vec3(0.0, 0.0, 0.0)

    move_frame_id = 500
    offset_frame = 150
    if frame_id > (move_frame_id-offset_frame)  and frame_id < move_frame_id:
        target_vels[tid] = wp.vec3(0.0, 3e-4, 0.0)
    elif frame_id > move_frame_id and frame_id < (move_frame_id+offset_frame):
        target_vels[tid] = wp.vec3(0.0, -3e-4, 0.0)

class TendonHolder:
    def __init__(self, model, control):
        self.model = model
        self.control = control

        # tendon related variables
        self.finger_len = None
        self.finger_num = None
        self.waypoints = None
        self.finger_waypoint_num = None
        self.waypoint_pair_ids = None
        self.tendon_tri_indices = None
        self.external_force = None
        
        # tendon variables to be recalcuated
        self.surface_normals = None
        self.tendon_length = None

        # control related variables
        self.init_tendon_length = None
        self.tendon_position = None
    
    def init_tendon_variables(self, requires_grad=False):
        self.tendon_length = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)
        self.init_tendon_length = wp.from_numpy(np.zeros(self.finger_num)-1, dtype=wp.float32, requires_grad=requires_grad)
        self.tendon_position = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)

        self.control.force = wp.zeros(self.finger_num, dtype=wp.float32, requires_grad=requires_grad)

    def apply_force(self, external_force, particle_q, success_flag):
        wp.launch(calculate_surface_normals,
            dim = len(self.waypoints),
            inputs=[
                len(self.waypoints),
                self.tendon_tri_indices,
                particle_q,
                self.model.tri_indices,
                ],
            outputs=[self.surface_normals])
        wp.launch(calculate_force,
            dim = len(self.waypoints),
            inputs=[
                self.waypoints,
                len(self.waypoints),
                self.finger_len,
                self.waypoint_pair_ids,
                particle_q,
                self.surface_normals,
                self.finger_waypoint_num,
                external_force],
            outputs=[self.control.waypoint_forces,
                     success_flag])

    def get_tendon_length(self, particle_q):
        wp.launch(compute_diff_length,
            dim = len(self.waypoints),
            inputs=[
                self.waypoints,
                len(self.waypoints),
                self.finger_waypoint_num,
                particle_q,
                ],
            outputs=[self.tendon_length])
        wp.launch(calculate_tendon,
            dim = self.finger_num,
            inputs=[
                self.tendon_length,
                ],
            outputs=[
                self.init_tendon_length, 
                self.tendon_position])
        return self.tendon_position

    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")
    
    def reset(self):
        self.control.waypoint_forces.zero_()
        if self.control.vel_values is not None:
            self.control.vel_values.zero_()
        self.surface_normals.zero_()
        self.tendon_length.zero_()
        self.tendon_position.zero_()

class TendonControl(Control):
    def __init__(self, model):
        super().__init__(model)
        self.finger_num = None
        self.pid = {'p': 5e-1, 'i': 0.0, 'd': 0.0}
        self.waypoint_ids = None
        self.waypoint_forces = None
        self.force = None
        self.target_positions = None
        # self.vel_ids = None
        self.vel_values = None
    
    def force_from_position(self, curr_pos, target_pos, frame):
        pid_p = self.pid['p']
        pid_i = self.pid['i']
        pid_d = self.pid['d']
        wp.launch(force_pid,
            dim = self.finger_num,
            inputs=[
                pid_p,
                target_pos,
                curr_pos,
                frame
                ],
            outputs=[self.force])
        return self.force
    
    def update_target_vel(self, time):
        wp.launch(target_vel,
                dim=1,
                inputs=[
                    time],
                outputs=[self.vel_values])

    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")
    
    def reset(self):
        super().reset()
        self.waypoint_forces.zero_()
        if self.vel_values is not None:
            self.vel_values.zero_()

class TendonModel(Model):
    def __init__(self, device):
        super().__init__(device)
    
    def print_requires_grad(self):
        for name, value in vars(self).items():
            if isinstance(value, wp.array):  # Check if it is a Warp array
                print(f"{name}: requires_grad={value.requires_grad}")
            elif isinstance(value, list):  # Check if it's a list of Warp arrays
                for i, v in enumerate(value):
                    if isinstance(v, wp.array):
                        print(f"{name}[{i}]: requires_grad={v.requires_grad}")

    def control(self, requires_grad=None, clone_variables=True) -> TendonControl:
        tendon_control = TendonControl(self)
        c = super().control(requires_grad, clone_variables)
        tendon_control.__dict__.update(c.__dict__)
        return tendon_control

class TendonModelBuilder(ModelBuilder):
    # Default triangle soft mesh settings
    default_tri_ke = 100.0
    default_tri_ka = 100.0
    default_tri_kd = 10.0
    default_tri_drag = 0.0
    default_tri_lift = 0.0

    def add_soft_grid_glue(
        self,
        pos: Vec3,
        rot: Quat,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        k_mu: float,
        k_lambda: float,
        k_damp: float,
        glue_points: wp.array(dtype=wp.vec3) = [],
        tri_ke: float = default_tri_ke,
        tri_ka: float = default_tri_ka,
        tri_kd: float = default_tri_kd,
        tri_drag: float = default_tri_drag,
        tri_lift: float = default_tri_lift,
    ):
        """Helper to create a rectangular tetrahedral FEM grid
        Args:
            glue_points: List of points to which the particles are glued
        """

        start_vertex = len(self.particle_q)
        sub_index = np.zeros((dim_x + 1) * (dim_y + 1) * (dim_z + 1), dtype=np.int32)
        new_count = 0

        mass = cell_x * cell_y * cell_z * density

        def grid_index(x, y, z):
            return (dim_x + 1) * (dim_y + 1) * z + (dim_x + 1) * y + x

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    v = wp.vec3(x * cell_x, y * cell_y, z * cell_z)
                    m = mass

                    p = wp.quat_rotate(rot, v) + pos

                    this_index = wp.from_numpy(np.zeros(1)-1, dtype=wp.int32)
                    wp.launch(find_glue_index,
                        dim = len(glue_points),
                        inputs=[
                            p,
                            glue_points,
                            ],
                        outputs=[this_index])
                    this_index = int(this_index.numpy()[0])
                    if this_index != -1:
                        # glue this particle
                        sub_index[grid_index(x, y, z)] = this_index
                    else:
                        # add this new particle
                        sub_index[grid_index(x, y, z)] = new_count + start_vertex
                        self.add_particle(p, vel, m)
                        new_count += 1

        # dict of open faces
        faces = {}

        def add_face(i: int, j: int, k: int):
            key = tuple(sorted((i, j, k)))

            if key not in faces:
                faces[key] = (i, j, k)
            else:
                del faces[key]

        def add_tet(i: int, j: int, k: int, l: int):
            self.add_tetrahedron(i, j, k, l, k_mu, k_lambda, k_damp)

            add_face(i, k, j)
            add_face(j, k, l)
            add_face(i, j, l)
            add_face(i, l, k)

        for z in range(dim_z):
            for y in range(dim_y):
                for x in range(dim_x):
                    v0 = sub_index[grid_index(x, y, z)]
                    v1 = sub_index[grid_index(x + 1, y, z)]
                    v2 = sub_index[grid_index(x + 1, y, z + 1)]
                    v3 = sub_index[grid_index(x, y, z + 1)]
                    v4 = sub_index[grid_index(x, y + 1, z)]
                    v5 = sub_index[grid_index(x + 1, y + 1, z)]
                    v6 = sub_index[grid_index(x + 1, y + 1, z + 1)]
                    v7 = sub_index[grid_index(x, y + 1, z + 1)]

                    if (x & 1) ^ (y & 1) ^ (z & 1):
                        add_tet(v0, v1, v4, v3)
                        add_tet(v2, v3, v6, v1)
                        add_tet(v5, v4, v1, v6)
                        add_tet(v7, v6, v3, v4)
                        add_tet(v4, v1, v6, v3)

                    else:
                        add_tet(v1, v2, v5, v0)
                        add_tet(v3, v0, v7, v2)
                        add_tet(v4, v7, v0, v5)
                        add_tet(v6, v5, v2, v7)
                        add_tet(v5, v2, v7, v0)

        # add triangles
        for _k, v in faces.items():
            self.add_triangle(v[0], v[1], v[2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)
    
    def init_builder_tendon_variables(self, finger_num, finger_len, scale, requires_grad):
        self.finger_num = finger_num
        self.finger_len = finger_len
        self.scale = scale
        self.requires_grad = requires_grad
        self.waypoints = [[] for _ in range(finger_num)]
        self.waypoint_ids = [[] for _ in range(finger_num)]
        self.waypoint_pair_ids = [[] for _ in range(finger_num)]
        self.waypoints_tri_indices = [[] for _ in range(finger_num)]

    def build_fem_model(self, 
                        finger_width=0.06, finger_rot=0.3,
                        h_dis_init=0.2,
                        obj_loader=None, 
                        finger_transform=None,
                        is_triangle=False):
        s = self.scale
        self.finger_transform = finger_transform

        cell_dim = [2, [1, 6], 2]
        cell_size = [0.008 * s, 
                     [0.003 * s, 0.002 * s], 
                     0.01 * s]
        conn_dim = [1, 1, cell_dim[2]]
        conn_size = 0.008 * s

        finger_THK = cell_dim[1][1] * cell_size[1][1]
        finger_LEN = (cell_dim[0] * cell_size[0] * (self.finger_len // 2 + 1) + conn_dim[0] * conn_size * (self.finger_len // 2)) / s
        self.finger_back_ids = []
        self.tet_block_ids = []
        self.block_id = 0
        finger_transforms = []

        finger_height = cell_dim[2]*cell_size[2]

        h_dis = 0.0
        pos_offset = np.array([0.0, h_dis, -finger_height/2]) 
        rot = wp.quat_rpy(0.0, 0.0, -math.pi/2 - finger_rot) #top
        transform = wp.transform(pos_offset, rot)
        if finger_transform:
            transform = finger_transform[0]
        self.add_finger(
            cell_dim, cell_size, conn_dim, conn_size, 
            transform=transform,
            index=0,
            is_triangle=is_triangle)
        finger_transforms.append(transform)
        
        pos_offset = np.array([finger_width*s/2, h_dis, finger_height/2]) # top
        pos_offset = np.array([0.0, h_dis, finger_height/2])
        rot = wp.quat_rpy(math.pi, 0.0, -math.pi/2 + finger_rot) #top
        transform = wp.transform(pos_offset, rot)
        if finger_transform:
            transform = finger_transform[1]
        self.add_finger(
            cell_dim, cell_size, conn_dim, conn_size, 
            transform=transform, 
            index=1,
            is_triangle=is_triangle)
        finger_transforms.append(transform)
        
        self.model = self.finalize(requires_grad=self.requires_grad)

        radii = wp.zeros(self.model.particle_count, dtype=wp.float32)
        radii.fill_(1e-3 * self.scale)
        self.model.particle_radius = radii
        self.model.ground = True
        self.model.gravity = wp.vec3(0.0, -9.8, 0.0)
        self.model.particle_kf = 1.0e1
        self.model.particle_mu = 1.0
        self.model.particle_max_velocity = 1.0e1
        self.model.particle_adhesion = 1.0e-3
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_kf = 1.0e1
        self.model.soft_contact_mu = 1.0
        self.model.soft_contact_margin = 1e-3
        self.model.rigid_contact_margin = 1e-4
        # self.model.enable_tri_collisions = True # self-collisions
        
        return finger_transforms, finger_LEN, finger_THK
 
    def add_finger(self, 
                   cell_dim, cell_size, conn_dim, conn_size, 
                   transform=None,
                   index=0,
                   is_triangle=False):
        density = 5.0

        # actual
        K = 2.0e6 # young's modulus
        # K = np.exp(13.902607917785645)
        v = 0.45 # poisson's ratio
        k_mu = K / (2 * (1 + v))
        k_lambda = K * v / ((1 + v) * (1 - 2 * v))
        self.init_K = K
        self.init_v = v
        k_damp = 5e-1

        self.generate_tendon_waypoints_hirose(
            cell_dim, cell_size, conn_dim, conn_size, 
            index=index) 

        particle_start_idx = len(self.particle_q)
        block_start_idx = len(self.tet_indices)

        for i in range(self.finger_len // 2 + 1):
            base_offset = i*(cell_size[0]*cell_dim[0]
                               + conn_size*conn_dim[0])

            dim_x = cell_dim[0] 
            dim_z = cell_dim[2]
            cell_x = cell_size[0]
            cell_z = cell_size[2]

            waypt0 = self.waypoints[index][2*i][1]
            waypt1 = self.waypoints[index][2*i-1][1]
            if i == 0: waypt0 = self.waypoints[index][0][1]
            y_offset = [0.0] 
            y_offset.append(cell_dim[1][0] * cell_size[1][0])
            if (waypt0 - y_offset[-1]) > 1e-3*self.scale and (cell_dim[1][1] * cell_size[1][1] - waypt0) > 1e-3*self.scale:
                y_offset.append(waypt0)
            if (waypt1 - y_offset[-1]) > 1e-3*self.scale and (cell_dim[1][1] * cell_size[1][1] - waypt1) > 1e-3*self.scale:
                y_offset.append(waypt1)
            y_offset.append(cell_dim[1][1] * cell_size[1][1])
 
            for y_idx in range(len(y_offset)-1):
                dim_y = 1
                cell_y = y_offset[y_idx+1] - y_offset[y_idx]
                this_add_func = self.add_soft_grid if y_idx == 0 else self.add_soft_grid_glue
                params = {"pos": wp.vec3(base_offset, y_offset[y_idx], 0.0),
                        "rot": wp.quat_identity(),
                        "vel": wp.vec3(0.0, 0.0, 0.0),
                        "dim_x": dim_x,
                        "dim_y": dim_y,
                        "dim_z": dim_z,
                        "cell_x": cell_x,
                        "cell_y": cell_y,
                        "cell_z": cell_z,
                        "density": density,
                        "k_mu": k_mu if y_idx < 2 else k_mu/10.0,
                        "k_lambda": k_lambda if y_idx < 2 else k_lambda/10.0,
                        "k_damp": k_damp if y_idx < 2 else k_damp*10.0,
                        "tri_ke": 1e-1,
                        "tri_ka": 1e1,
                        "tri_kd": 1e-1 if y_idx < 2 else 1e-1*10.0}
                if y_idx == 0:
                    params["fix_left"] = True if i == 0 else False
                else:
                    params["glue_points"] = wp.array(self.particle_q, dtype=wp.vec3, requires_grad=self.requires_grad)
                
                this_add_func(**params)

            self.tet_block_ids.extend([self.block_id for _ in range(len(self.tet_indices) - block_start_idx)])
            block_start_idx = len(self.tet_indices)
            self.block_id += 1
        
        for i in range(self.finger_len // 2):  
            base_offset = i*(cell_size[0]*cell_dim[0]
                               + conn_size*conn_dim[0])

            dim_x = conn_dim[0]
            dim_y = conn_dim[1]
            dim_z = cell_dim[2]
            cell_x = conn_size
            cell_y = cell_size[1][0]
            cell_z = cell_size[2]
            self.add_soft_grid_glue(
                pos=wp.vec3(base_offset + cell_size[0]*cell_dim[0], 0.0, 0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell_x,
                cell_y=cell_y,
                cell_z=cell_z,
                density=density,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
                glue_points=wp.array(self.particle_q, dtype=wp.vec3),
                tri_ke=1e-1,
                tri_ka=1e1,
                tri_kd=1e-1
            )

            self.tet_block_ids.extend([self.block_id for _ in range(len(self.tet_indices) - block_start_idx)])
            block_start_idx = len(self.tet_indices)
            self.block_id += 1
        particle_end_idx = len(self.particle_q)
        self.find_back_ids(particle_start_idx, 
                           particle_end_idx,
                           low_threshold=np.array(
                               [1e-2, cell_dim[1][1]*cell_size[1][1], -1e-6]),
                            high_threshold=np.array(
                                [1e3, 1e3, 1e3]))

        self.find_waypoints_tri_indices(index)

        if is_triangle:
            # reform sharp finger
            Lx = ((cell_dim[0] * cell_size[0]) * (self.finger_len // 2 + 1) + (conn_dim[0] * conn_size) * (self.finger_len // 2))
            Ly = cell_dim[1][1] * cell_size[1][1] 
            x0 = cell_dim[0] * cell_size[0]
            beta = Ly - 0.035
            
            self.particle_q, self.tet_poses, self.tri_poses, self.tri_areas = utils.sharp_points(self.particle_q,
                                self.tet_indices,
                                self.tet_poses,
                                self.tri_indices,
                                self.tri_poses,
                                self.tri_areas,
                                Lx,
                                Ly,
                                x0,
                                beta,
                                particle_start_idx,
                                particle_end_idx, 
                                is_triangle)

        # apply transforms
        self.particle_q = utils.transform_points(
            self.particle_q,
            transform,
            particle_start_idx, particle_end_idx)
        self.waypoints[index] = utils.transform_points(
            np.array(self.waypoints[index]),
            transform,
            0, len(self.waypoints[index]))
    
    def find_back_ids(self, 
                      particle_start_idx, particle_end_idx, 
                      low_threshold=np.zeros(3),
                      high_threshold=np.zeros(3)+1e5):
        for i in range(particle_start_idx, particle_end_idx):
            flag = True
            for index in range(3):
                if self.particle_q[i][index] < low_threshold[index]:
                    flag = False
                if self.particle_q[i][index] > high_threshold[index]:
                    flag = False
            if not flag: continue
            if i not in self.finger_back_ids:
                self.finger_back_ids.append(i)

    def find_waypoints_tri_indices(self, n):
        waypoint_ids = []
        waypoints_tri_indices = []
        for i in range(len(self.waypoints[n])):
            this_wp = self.waypoints[n][i]
            c_idx, face_idx = self.find_triangle_idx(
                this_wp,
                wp.array(self.particle_q).numpy(),
                wp.array(self.tri_indices).numpy())
            waypoint_ids.append(c_idx)
            waypoints_tri_indices.append(face_idx)
        self.waypoint_ids[n] = waypoint_ids
        self.waypoints_tri_indices[n] = waypoints_tri_indices

    def generate_tendon_waypoints_hirose(
            self, cell_dim, cell_size, 
            conn_dim, conn_size, 
            index=0):
        waypoints = []
        waypoint_pair_ids = []
        
        Lx = ((cell_dim[0]*cell_size[0])*(self.finger_len//2 +1) +
            (conn_dim[0]*conn_size)*(self.finger_len//2))
        Ly = cell_dim[1][1]*cell_size[1][1] 
        z_perct = 0.5* np.ones(self.finger_len)
        base_offset = cell_size[0]*cell_dim[0]

        def get_y(x):
            y = Ly * (1 - x/Lx)**2
            y += cell_size[1][0]*conn_dim[1] + 2e-3
            return y

        for i in range(self.finger_len):
            pos = [0, 0, 0]
            if (i % 2) == 0:
                # finger body block
                pos = [base_offset, 
                        get_y(base_offset), 
                        z_perct[i]*(cell_size[2]*cell_dim[2])]
                base_offset += conn_size*conn_dim[0]
                waypoint_pair_ids.append(i+1 if i < self.finger_len-1 else i)
            else:
                # connector block
                pos = [base_offset,
                        get_y(base_offset),
                        z_perct[i]*(cell_size[2]*cell_dim[2])]
                base_offset += cell_size[0]*cell_dim[0]
                waypoint_pair_ids.append(i-1)
            waypoints.append(pos) 
        self.waypoint_pair_ids[index] = waypoint_pair_ids
        
        waypoints = np.array(waypoints)
        self.waypoints[index] = waypoints.tolist()

    def find_triangle_idx(self, waypt, points, 
                          faces=None,
                          start_idx=0, end_idx=0):
        if end_idx <= start_idx:
            end_idx = points.shape[0]
        force_idx = []
        curr_dis = np.linalg.norm(points[0, :] - waypt)
        for i in range(start_idx, end_idx):
            dis = np.linalg.norm(points[i, :] - waypt)
            if dis < curr_dis:
                curr_dis = dis
                force_idx.append(i)

        candidate = None
        max_area = 0.0
        for fid in range(len(force_idx)):
            this_force_idx = force_idx[len(force_idx)-1-fid]
            for i in range(faces.shape[0]):
                if this_force_idx in faces[i, :]:
                    # return this_force_idx, i
                    # find the face that is verticle
                    face_pts = points[faces[i, :], :]
                    mid_pt = np.mean(face_pts, axis=0)
                    if np.abs(mid_pt[0] - waypt[0]) < 1e-6:
                        # return this_force_idx, i
                        this_area = np.linalg.norm(np.cross(face_pts[1, :] - face_pts[0, :], face_pts[2, :] - face_pts[0, :]))
                        if this_area > max_area:
                            max_area = this_area
                            candidate = (this_force_idx, i)
        if candidate: return candidate
        assert False, "No verticle triangle found"

    def generate_fixpoints(self, cell_dim, cell_size, 
                           curr_points, index=0,
                           start_idx=0, end_idx=0):
        offset = []
        for y_dim in cell_dim[1]:
            y_base = y_dim * cell_size[1][0]
            for yi in range(y_dim):
                for zi in range(cell_dim[2]):
                    offset.append([0.0, y_base + yi*cell_size[1][1], zi*cell_size[2]])

        self.fixpoints[index] = offset
        for i in range(len(offset)):
            c_idx, _ = self.find_triangle_idx(
                offset[i],
                np.array(curr_points),
                start_idx=start_idx, end_idx=end_idx)
            self.fixpoints_ids[index].append(c_idx)


    def finalize(self, device=None, requires_grad=False) -> TendonModel:
        tendon_model = TendonModel(device)
        m = super().finalize(device=device, requires_grad=requires_grad)
        tendon_model.__dict__.update(m.__dict__)
        return tendon_model
    
class TendonRenderer(wp.sim.render.SimRenderer):
    def render(self, state, highlight_pt_ids=[], force_arr=[], force_scale=0.1, additional_pts=None):
        # super().render(state)
        if self.skip_rendering:
            return

        if self.model.particle_count:
            particle_q = state.particle_q.numpy()

            # render particles
            self.render_points(
                "particles", particle_q, 
                radius=self.model.particle_radius.numpy(), 
                colors=(0.8, 0.3, 0.2)
            )
            if len(highlight_pt_ids) > 0:
                self.render_points(
                    "force_particles", particle_q[highlight_pt_ids, :],
                    radius=0.01,
                    colors=(1, 0, 0)
                )

            # render tris
            if self.model.tri_count:
                self.render_mesh(
                    "surface",
                    particle_q,
                    self.model.tri_indices.numpy().flatten(),
                    colors=(((0.75, 0.25, 0.0),) * len(particle_q)),
                )

            # render springs
            if self.model.spring_count:
                self.render_line_list(
                    "springs", particle_q, self.model.spring_indices.numpy().flatten(), (0.25, 0.5, 0.25), 0.02
                )
            
            if len(force_arr) > 0:
                assert len(force_arr) == len(highlight_pt_ids)
                this_force_arr = force_arr.numpy()
                for i in range(len(highlight_pt_ids)):
                    idx = highlight_pt_ids[i]
                    f_arr = this_force_arr[i] * force_scale
                    f_end = particle_q[idx] + f_arr
                    points = [particle_q[idx], f_end]
                    # print("force point:", points)
                    self.render_line_strip(
                        name=f"force_{i}", vertices=points, color=(0.25, 0.5, 0.25), radius=0.005)
            
        if additional_pts is not None:
            for i in range(len(additional_pts)):
                additional_pt = additional_pts[i].numpy()
                self.render_points(
                    f"additional_pt{i}", additional_pt, 
                    radius=0.005, colors=(0.2*i, 0.0, 1.0)
                )

        # update bodies
        if self.model.body_count:
            self.update_body_transforms(state.body_q)