import numpy as np
import warp as wp
from warp.sim.collide import mesh_sdf
from warp.sim.model import ModelShapeGeometry, State
from scipy.spatial.transform import Rotation   

def get_sharp_y(x, Lx, Ly, beta, x0):
    # Lx, Ly, beta, x0
    return (Ly - beta) * Lx / (Lx - x0) * (1 - x / Lx) + beta
    
@wp.kernel
def wp_sharp_points(particle_q:wp.array(dtype=wp.vec3),
                    Lx: wp.float32,
                    Ly: wp.float32,
                    x0: wp.float32,
                    beta: wp.float32,
                    start_idx: wp.int32,
                    end_idx: wp.int32):
    tid = wp.tid()
    if tid < start_idx or tid >= end_idx: return
    
    if wp.abs(particle_q[tid][1] - Ly) > 1e-6: return
    x = particle_q[tid][0]
    new_y = (Ly - beta) * Lx / (Lx - x0) * (1.0 - x / Lx) + beta
    if new_y <= particle_q[tid][1]:
        particle_q[tid][1] = new_y


@wp.kernel
def wp_modify_tet(particle_q: wp.array(dtype=wp.vec3),
                  wp_tet_indices: wp.array2d(dtype=int),
                  wp_tet_poses: wp.array(dtype=wp.mat33),
                  ):
    tid = wp.tid()
    i, j, k, l = wp_tet_indices[tid, 0], wp_tet_indices[tid, 1], wp_tet_indices[tid, 2], wp_tet_indices[tid, 3]
    p, q, r, s = particle_q[i], particle_q[j], particle_q[k], particle_q[l]
    
    qp = q - p
    rp = r - p
    sp = s - p
    
    # Dm = wp.transpose(wp.mat33(qp, rp, sp))
    Dm = wp.mat33(qp, rp, sp)
    volume = wp.determinant(Dm) / 6.0
    
    if volume <= 0.0:
        wp.printf("inverted tetrahedral element %d %d %d %d\n", i, j, k, l)
    else:
        inv_Dm = wp.inverse(Dm)
        wp_tet_poses[tid] = inv_Dm

@wp.kernel
def wp_modify_tri(particle_q: wp.array(dtype=wp.vec3),
                  wp_tri_indices: wp.array2d(dtype=int),
                  wp_tri_poses: wp.array(dtype=wp.mat22),
                  wp_tri_areas: wp.array(dtype=wp.float32),
                  ):
    tid = wp.tid()
    i, j, k = wp_tri_indices[tid, 0], wp_tri_indices[tid, 1], wp_tri_indices[tid, 2]
    p, q, r = particle_q[i], particle_q[j], particle_q[k]
    
    qp = q - p
    rp = r - p
    
    n = wp.normalize(wp.cross(qp, rp))
    e1 = wp.normalize(qp)
    e2 = wp.normalize(wp.cross(n, e1))
    
    D = wp.mat22(wp.dot(e1, qp), wp.dot(e1, rp), wp.dot(e2, qp), wp.dot(e2, rp))
    
    area = wp.determinant(D) / 2.0
    if area <= 0.0:
        wp.print("inverted or degenerate triangle element")
    else:
        # wp.print("Storing poses and areas...")
        inv_D = wp.inverse(D)
        wp_tri_poses[tid] = inv_D
        wp_tri_areas[tid] = area
 
def sharp_points(points,
                tet_indices,
                tet_poses,
                tri_indices,
                tri_poses,
                tri_areas,
                Lx,
                Ly,
                x0,
                beta,
                particle_start_idx,
                particle_end_idx,
                flag,
                ):
    tet_num = len(tet_indices)
    tri_num = len(tri_indices)
    # breakpoint()
    wp_particle_q = wp.array(points, dtype=wp.vec3)
    wp_tet_indices = wp.array2d(tet_indices, dtype=int)
    wp_tet_poses = wp.array(tet_poses, dtype=wp.mat33)
    wp_tri_indices = wp.array2d(tri_indices, dtype=int)
    wp_tri_poses = wp.array(tri_poses, dtype=wp.mat22)
    wp_tri_areas = wp.array(tri_areas, dtype=wp.float32)
    # wp_tet_poses_np_before = wp_tet_poses.numpy()
   
    wp.launch(wp_sharp_points,
        dim = len(points),
        inputs=[
            wp_particle_q,
            Lx,
            Ly,
            x0,
            beta,
            particle_start_idx,
            particle_end_idx,
            ])
    if flag:
        wp.launch(wp_modify_tet,
            dim = tet_num,
            inputs=[
                wp_particle_q,
                wp_tet_indices,
                wp_tet_poses,
                ])

        wp.launch(wp_modify_tri,
            dim = tri_num,
            inputs=[
                wp_particle_q,
                wp_tri_indices,
                wp_tri_poses,
                wp_tri_areas,
                ])
    
    wp_particle_q = wp_particle_q.numpy().tolist()
    wp_tet_poses = wp_tet_poses.numpy().tolist()
    wp_tri_poses = wp_tri_poses.numpy().tolist()
    wp_tri_areas = wp_tri_areas.numpy().tolist()
    
    return wp_particle_q, wp_tet_poses, wp_tri_poses, wp_tri_areas

def reset_tet(particle_q: wp.array, model):
    tet_num = len(model.tet_indices)
    tri_num = len(model.tri_indices)

    wp.launch(wp_modify_tet,
        dim = tet_num,
        inputs=[
            particle_q,
            model.tet_indices,
            model.tet_poses,
            ])
    wp.launch(wp_modify_tri,
        dim = tri_num,
        inputs=[
            particle_q,
            model.tri_indices,
            model.tri_poses,
            model.tri_areas,
            ])
    
@wp.func
def soft_normalize(v: wp.vec3,
                   eps: wp.float32 = 1e-6):
    return v / (wp.length(v) + eps)

@wp.func
def softplus(x: float, beta: float = 10.0):
    """Smooth approximation of max(0, x)"""
    return 1.0 / beta * wp.log(1.0 + wp.exp(beta * x))

@wp.func
def smooth_zero(vt: wp.vec3, epsilon: float):
    length = wp.length(vt)
    weight = length / wp.sqrt(length * length + epsilon * epsilon)
    return weight * vt

@wp.func
def norm_huber(v: wp.vec3, delta: float = 1.0):
    a = wp.dot(v, v)
    if a <= delta * delta:
        return 0.5 * a
    return delta * (wp.sqrt(a) - 0.5 * delta)

@wp.func
def check_tet_degenerate(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):

    qp = v1 - v0
    rp = v2 - v0
    sp = v3 - v0
    
    # Dm = wp.transpose(wp.mat33(qp, rp, sp))
    Dm = wp.mat33(qp, rp, sp)
    volume = wp.determinant(Dm) / 6.0
    
    if volume <= 0.0:
        return True
    flag = wp.abs(wp.dot(v0 - v3, wp.cross(v1 - v3, v2 - v3))) < 1e-6
    return flag


@wp.func
def compute_centroid(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3):
    centroid = v0 + v1 + v2 + v3
    return centroid / 4.0

@wp.func
def get_tet_quat(t00: wp. vec3, t01: wp.vec3, t02: wp.vec3, t03: wp.vec3,
                        t10: wp.vec3, t11: wp.vec3, t12: wp.vec3, t13: wp.vec3):
    c0 = compute_centroid(t00, t01, t02, t03)
    c1 = compute_centroid(t10, t11, t12, t13)

    t = c1 - c0 # translation

    # Center the vertices
    # Q_start = [t0[i] - c0 for i in range(4)]
    # Q_end = [t1[i] - c1 for i in range(4)]

    # Compute the covariance matrix H
    H = wp.mat33(0.0)
    H += wp.outer(t00 - c0, t10 - c1)
    H += wp.outer(t01 - c0, t11 - c1)
    H += wp.outer(t02 - c0, t12 - c1)
    H += wp.outer(t03 - c0, t13 - c1)

    # Perform SVD decomposition of H
    U = wp.mat33(0.0)
    sigma = wp.vec3(0.0)
    Vt = wp.mat33(0.0)
    wp.svd3(H, U, sigma, Vt)

    # Compute the rotation matrix
    R = wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            R[i, j] = 0.0
            for k in range(3):
                R[i, j] += Vt[i, k] * U[j, k]

    # Correct the rotation if det(R) < 0
    if wp.determinant(R) < 0.0:
        for i in range(3):
            Vt[2, i] *= -1.0
        for i in range(3):
            for j in range(3):
                R[i, j] = 0.0
                for k in range(3):
                    R[i, j] += Vt[i, k] * U[j, k]

    # return wp.transform(t, wp.quat_from_matrix(R))
    return wp.quat_from_matrix(R)

@wp.func
def element_min(v1: float, v2: wp.vec3):
    return wp.vec3(
        wp.min(v1, v2.x),
        wp.min(v1, v2.y),
        wp.min(v1, v2.z))

@wp.kernel
def wp_transform_points(
    points: wp.array(dtype=wp.vec3),
    start_index: wp.int32,
    end_index: wp.int32,
    transform: wp.transform):
    tid = wp.tid()
    if tid < start_index or tid >= end_index: return
    points[tid] = wp.transform_point(transform, points[tid])

def transform_points(points, transform, start_index, end_index):
    wp_points = wp.array(points, dtype=wp.vec3)
    wp.launch(wp_transform_points,
        dim = len(points),
        inputs=[
            wp_points,
            # len(points),
            start_index,
            end_index,
            transform,
            ])
    return wp_points.numpy().tolist()


def remove_nan(torch_var, clip_th=1e-1, clip=False):
    torch_var[torch_var.isnan()] = 0
    if clip:
        torch_var[torch_var > clip_th] = clip_th
        torch_var[torch_var < -clip_th] = -clip_th

@wp.kernel
def transform_points_out(
    points: wp.array(dtype=wp.vec3),
    trans_id: wp.int32,
    trans: wp.array(dtype=wp.transform),
    out_points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    # transform = wp.transform(wp.vec3(trans[0], trans[1], trans[2]),
    #                          wp.quaternion(trans[3], trans[4], trans[5], trans[6]))
    out_points[tid] = wp.transform_point(trans[trans_id], points[tid])


@wp.kernel
def multi_transform_points_out(
    points: wp.array(dtype=wp.vec3),
    trans_id: wp.int32,
    trans: wp.array(dtype=wp.transform),
    out_points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    # transform = wp.transform(wp.vec3(trans[0], trans[1], trans[2]),
    #                          wp.quaternion(trans[3], trans[4], trans[5], trans[6]))
    out_points[tid] = wp.transform_point(trans[trans_id], points[tid])

@wp.kernel
def transform_initial_points(
    points: wp.array(dtype=wp.vec3),
    points_curr: wp.array(dtype=wp.vec3),
    trans: wp.transform):
    
    tid = wp.tid()
    points[tid] = wp.transform_point(trans, points[tid])
    points_curr[tid] = wp.transform_point(trans, points_curr[tid])

@wp.kernel
def wp_transform_points2(
    points: wp.array(dtype=wp.vec3),
    num_points: wp.int32,
    start_index: wp.int32,
    end_index: wp.int32,
    actions: wp.array(dtype=wp.float32),
    index: wp.int32):
    tid = wp.tid()
    if tid >= num_points: return
    if tid < start_index or tid >= end_index: return
    pos = wp.vec3(actions[index + 0], actions[index + 1], actions[index + 2])
    # rot = wp.quat_rpy(actions[index + 3], actions[index + 4], actions[index + 5]) 
    rot = wp.quat_rpy(0.0, 0.0, 0.0)
    transform = wp.transform(pos, rot)
    points[tid] = wp.transform_point(transform, points[tid])
    
def update_points(points, actions, index, start_index, end_index):
    wp_points = points#wp.array(points, dtype=wp.vec3)
    wp.launch(wp_transform_points2,
        dim = len(points),
        inputs=[
            wp_points,
            len(points),
            start_index,
            end_index,
            actions,
            index,
            ])
    return wp_points

@wp.kernel
def transform_arr_points(
    points: wp.array(dtype=wp.vec3),
    point_ids: wp.array(dtype=wp.int32),
    transform: wp.array(dtype=wp.transform),
    out_points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    pid = point_ids[tid]
    out_points[pid] = wp.transform_point(transform[0], points[pid])

@wp.func
def transform_to9d_func(trans: wp.transform, 
                        trans_9d: wp.array(dtype=wp.float32)):
    # translation
    T = wp.transform_get_translation(trans)
    for i in range(3):
        trans_9d[i] = T[i]

    # rotation
    quat = wp.transform_get_rotation(trans)
    R = wp.quat_to_matrix(quat)
    for i in range(6):
        trans_9d[i + 3] = R[i // 3, i % 3]


@wp.kernel
def transform_to9d(trans: wp.array(dtype=wp.float32), # 7d
                   transform_9d: wp.array(dtype=wp.float32)):
    T = wp.vec3(trans[0], trans[1], trans[2])
    quat = wp.quat(trans[3], trans[4], trans[5], trans[6])
    this_trans = wp.transform(T, quat)
    transform_to9d_func(this_trans, transform_9d)
    # # translation
    # for i in range(3):
    #     transform_9d[i] = trans[i]

    # # rotation
    # quat = wp.quaternion(trans[3], trans[4], trans[5], trans[6])
    # R = wp.quat_to_matrix(quat)
    # for i in range(6):
    #     transform_9d[i + 3] = R[i // 3, i % 3]
    
@wp.kernel
def multi_transform_to11d(trans: wp.array(dtype=wp.float32), # 9d
                          joints_per_env: int, 
                          transform_9d: wp.array(dtype=wp.float32),
                          transform_2d: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    env_offset = tid * joints_per_env
    offset_9d = tid * 9 
    offset_2d = tid * 2
    # translation
    for i in range(3):
        transform_9d[offset_9d + i] = trans[env_offset + i]

    # rotation
    quat = wp.quaternion(trans[env_offset + 3], trans[env_offset + 4], trans[env_offset + 5], trans[env_offset + 6])
    R = wp.quat_to_matrix(quat) # R_{2 x 3}
    for i in range(6):
        transform_9d[offset_9d + i + 3] = R[i // 3, i % 3]
        
    # prismatic joint
    for i in range(2):
        transform_2d[offset_2d + i] = trans[env_offset + i + 7]

@wp.kernel
def transform_from9d(transform_9d: wp.array(dtype=float),
                    trans: wp.array(dtype=float)):
    T = transform_from9d_func(transform_9d)
    for i in range(3):
        trans[i] = transform_9d[i]
    q = wp.transform_get_rotation(T)
    for i in range(4):
        trans[i + 3] = q[i]

@wp.kernel
def multi_transform_from11d(transform_9d: wp.array(dtype=float),
                            transform_2d: wp.array(dtype=float),
                            joints_per_env: int,
                            trans: wp.array(dtype=float)):
    tid = wp.tid() 
    env_offset = tid * joints_per_env
    offset_9d = tid * 9 
    offset_2d = tid * 2
    
    # get transform from 9d
    a1 = wp.vec3(transform_9d[offset_9d + 3], transform_9d[offset_9d + 4], transform_9d[offset_9d + 5])
    a2 = wp.vec3(transform_9d[offset_9d + 6], transform_9d[offset_9d + 7], transform_9d[offset_9d + 8])
    b1 = wp.normalize(a1)
    b2 = a2 - (wp.dot(b1, a2) * b1)
    b2_norm = wp.normalize(b2)
    b3 = wp.cross(b1, b2_norm)
    R_t = wp.mat33(b1, b2_norm, b3)
    R = wp.transpose(R_t)
    trans_q = wp.quat_from_matrix(R)
    T = wp.transform(
        wp.vec3(transform_9d[offset_9d + 0], transform_9d[offset_9d + 1], transform_9d[offset_9d + 2]),
        trans_q)
    
    for i in range(3):
        trans[env_offset + i] = transform_9d[offset_9d + i]
    q = wp.transform_get_rotation(T)
    for i in range(4):
        trans[env_offset + i + 3] = q[i]
    for i in range(2):
        trans[env_offset + i + 7] = transform_2d[offset_2d + i]
        
@wp.func
def transform_from9d_func(transform_9d: wp.array(dtype=float)):
    a1 = wp.vec3(transform_9d[3], transform_9d[4], transform_9d[5])
    a2 = wp.vec3(transform_9d[6], transform_9d[7], transform_9d[8])
    b1 = wp.normalize(a1)
    b2 = a2 - (wp.dot(b1, a2) * b1)
    b2_norm = wp.normalize(b2)
    b3 = wp.cross(b1, b2_norm)
    R_t = wp.mat33(b1, b2_norm, b3)
    R = wp.transpose(R_t)
    trans_q = wp.quat_from_matrix(R)
    T = wp.transform(
        wp.vec3(transform_9d[0], transform_9d[1], transform_9d[2]),
        trans_q)
    return T

@wp.kernel
def transform_matrix_from9d(transform_9d: wp.array(dtype=float),
                    trans: wp.array(dtype=wp.transform)):
    # a1 = wp.vec3(transform_9d[3], transform_9d[4], transform_9d[5])
    # a2 = wp.vec3(transform_9d[6], transform_9d[7], transform_9d[8])
    # b1 = wp.normalize(a1)
    # b2 = a2 - (wp.dot(b1, a2) * b1)
    # b2_norm = wp.normalize(b2)
    # b3 = wp.cross(b1, b2_norm)
    # R_t = wp.mat33(b1, b2_norm, b3)
    # R = wp.transpose(R_t)
    # trans_q = wp.quat_from_matrix(R)
    # T = wp.transform(wp.vec3(transform_9d[0], transform_9d[1], transform_9d[2]), trans_q)
    T = transform_from9d_func(transform_9d)
    
    trans[0] = T

@wp.kernel
def compute_spatial_vector(
    transform_a: wp.array(dtype=wp.transform),
    transform_b: wp.array(dtype=float), # 7d vector
    body_id: wp.int32,
    sim_dt: wp.float32,
    spatial_vector_arr: wp.array(dtype=wp.spatial_vector)
):
    tid = wp.tid()

    # Extract translations
    pos_a = wp.transform_get_translation(transform_a[body_id])
    pos_b = wp.vec3(transform_b[0], transform_b[1], transform_b[2])

    # Extract rotations (quaternions)
    rot_a = wp.transform_get_rotation(transform_a[body_id])
    rot_b = wp.quaternion(transform_b[3], transform_b[4], transform_b[5], transform_b[6])

    # Compute linear velocity (translation difference)
    linear_velocity = (pos_b - pos_a) / sim_dt

    # Compute angular velocity (rotation difference)
    rot_diff = wp.quat_inverse(rot_a) * rot_b
    
    # Clamp the quaternion scalar part to prevent NaN in acos
    cos_half_angle = wp.clamp(rot_diff[3], -0.999999, 0.999999)

    # Compute sin(theta/2) safely using identity: sin² + cos² = 1
    sin_half_angle = wp.sqrt(wp.max(0.0, 1.0 - cos_half_angle * cos_half_angle))

    eps = 1e-6  # Small number to avoid division by zero
    angle = 2.0 * wp.acos(cos_half_angle)  # Default case

    # Use Taylor series for small angles to maintain smooth gradients
    if sin_half_angle < eps:
        angle = 2.0 * sin_half_angle  # Approximate small angles

    # Compute rotation axis safely
    if sin_half_angle > eps:
        axis = wp.vec3(rot_diff[0], rot_diff[1], rot_diff[2]) / (sin_half_angle + eps)  # Avoid divide-by-zero
    else:
        axis = wp.vec3(rot_diff[0], rot_diff[1], rot_diff[2])  # Use raw values for small angles

    # Compute angular velocity
    angular_velocity = axis * (angle / sim_dt)
    
    # Store spatial vector: [angular_velocity, linear_velocity]
    spatial_vector_arr[body_id] = wp.spatial_vector(angular_velocity, linear_velocity)

@wp.kernel
def transform9d_multiply(trans1: wp.array(dtype=float),
                        trans2: wp.array(dtype=float)):
    T1 = transform_from9d_func(trans1)
    T2 = transform_from9d_func(trans2)
    T = wp.transform_multiply(T1, T2)
    transform_to9d_func(T, trans2)
    # T_trans = wp.transform_get_translation(T)
    # T_quat = wp.transform_get_rotation(T)
    # for i in range(3):
    #     trans2[i] = T_trans[i]
    # for i in range(4):
    #     trans2[i + 3] = T_quat[i]

@wp.kernel
def transform_from11d(transform_9d: wp.array(dtype=float),
                      transform_2d: wp.array(dtype=float),
                    trans: wp.array(dtype=float)):
    T = transform_from9d_func(transform_9d)
    for i in range(3):
        trans[i] = transform_9d[i]
    q = wp.transform_get_rotation(T)
    for i in range(4):
        trans[i + 3] = q[i]
    for i in range(2):
        trans[i + 7] = transform_2d[i]

@wp.kernel
def transform_to11d(trans: wp.array(dtype=wp.float32), # 9d
                   transform_9d: wp.array(dtype=wp.float32),
                   transform_2d: wp.array(dtype=wp.float32)):
    # translation
    for i in range(3):
        transform_9d[i] = trans[i]

    # rotation
    quat = wp.quaternion(trans[3], trans[4], trans[5], trans[6])
    R = wp.quat_to_matrix(quat) # R_{2 x 3}
    for i in range(6):
        transform_9d[i + 3] = R[i // 3, i % 3]
        
    # prismatic joint
    for i in range(2):
        transform_2d[i] = trans[i + 7]
        

def test_transform():
    t = wp.vec3(1.0, 2.0, 3.0)
    q = wp.quat_rpy(0.1, 0.2, 0.3)
    print("init quaternion:", q)
    trans = wp.array([t[0], t[1], t[2], q[0], q[1], q[2], q[3]], dtype=wp.float32)
    print("init trans:", trans)
    transform_9d = wp.zeros(9, dtype=wp.float32)
    wp.launch(transform_to9d, dim=1,
                inputs=[trans],
                outputs=[transform_9d])
    print("transform_9d:", transform_9d.numpy())

    trans_after = wp.array([0.0]*7, dtype=wp.float32)
    # trans_arr = wp.array([trans], dtype=wp.transform)
    wp.launch(transform_from9d, dim=1,
                inputs=[transform_9d],
                outputs=[trans_after])
    print("transform:", trans_after)
    
# Taken from warp example_quadruped.py
def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets

def set_state_grad(state, requires_grad):
    state.particle_q.requires_grad = requires_grad
    state.particle_qd.requires_grad = requires_grad
    state.particle_f.requires_grad = requires_grad

    state.body_q.requires_grad = requires_grad
    state.body_qd.requires_grad = requires_grad
    state.body_f.requires_grad = requires_grad

def load_object(builder, obj_loader,
                 object='ycb', 
                 ycb_object_name='006_mustard_bottle',
                 obj_rot=wp.quat_identity(),
                 scale=5.0,
                 density=1e1,
                 use_simple_mesh=True,
                 is_fix=False):
    s = scale
    object_com = None
    if object == 'box':
        obj_loader.add_box(
            builder, 
            0.02*s, 0.01*s, 0.01*s, 
            pos=np.array([0.02*s, 0.05/2*s, 0.02/2*s]),
            # rot=wp.quat_identity(),
            rot=wp.quat_rpy(0.0, np.pi/4, 0.0),
            scale=s)
    elif object == 'fix_box':
        obj_loader.add_fix_box(
            builder, 0.1*s, 0.1*s, 0.06*s, 
            pos=np.array([0.1*s, 0.0, 0.0]),
            rot=wp.quat_identity(),
            scale=s)
    elif object == 'ycb':
        object_com, body_id, geo_id = obj_loader.add_ycb(builder, 
                    # np.array([0.1, 0.0, 0.03]) *self.scale, # side finger position
                    np.array([0.0, 0.0, 0.0]) *s, # top finger position
                    # wp.quat_rpy(-np.pi/2, 0.0, 0.0),
                    obj_rot,
                    obj_name=ycb_object_name,
                    scale=s,
                    is_fix=is_fix,
                    use_simple_mesh=use_simple_mesh,
                    density=density)
    return object_com, body_id, geo_id

@wp.kernel
def finger_com(vertices: wp.array(dtype=wp.vec3),
               num_vertices: int,
            #    object_com: wp.array(dtype=wp.vec3),
               com_sum: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    # diff = (vertices[tid] - object_com[0]) / float(num_vertices)
    diff = (vertices[tid]) / float(num_vertices)
    # compute center of mass
    wp.atomic_add(com_sum, 0, diff)

@wp.kernel
def mesh_dis(geo: ModelShapeGeometry,
             object_id: wp.int32,
             object_body_id: wp.int32,
             body_q: wp.array(dtype=wp.transform),
             finger_mesh: wp.array(dtype=wp.vec3),
             collision_dis: float,
             distance_param: float,
             penetration_param: float,
             frame: int,
             total_dis: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    max_dis = 100.0

    finger_pt = finger_mesh[tid]
    object_trans = body_q[object_body_id]
    finger_local = wp.transform_point(wp.transform_inverse(object_trans), finger_pt)
    mesh_b = geo.source[object_id]
    geo_scale_b = geo.scale[object_id]

    d = mesh_sdf(mesh_b, wp.cw_div(finger_local, geo_scale_b), max_dis)
    d -= collision_dis
    
    penalty = d*d * distance_param
    # penalty = 0.0
    if d < 0.0:
        penalty = d*d * penetration_param# Quadratic force to enforce boundary
        # wp.printf("collision detected: %f, dis:%f\n", d, collision_dis)
    
    # penalize ground collision
    d = finger_pt[1] - collision_dis
    if d > 0.0:
        penalty += d*d * distance_param/1e6
    elif d < 0.0:
        # penalty += penetration_param * finger_pt[1]*finger_pt[1]
        penalty += d*d * penetration_param
    wp.atomic_add(total_dis, frame, penalty)

@wp.kernel
def joint_grip(joint_q:wp.array(dtype=float),
               add_joint_q: wp.array(dtype=float),
               limit_lower: wp.array(dtype=float),
               limit_upper: wp.array(dtype=float)):
    tid = wp.tid()
    joint_base = tid * 9 + 1
    joint_id_0 = tid * 9 + 7
    joint_id_1 = tid * 9 + 8
    if joint_q[joint_id_0] <= limit_upper[0] + 0.1:
        joint_q[joint_id_0] = joint_q[joint_id_0] + add_joint_q[joint_id_0]
    if joint_q[joint_id_1] >= limit_lower[0] - 0.1:
        joint_q[joint_id_1] = joint_q[joint_id_1] - add_joint_q[joint_id_1]
    joint_q[joint_base] += add_joint_q[joint_base]
    # if joint_q[joint_id_0] + 0.0001 <= limit_upper[0] + 0.1:
    #     joint_q[joint_id_0] = joint_q[joint_id_0] + 0.0001
    # if joint_q[joint_id_1] - 0.0001 >= limit_lower[0] - 0.1:
    #     joint_q[joint_id_1] = joint_q[joint_id_1] - 0.0001

@wp.kernel
def joint_qd_grip(joint_qd: wp.array(dtype=float)):
    tid = wp.tid()
    joint_id_0 = tid * 9 + 7
    joint_id_1 = tid * 9 + 8
    joint_qd[joint_id_0] = 1.0

@wp.kernel
def apply_joint_act(joint_act: wp.array(dtype=float)):
    tid = wp.tid()
    # joint_id_0 = tid + 0
    joint_act[0] = 1e-4
    joint_act[1] = 1e-4
        
@wp.kernel
def get_force(body_f: wp.array(dtype=wp.spatial_vector),
              force: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    body_id_0 = tid * 4 + 1
    body_id_1 = tid * 4 + 2
    # object_body_id = tid * 4
    
    # body_force = body_f[object_body_id]
    # f = wp.vec3(body_force[3], body_force[4], body_force[5])
    # force[tid * 2] = wp.length(f)
    
    body_force_gripper1 = body_f[body_id_0]
    f1 = wp.vec3(body_force_gripper1[3], body_force_gripper1[4], body_force_gripper1[5])
    body_force_gripper2 = body_f[body_id_1]
    f2 = wp.vec3(body_force_gripper2[3], body_force_gripper2[4], body_force_gripper2[5])

    force[tid] = (wp.length(f1) + wp.length(f2)) * 0.5
    
@wp.kernel
def max_force_per_env(curr_force: wp.array(dtype=wp.float32),
                      max_force: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if max_force[tid] < curr_force[tid]:
        max_force[tid] = curr_force[tid]
        
    
def copy_state(state) -> State:
    new_state = State()
    new_state.particle_q = wp.clone(state.particle_q)
    new_state.particle_qd = wp.clone(state.particle_qd)
    new_state.particle_f = wp.clone(state.particle_f)

    new_state.body_q = wp.clone(state.body_q)
    new_state.body_qd = wp.clone(state.body_qd)
    new_state.body_f = wp.clone(state.body_f)
    # new_state.ground_f = wp.clone(state.ground_f)
    return new_state

def add_random_to_pose(pose, t_mean=0.0, t_std=0.0, r_mean=0.0, r_std=0.0):
    t = pose[:3, 3]
    rpy = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=False)

    t += np.random.normal(t_mean, t_std, size=3)
    rpy += np.random.normal(r_mean, r_std, size=3)

    new_pose = np.eye(4)
    new_pose[:3, 3] = t
    new_pose[:3, :3] = Rotation.from_euler('xyz', rpy, degrees=False).as_matrix()
    return new_pose

def mat33_to_quat(m):
    return Rotation.from_matrix(m).as_quat()

@wp.kernel
def update_particles_input(
    particle_q: wp.array(dtype=wp.vec3),
    input_q: wp.array(dtype=wp.vec3),
    offset: int):
    tid = wp.tid()
    particle_q[tid + offset] = input_q[tid]

def transform_points_scipy(points, transform7d, inv_trans=None):
    transformed_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        t = np.array(transform7d[:3])
        q = np.array(transform7d[3:])
        # Create a transformation matrix from translation and quaternion
        rot = Rotation.from_quat(q).as_matrix()
        trans_pt = np.dot(rot, points[i, :]) + t
        transformed_points[i, :] = trans_pt
    return transformed_points

@wp.kernel
def update_object_mass(ratio: wp.float32,
                        object_id: wp.int32,
                        body_mass: wp.array(dtype=wp.float32),
                        body_inv_mass: wp.array(dtype=wp.float32),
                        body_inertia: wp.array(dtype=wp.mat33),
                        body_inv_inertia: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    if tid == object_id:
        body_mass[tid] *= ratio
        body_inv_mass[tid] /= ratio
        body_inertia[tid] *= ratio
        body_inv_inertia[tid] /= ratio

def update_object_density(model, object_id, density, old_density):
    ratio = density / old_density
    # can be potential bug if more than 1 object
    wp.launch(update_object_mass, dim=1,
        inputs=[
            ratio,
            object_id],
        outputs=[
            model.body_mass,
            model.body_inv_mass,
            model.body_inertia,
            model.body_inv_inertia,
        ])