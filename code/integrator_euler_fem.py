import warp as wp
from warp.sim.integrator_euler import *
from warp.sim.model import PARTICLE_FLAG_ACTIVE, Control, Model, ModelShapeGeometry, ModelShapeMaterials, State
from utils import * 

@wp.kernel
def eval_external_particle_forces(
        input_ids: wp.array(dtype=int),
        input_forces: wp.array(dtype=wp.vec3), 
        particle_f: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    wp.atomic_add(particle_f, input_ids[tid], input_forces[tid])

@wp.kernel
def eval_triangles_contact(
    # idx : wp.array(dtype=int), # list of indices for colliding particles
    num_particles: int,  # size of particles
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    face_no = tid // num_particles  # which face
    particle_no = tid % num_particles  # which particle

    # at the moment, just one particle
    pos = x[particle_no]

    i = indices[face_no, 0]
    j = indices[face_no, 1]
    k = indices[face_no, 2]

    if i == particle_no or j == particle_no or k == particle_no:
        return

    p = x[i]  # point zero
    q = x[j]  # point one
    r = x[k]  # point two

    # vp = v[i] # vel zero
    # vq = v[j] # vel one
    # vr = v[k] # vel two

    # qp = q-p # barycentric coordinates (centered at p)
    # rp = r-p

    bary = triangle_closest_point_barycentric(p, q, r, pos)
    closest = p * bary[0] + q * bary[1] + r * bary[2]

    diff = pos - closest
    dist = wp.dot(diff, diff)
    n = wp.normalize(diff)
    c = wp.min(dist - 5e-4, 0.0)  # 0 unless within threshold of surface
    # c = wp.leaky_min(dot(n, x0)-0.01, 0.0, 0.0)
    fn = n * c * 1e5

    wp.atomic_sub(f, particle_no, fn)

    # # apply forces (could do - f / 3 here)
    wp.atomic_add(f, i, fn * bary[0])
    wp.atomic_add(f, j, fn * bary[1])
    wp.atomic_add(f, k, fn * bary[2])


def eval_external_forces(model: Model, state: State, control: Control, particle_f: wp.array):
    if model.particle_count:
        wp.launch(
            kernel=eval_external_particle_forces,
            dim=len(control.waypoint_ids),
            inputs=[
                control.waypoint_ids,
                control.waypoint_forces],
            outputs=[particle_f],
            device=model.device,
        )

def eval_triangle_contact_forces(model: Model, state: State, particle_f: wp.array):
    if model.enable_tri_collisions:
        wp.launch(
            kernel=eval_triangles_contact,
            dim=model.tri_count * model.particle_count,
            inputs=[
                model.particle_count,
                state.particle_q,
                state.particle_qd,
                model.tri_indices,
                model.tri_materials,
            ],
            outputs=[particle_f],
            device=model.device,
        )

@wp.kernel
def eval_tetrahedra(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    indices: wp.array2d(dtype=int),
    pose: wp.array(dtype=wp.mat33),
    activation: wp.array(dtype=float),
    materials: wp.array2d(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    act = activation[tid]

    k_mu = materials[tid, 0]
    k_lambda = materials[tid, 1]
    k_damp = materials[tid, 2]

    x0 = x[i]
    x1 = x[j]
    x2 = x[k]
    x3 = x[l]

    v0 = v[i]
    v1 = v[j]
    v2 = v[k]
    v3 = v[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0

    Ds = wp.mat33(x10, x20, x30)
    Dm = pose[tid]

    inv_rest_volume = wp.determinant(Dm) * 6.0
    rest_volume = 1.0 / inv_rest_volume

    alpha = 1.0 + k_mu / k_lambda - k_mu / (4.0 * k_lambda)

    # scale stiffness coefficients to account for area
    k_mu = k_mu * rest_volume
    k_lambda = k_lambda * rest_volume
    k_damp = k_damp * rest_volume

    # F = Xs*Xm^-1
    F = Ds * Dm
    dFdt = wp.mat33(v10, v20, v30) * Dm

    col1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    col2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    col3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])

    # -----------------------------
    # Neo-Hookean (with rest stability [Smith et al 2018])

    Ic = wp.dot(col1, col1) + wp.dot(col2, col2) + wp.dot(col3, col3)

    # deviatoric part
    P = F * k_mu * (1.0 - 1.0 / (Ic + 1.0)) + dFdt * k_damp
    H = P * wp.transpose(Dm)

    f1 = wp.vec3(H[0, 0], H[1, 0], H[2, 0])
    f2 = wp.vec3(H[0, 1], H[1, 1], H[2, 1])
    f3 = wp.vec3(H[0, 2], H[1, 2], H[2, 2])

    # hydrostatic part
    J = wp.determinant(F)

    # print(J)
    s = inv_rest_volume / 6.0
    dJdx1 = wp.cross(x20, x30) * s
    dJdx2 = wp.cross(x30, x10) * s
    dJdx3 = wp.cross(x10, x20) * s

    f_volume = (J - alpha + act) * k_lambda
    f_damp = (wp.dot(dJdx1, v1) + wp.dot(dJdx2, v2) + wp.dot(dJdx3, v3)) * k_damp

    f_total = f_volume + f_damp

    f1 = f1 + dJdx1 * f_total
    f2 = f2 + dJdx2 * f_total
    f3 = f3 + dJdx3 * f_total
    f0 = -(f1 + f2 + f3)

    # apply forces
    wp.atomic_sub(f, i, f0)
    wp.atomic_sub(f, j, f1)
    wp.atomic_sub(f, k, f2)
    wp.atomic_sub(f, l, f3)

def eval_tetrahedral_forces(model: Model, state: State, control: Control, particle_f: wp.array):
    if model.tet_count:
        wp.launch(
            kernel=eval_tetrahedra,
            dim=model.tet_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                model.tet_indices,
                model.tet_poses,
                control.tet_activations,
                model.tet_materials,
            ],
            outputs=[particle_f],
            device=model.device,
        )

@wp.kernel
def eval_particle_contacts(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.uint32),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    particle_ke: float,
    particle_kd: float,
    particle_kf: float,
    particle_mu: float,
    particle_ka: float,
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3), #not
    contact_body_vel: wp.array(dtype=wp.vec3), #yes
    contact_normal: wp.array(dtype=wp.vec3), #no
    contact_max: int,
    body_f_in_world_frame: bool,
    # outputs
    particle_f: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    count = min(contact_max, contact_count[0])
    if tid >= count:
        return

    shape_index = contact_shape[tid]
    body_index = shape_body[shape_index]
    particle_index = contact_particle[tid]
    if (particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE) == 0:
        return

    px = particle_x[particle_index]
    pv = particle_v[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    body_v_s = wp.spatial_vector()

    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        body_v_s = body_qd[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)
    # wp.printf("tid: %d, contact_body_pos: %f %f %f\n", tid, contact_body_pos[tid][0], contact_body_pos[tid][1], contact_body_pos[tid][2])

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - particle_radius[particle_index]

    if c > particle_ka:
        return

    # take average material properties of shape and particle parameters
    ke = 0.5 * (particle_ke + shape_materials.ke[shape_index])
    kd = 0.5 * (particle_kd + shape_materials.kd[shape_index])
    kf = 0.5 * (particle_kf + shape_materials.kf[shape_index])
    mu = 0.5 * (particle_mu + shape_materials.mu[shape_index])

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    # body_v is not differentiable
    # contact_body_vel is differentiable
    bv = body_v + wp.transform_vector(X_wb, contact_body_vel[tid])
    if body_f_in_world_frame:
        bv += wp.cross(body_w, bx)
    else:
        bv += wp.cross(body_w, r)

    # relative velocity
    # this bv is potentially not differentiable
    v = pv - bv

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn

    # Regularization parameter
    eps = 1e-5
    # vt = smooth_zero(v - n * vn, eps)

    # contact elastic
    # fn = n * c * ke
    fn = c * ke

    # contact damping
    # fd = n * wp.min(vn, 0.0) * kd 
    fd = wp.min(vn, 0.0) * kd 
    # Smooth normal force
    # fd = n * smooth_min(vn, 0.0, eps) * kd
    # vn_smooth = - wp.sqrt(vn * vn + eps * eps)
    # fd = n * vn_smooth * kd

    # ft = wp.vec3(0.0)
    # viscous friction
    # ft = vt*kf
    # ft = vt* 1.0e1
    # wp.printf("tid: %d, viscous: %f %f %f\n", tid, ft[0], ft[1], ft[2])

    # Coulomb friction (box)
    # lower = mu * c * ke
    # upper = -lower

    # vx = wp.clamp(wp.dot(wp.vec3(kf, 0.0, 0.0), vt), lower, upper)
    # vz = wp.clamp(wp.dot(wp.vec3(0.0, 0.0, kf), vt), lower, upper)

    # ft = wp.vec3(vx, 0.0, vz)

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    # vt_length = wp.sqrt(wp.dot(vt, vt) + eps*eps)
    # ft += vt / vt_length * wp.min(kf * vt_length, abs(mu * c * ke))
    # ft += wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))
    # ft += wp.normalize(vt) * wp.min(1e4 * wp.length(vt), abs(mu * c * 1e3))
    ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))

    # from eric
    # use a smooth vector norm to avoid gradient instability at/around zero velocity
    # vs = norm_huber(vt, delta=1.0)
    # if vs > 0.0:
    #     fr = vt / vs
    #     ft += fr * wp.min(kf * vs, -mu * (fn + fd))

    # static friction
    # vt_length = wp.sqrt(wp.dot(vt, vt) + eps*eps)
    # ft += wp.normalize(vt) * 1e5*mu * c * ke
    # ft += wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(mu * c * ke))
    # wp.printf("tid: %d, static: %f %f %f\n", tid, ft[0], ft[1], ft[2])

    f_total = n * (fn + fd) + ft
    # f_total = fn + fd + ft
    # f_total = fn
    # f_total = wp.vec3(1.0, 1.0, 1.0)
    # f_total = contact_normal[tid]


    wp.atomic_sub(particle_f, particle_index, f_total)

    if body_index >= 0:
        if body_f_in_world_frame:
            wp.atomic_sub(body_f, body_index, wp.spatial_vector(wp.cross(bx, f_total), f_total))
        else:
            # wp.atomic_add(body_f, body_index, wp.spatial_vector(wp.cross(wp.vec3(1.0, 1.0, 1.0), f_total), f_total))
            wp.atomic_add(body_f, body_index, wp.spatial_vector(wp.cross(r, f_total), f_total))

def eval_particle_body_contact_forces(
    model: Model, state: State, particle_f: wp.array, body_f: wp.array, body_f_in_world_frame: bool = False
):
    if model.particle_count and model.shape_count > 1:
        wp.launch(
            kernel=eval_particle_contacts,
            dim=model.soft_contact_max,
            inputs=[
                state.particle_q,
                state.particle_qd,
                state.body_q,
                state.body_qd,
                model.particle_radius,
                model.particle_flags,
                model.body_com,
                model.shape_body,
                model.shape_materials,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_kf,
                model.soft_contact_mu,
                model.particle_adhesion,
                model.soft_contact_count,
                model.soft_contact_particle,
                model.soft_contact_shape,
                model.soft_contact_body_pos,
                model.soft_contact_body_vel,
                model.soft_contact_normal,
                model.soft_contact_max,
                body_f_in_world_frame,
            ],
            # outputs
            outputs=[particle_f, body_f],
            device=model.device,
        )

@wp.kernel
def additional_vels(
    target_vels: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    wp.atomic_add(particle_v, tid, target_vels[0])

def compute_additional_vels(model: Model, state: State, control: Control):
    wp.launch(
        kernel=additional_vels,
        dim = len(state.particle_q),
        inputs=[control.vel_values],
        outputs=[state.particle_qd],
        device=model.device,
    )

def compute_fem_forces(model: Model, state: State, control: Control, particle_f: wp.array, body_f: wp.array, dt: float):
    # damped springs
    eval_spring_forces(model, state, particle_f)

    # triangle elastic and lift/drag forces
    eval_triangle_forces(model, state, control, particle_f)

    # triangle/triangle contacts
    eval_triangle_contact_forces(model, state, particle_f)

    # triangle bending
    eval_bending_forces(model, state, particle_f)

    # tetrahedral FEM
    eval_tetrahedral_forces(model, state, control, particle_f)

    # body joints
    eval_body_joint_forces(model, state, control, body_f)

    # particle-particle interactions
    eval_particle_forces(model, state, particle_f)

    # particle ground contacts
    eval_particle_ground_contact_forces(model, state, particle_f)

    # body contacts
    eval_body_contact_forces(model, state, particle_f)

    # particle shape contact
    eval_particle_body_contact_forces(model, state, particle_f, body_f, body_f_in_world_frame=False)

    eval_external_forces(model, state, control, particle_f) 

class FEMIntegrator(SemiImplicitIntegrator):
    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            if control is None:
                control = model.control(clone_variables=False)

            compute_fem_forces(model, state_in, control, particle_f, body_f, dt)
            compute_additional_vels(model, state_in, control)

            self.integrate_bodies(model, state_in, state_out, dt, self.angular_damping)

            self.integrate_particles(model, state_in, state_out, dt)

            return state_out
