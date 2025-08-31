import os
from collections import OrderedDict
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from train import MLPModel
from train_eval import load_object_model, parse_model_filename
from init_pose import InitializeFingers
from sim_gen import FEMTendon
from generate_partial_pointcloud import *
from scipy.spatial.transform import Rotation as R

def logit(x):
    eps = 1e-6
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x / (1 - x))

def compute_loss(pred, target):
    abs_error = torch.abs(pred - target)
    weights = torch.ones_like(pred) * 1e-3
    output_dim = pred.shape[1]
    idx = torch.arange(output_dim, device=pred.device)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    mask[:, [4, 7]] = True
    weighted_vals = 1e3 * (pred < 0).float() + 1e-3 * (pred >= 0).float()
    weights = torch.where(mask, weighted_vals, weights)
    weights[pred[:, -1] > 0.0, 9] = 1e3

    loss = torch.sum(weights * abs_error)
    return loss

def load_com(curr_dir, object_name):
    com_dir_file = os.path.join(curr_dir, '../models/coms/coms.npz')
    com = np.load(com_dir_file, allow_pickle=True)['com_dict'].item()[object_name]
    com_data = np.repeat(com[np.newaxis, :], 1, axis=0)

    return com_data

def load_data(curr_dir, object_name):
    num_points = 1024
    pc_dir = os.path.join(curr_dir, f"../models/partial_pointclouds/{object_name}")

    pc_file = None
    for file in os.listdir(pc_dir):
        if file.startswith(f'sample{num_points}') and file.endswith(".npz"):
            pc_file = os.path.join(pc_dir, file)
            print("Using point cloud file:", pc_file)
            break
    if pc_file is None:
        raise FileNotFoundError(f"No NPZ file found in {pc_dir} for {object_name} with {num_points} points")
    pc = np.load(pc_file, allow_pickle=True)['pc'].T
    pc_data = np.repeat(pc[np.newaxis, :], 1, axis=0)

    return pc_data, load_com(curr_dir, object_name)

def normalize_point_cloud(pc_tensor, centroids=None, max_distance=None):
    B, D, N = pc_tensor.shape
    if centroids is None:
        centroids = pc_tensor.mean(dim=2, keepdim=True)
    pc_centered = pc_tensor - centroids

    if max_distance is None:
        distances = torch.norm(pc_centered, dim=1)
        max_distance = distances.max(dim=1, keepdim=True)[0].unsqueeze(2)
        max_distance[max_distance < 1e-6] = 1e-6

    pc_normalized = pc_centered / max_distance
    return pc_normalized, centroids, max_distance

class StiffnessOptimizer:
    def __init__(self, 
                 model_dir="../sim_models/",
                 model_name="",
                 density=2.0, layer_num=5, hidden_dim=1024,
                 pointnet_dim=128, lr=1e-1, seed=42, epochs=10000,
                 save_progress=False,
                 use_density=True,):
                #  device="cuda"):
        self.model_dir = model_dir
        self.model_name = model_name
        self.density = density
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.pointnet_dim = pointnet_dim
        self.lr = lr
        self.seed = seed
        self.epochs = epochs
        self.save_progress = save_progress
        self.use_density = use_density

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.writer = None
        # self.writer = SummaryWriter(log_dir=os.path.join(self.curr_dir, "../runs/train_opt"))

        self.min_stiff = 13.5
        self.max_stiff = 17.0
        self.sample_num_points = 1024
        self.stiff_range = self.max_stiff - self.min_stiff

        self.model, self.X_mean, self.X_std, self.Y_mean, self.Y_std = self.load_model()

        self.pc_norm = []
        self.com_tensor = []
        self.invalid_poses = OrderedDict()
        self.valid_poses = OrderedDict()
        self.group_slices = []
        self.object_list = []

    def add_object(self, object_name, pose_id):
        self.object_name = object_name
        self.invalid_poses[object_name] = []
        self.valid_poses[object_name] = []
        start_idx, end_idx = len(self.pc_norm), len(self.pc_norm)

        # get the number of poses
        pose_dir = os.path.join(self.curr_dir, f"../pose_info")
        num_poses = np.load(pose_dir + f"/{object_name}.npz", allow_pickle=True)['T_pose'].shape[0]

        if pose_id >= -1:
            if pose_id >= num_poses:
                raise ValueError(f"Pose ID {pose_id} is out of range for object {object_name} with {num_poses} poses.")
                return

            pc_data, com_data = self.input_from_poseid(object_name, pose_id)
            pc_norm, com_tensor = self.norm_pc_com(pc_data, com_data)
            self.pc_norm.append(pc_norm)
            self.com_tensor.append(com_tensor)
        elif pose_id == -2:
            pc_data, com_data = self.input_from_fix(object_name)
            pc_norm, com_tensor = self.norm_pc_com(pc_data, com_data)
            self.pc_norm.append(pc_norm)
            self.com_tensor.append(com_tensor)
        elif pose_id == -3:
            for i in range(10):
                if i >= num_poses: break
                pc_data, com_data = self.input_from_poseid(object_name, i)
                if pc_data is None or com_data is None:
                    self.invalid_poses[object_name].append(i)
                    continue
                self.valid_poses[object_name].append(i)
                pc_norm, com_tensor = self.norm_pc_com(pc_data, com_data)
                self.pc_norm.append(pc_norm)
                self.com_tensor.append(com_tensor)
                end_idx = len(self.pc_norm)
        elif pose_id == -4:
            for i in range(10):
                if i >= num_poses: break
                pc_data, com_data = self.input_from_poseid_hdis(object_name, i)
                if pc_data is None or com_data is None:
                    self.invalid_poses[object_name].append(i)
                    continue
                self.valid_poses[object_name].append(i)
                pc_norm, com_tensor = self.norm_pc_com(pc_data, com_data)
                self.pc_norm.append(pc_norm)
                self.com_tensor.append(com_tensor)
                end_idx = len(self.pc_norm)
        else:
            raise ValueError(f"Invalid pose_id: {pose_id}. It should be >= -3.")
        end_idx = len(self.pc_norm)
        self.group_slices.append((start_idx, end_idx))

        self.object_list.append(object_name)
        print(f"Object {object_name} has {len(self.valid_poses[object_name])} valid poses and {len(self.invalid_poses[object_name])} invalid poses, index range: {start_idx}-{end_idx}")
    
    def finalize(self):
        self.pc_norm = torch.cat(self.pc_norm, dim=0)
        self.com_tensor = torch.cat(self.com_tensor, dim=0)

        if self.save_progress:
            self.save_dir = os.path.join(self.curr_dir, f"../experiments/diffsim_opt/")
            self.progress_files = []
            prefix = "opt_"
            if len(self.object_list) > 1: 
                prefix = "joint_"
            for obj_name in self.object_list:
                file_name = os.path.join(self.save_dir, f"{prefix}{obj_name}.txt")
                if os.path.exists(file_name):
                    os.remove(file_name)
                progress_file = open(file_name, "w")
                progress_file.write(f"Object: {obj_name}\n")
                progress_file.write(f"Density: {self.density}\n")
                progress_file.write(f"Model: {self.model_name}\n")
                self.progress_files.append(progress_file)


    def load_model(self):
        model_paths = load_object_model(os.path.join(self.curr_dir, self.model_dir), 'all', self.model_name)
        layer_num, dropout, hidden_dim, density, pointnet_dim, model_path = parse_model_filename(
            model_paths, density=self.density, layer_num=self.layer_num,
            hidden_dim=self.hidden_dim, pointnet_dim=self.pointnet_dim)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model_state = checkpoint["model_state_dict"]
        norm_params = checkpoint["norm_params"]

        X_mean = norm_params["X_mean"].to(self.device)
        X_std = norm_params["X_std"].to(self.device)
        Y_mean = norm_params["Y_mean"].to(self.device)
        Y_std = norm_params["Y_std"].to(self.device)

        nopt_dim = 22 + 3 + 1
        if not self.use_density:
            nopt_dim = 22 + 3
        model = MLPModel(nopt_dim=nopt_dim, output_dim=10, 
                         hidden_dim=hidden_dim,
                         layer_num=layer_num, dropout=dropout, pointnet_dim=pointnet_dim)
        model.load_state_dict(model_state)
        model.to(self.device)
        model.eval()
        return model, X_mean, X_std, Y_mean, Y_std

    def norm_pc_com(self, pc_data, com_data):
        pc_tensor = torch.from_numpy(pc_data).float().to(self.device)
        pc_norm, pc_centroids, pc_scales = normalize_point_cloud(pc_tensor)

        com_tensor = torch.from_numpy(com_data).float().to(self.device)
        com_tensor = normalize_point_cloud(com_tensor.unsqueeze(2),
                                           centroids=pc_centroids,
                                           max_distance=pc_scales)[0].squeeze(2)
        return pc_norm, com_tensor
    
    def input_from_poseid(self, object_name, pose_id):
        finger_len = 11
        finger_rot = np.pi/30
        finger_width = 0.08
        scale = 5.0
        r, p, y = [-math.pi/2, 0.0, 0.0]
        object_rot = wp.quat_rpy(r, p, y)
        is_triangle = True

        finger_transform = None
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        pose_file = curr_dir + f"/../pose_info/init_opt/{object_name}.npz"
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

        if finger_transform is None:
            init_finger = InitializeFingers( 
                                finger_len=finger_len, 
                                finger_rot=finger_rot,
                                finger_width=finger_width,
                                stop_margin=0.0005,
                                scale=scale,
                                num_envs=1,
                                ycb_object_name=object_name,
                                object_rot=object_rot,
                                num_frames=30, 
                                iterations=10000,
                                is_render=False,
                                is_triangle=is_triangle,
                                pose_id=pose_id,
                                add_random=False,
                                post_height_offset=0.0)
            finger_transform, jq = init_finger.get_initial_position()
            if finger_transform is None or jq is None:
                print("Failed to initialize fingers. Redoing...")
                finger_transform, jq = init_finger.get_initial_position()
                if finger_transform is None or jq is None:
                    print("Still failed to initialize fingers after retrying.")
                    return None, None

        tendon = FEMTendon(
            object_rot=object_rot,
            object_density=1e1,
            ycb_object_name=object_name,
            is_render=False,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform)
        points = tendon.states[0].particle_q.numpy()

        # Create a point cloud from the points
        pc = trimesh.points.PointCloud(points)
        
        # Compute the minimum enclosing oriented bounding box
        obb = pc.bounding_box_oriented  # returns a trimesh.primitives.Box
        
        # Load the target mesh
        mesh = transform_and_scale_mesh(
                            tendon.obj_loader.mesh,
                            np.zeros(3), [r, p, y], scale)

        try:
            points_in_box = sample_points_in_box(mesh, obb, num_points=self.sample_num_points, oversample_factor=50)
        except ValueError as e:
            print(f"Error sampling points for {object_name} with pose id {pose_id}: {e}")
            return None, None
        
        points_in_box = points_in_box.T[np.newaxis, :, :]
        return points_in_box, load_com(self.curr_dir, object_name)
    
    def input_from_fix(self, object_name):
        finger_len = 11
        finger_rot = np.pi/30
        finger_width = 0.08
        scale = 5.0
        r, p, y = [-math.pi/2, 0.0, 0.0]
        object_rot = wp.quat_rpy(r, p, y)
        is_triangle = True

        init_finger = InitializeFingers( 
                            finger_len=finger_len, 
                            finger_rot=finger_rot,
                            finger_width=finger_width,
                            stop_margin=0.001,
                            scale=scale,
                            num_envs=1,
                            ycb_object_name=object_name,
                            object_rot=object_rot,
                            num_frames=30, 
                            iterations=10000,
                            is_render=True,
                            is_triangle=is_triangle,
                            pose_id=0,
                            add_random=False,
                            post_height_offset=0.0)
        quat = R.from_euler('xyz', [0.0, 1.6, 0.0]).as_quat()
        finger_transform, jq = init_finger.get_fixed_position(
            np.array([0.01, 0.95, -0.01,
                    quat[0], quat[1], quat[2], quat[3],
                    -0.23, 0.23]))

        tendon = FEMTendon(
            object_rot=object_rot,
            object_density=1e1,
            ycb_object_name=object_name,
            is_render=False,
            finger_len=finger_len,
            finger_rot=finger_rot,
            finger_width=finger_width,
            scale=scale,
            finger_transform=finger_transform)
        points = tendon.states[0].particle_q.numpy()

        # Create a point cloud from the points
        pc = trimesh.points.PointCloud(points)
        
        # Compute the minimum enclosing oriented bounding box
        obb = pc.bounding_box_oriented  # returns a trimesh.primitives.Box
        
        # Load the target mesh
        mesh = transform_and_scale_mesh(
                            init_finger.obj_loader.mesh,
                            np.zeros(3), [r, p, y], scale)

        try:
            points_in_box = sample_points_in_box(mesh, obb, num_points=self.sample_num_points, oversample_factor=50)
        except ValueError as e:
            print(f"Error sampling points for {object_name}: {e}")
            return None, None
        
        points_in_box = points_in_box.T[np.newaxis, :, :]
        return points_in_box, load_com(self.curr_dir, object_name)
    
    def get_loss(self, object_name, pose_id, stiffness, density, option='optimize'):
        pc_data, com_data = None, None
        if option == 'optimize':
            # use the pose from optimized initilization
            pc_data, com_data = self.input_from_poseid(object_name, pose_id)
        if option == 'hdis':
            # use the pose from heuristic displacement
            pc_data, com_data = self.input_from_
        if pc_data is None or com_data is None:
            return None
        pc_norm, com_tensor = self.norm_pc_com(pc_data, com_data)
        stiffness = torch.from_numpy(stiffness).float().to(self.device)
        density = torch.tensor([density], dtype=torch.float32, device=self.device)

        # Normalize the stiffness
        norm_stiffness = ((stiffness - self.X_mean[:22]) / self.X_std[:22]).unsqueeze(0)
        norm_density = density.expand(1, -1)  # [B, 1]

        X_tensor = torch.cat((norm_stiffness, com_tensor, norm_density), dim=1).to(self.device)
        if not self.use_density:
            X_tensor = torch.cat((norm_stiffness, com_tensor), dim=1).to(self.device)
        norm_pred = self.model(pc_norm, X_tensor, mode='train')  # shape: [B, 10]
            
        pred = norm_pred * self.Y_std + self.Y_mean  # shape: [B, 10]
        
        loss = compute_loss(pred, torch.zeros(10, dtype=torch.float32, device=self.device))
        return loss.item()
    
    def optimize_batch(self, pc_norm, com_tensor):
        z = logit((self.min_stiff + self.stiff_range * torch.rand(22, dtype=torch.float32, device=self.device)
                  - self.min_stiff) / self.stiff_range)
        z = logit((self.min_stiff + self.stiff_range * torch.ones(22, dtype=torch.float32, device=self.device)*0.6 - self.min_stiff) / self.stiff_range)
        print("z:", z)
        z = z.clone().detach().requires_grad_(True)
        optimizer = optim.AdamW([z], lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

        target = torch.zeros(10, dtype=torch.float32, device=self.device)
        prev_loss = float("inf")
        early_patience = 0
        early_patience_limit = 1000
        pose_patience = 0
        pose_patience_limit = 100

        # Get the batch size from pc_norm which is assumed to be [B, 3, N]
        B = pc_norm.shape[0]
        min_pose_id = np.zeros(len(self.group_slices), dtype=int)
        prev_pose_ids = [[] for _ in range(len(self.group_slices))]
        is_pose_id_fixed = np.zeros(len(self.group_slices), dtype=bool)

        this_den = torch.tensor([self.density], dtype=torch.float32, device=self.device)

        # start timer
        start_time = time.time()
        
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # Compute stiffness from z:
            stiffness = self.min_stiff + self.stiff_range * torch.sigmoid(z)   # shape: [22]
            norm_stiffness = ((stiffness - self.X_mean[:22]) / self.X_std[:22]).unsqueeze(0)  # shape: [1, 22]
            norm_stiffness_expanded = norm_stiffness.expand(B, -1)   # [B, 22]
            
            norm_density_expanded = this_den.expand(B, -1)  # [B, 1]
            
            # Concatenate all inputs
            X_tensor = torch.cat((norm_stiffness_expanded, com_tensor, norm_density_expanded), dim=1)  # shape: [B, 22 + 3 + 1]
            
            # Forward pass: note that self.model takes (pc_norm, stiffness+com + density features)
            norm_pred = self.model(pc_norm, X_tensor, mode='train')  # shape: [B, 10]
            pred = norm_pred * self.Y_std + self.Y_mean  # shape: [B, 10]
            
            loss_arr = torch.full((B,), float("inf"), device=self.device)

            # Compute per-sample loss
            for i in range(B):
                loss_arr[i] = compute_loss(pred[i, :].unsqueeze(0), target.unsqueeze(0))

            # Sum of best loss from each object group
            total_loss = 0.0
            best_pose_indices = []

            for i in range(len(self.group_slices)):
                start, end = self.group_slices[i]
                group_losses = loss_arr[start:end]
                # sort the losses in the group and add the least 5 ones
                sorted_indices = torch.argsort(group_losses, dim=0)
                best_indices = sorted_indices[:5]

                best_indices_list = best_indices.tolist()
                if prev_pose_ids[i] == best_indices_list:
                    pose_patience += 1
                    if pose_patience > pose_patience_limit:
                        is_pose_id_fixed[i] = True

                prev_pose_ids[i] = best_indices_list
                if is_pose_id_fixed[i]:
                    best_indices = best_indices[0:1]
                total_loss += group_losses[best_indices].sum()
                best_pose_indices.append(best_indices[0].item())

                if epoch % 1000 == 0:
                    print("total_loss:", total_loss.item())
                    print(f"Best pose indices for group {i}: {best_indices}")
            min_pose_id = np.array(best_pose_indices)

            # For debugging:
            if epoch % 1000 == 0:
                print("Loss per sample:", loss_arr)
                for idx, (start, end) in enumerate(self.group_slices):
                    print(f"Group {idx}: min loss = {torch.min(loss_arr[start:end]).item():.6f}")
            
            loss = total_loss
            if self.writer is not None:
                self.writer.add_scalar("Loss/Loss", loss.item(), epoch)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())
            if self.writer is not None:
                self.writer.add_scalar("Grad/gradient", torch.norm(z.grad), epoch)

            if self.save_progress:
                for i in range(len(self.group_slices)):
                    start, end = self.group_slices[i]
                    self.progress_files[i].write(f"{epoch},{loss_arr[start:end][min_pose_id[i]].sum():.6f},{torch.norm(z.grad):.6f},{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}\n")
                    self.progress_files[i].flush()
            if loss.item() < 1e-4:
                break
            if abs(loss.item() - prev_loss) < 1e-6:
                early_patience += 1
                if early_patience > early_patience_limit:
                    print("Early stopping due to no improvement in loss:", loss.item())
                    break
            else:
                early_patience = 0
            prev_loss = loss.item()

        # end timer
        elapsed_time = time.time() - start_time
        print(f"Optimization completed in {elapsed_time:.2f} seconds.")
        
        final_stiffness = self.min_stiff + self.stiff_range * torch.sigmoid(z)

        return final_stiffness.detach().cpu().numpy(), pred.detach().cpu().numpy(), min_pose_id, loss.detach().cpu().numpy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_name", type=str, default="all", help="Name of the object to optimize")
    parser.add_argument("--density", type=float, default=2.0, help="Density of the object")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs for optimization")

    # pose_id >= 0: actual pose id for the object
    # pose_id == -1: random pose
    # pose_id == -2: fixed pose
    # pose_id == -3: use all poses
    # pose_id == -4: use heuristic displacement for all poses
    parser.add_argument("--pose_id", type=int, default=0, help="Pose ID for the object")
    parser.add_argument("--model_name", type=str, default="", help="Name of the model to use")
    parser.add_argument("--save", action="store_true", help="Save the optimized model")
    parser.add_argument("--save_progress", action="store_true", help="Save the progress of the optimization")

    args = parser.parse_args()

    curr_dir = os.path.dirname(os.path.abspath(__file__))

    optimizer = StiffnessOptimizer(
                epochs=args.epochs,
                model_name=args.model_name + ".pth",
                density=args.density,
                save_progress=args.save_progress,)
                # device=args.device,)
    object_names = [args.object_name]
    if args.object_name == "all":
        object_names = get_object_names(curr_dir + '/../data_gen/')
    # object_names = ['006_mustard_bottle'
    #                 , '050_medium_clamp', '030_fork', '004_sugar_box', '025_mug']
    print("object_names:", object_names)
    
    for name in object_names:
        optimizer.add_object(name, args.pose_id)
    optimizer.finalize()
    # stiffness, prediction = optimizer.optimize(optimizer.pc_norm, optimizer.com_tensor)
    stiffness, prediction, min_id, loss = optimizer.optimize_batch(optimizer.pc_norm, optimizer.com_tensor)

    if args.save:
        save_dir = os.path.join(curr_dir, f"../stiff_results/{args.model_name}/")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{args.object_name}_pose{args.pose_id}density{args.density}.npz")
        np.savez(save_path, stiffness=stiffness, prediction=prediction, min_id=min_id, loss=loss)
        print(f"Optimized stiffness saved to {save_path}")

    print("\nOptimized stiffness (22d):")
    print(stiffness[0:11])
    print(stiffness[11:])
    print("\nFinal model prediction (9d):")
    print(prediction)

    if args.save_progress:
        for progress_file in optimizer.progress_files:
            progress_file.close()

