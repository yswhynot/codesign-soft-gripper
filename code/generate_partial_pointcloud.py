import os
# import open3d as o3d
import math
import argparse
import numpy as np
import trimesh
import warp as wp

from trimesh.transformations import translation_matrix, euler_matrix, scale_matrix, concatenate_matrices
from sim_gen import FEMTendon

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut a mesh with the minimum enclosing oriented bounding box from a set of 3D points"
    )
    parser.add_argument("--object_name", type=str, default="all", help="Name of the object")
    return parser.parse_args()

def transform_and_scale_mesh(mesh, pos, rot, scale):
    """
    Transform a mesh by applying translation, rotation, and scaling.
    
    Args:
        mesh (trimesh.Trimesh): The input mesh.
        pos (array-like): Translation vector, e.g. [tx, ty, tz].
        rot (array-like): euler angles in radians for rotation about x, y, z axes.
        scale (float): Uniform scaling factor.
    
    Returns:
        transformed_mesh (trimesh.Trimesh): The transformed mesh.
    """
    this_mesh = trimesh.Trimesh(
                vertices=mesh.vertices - np.mean(mesh.vertices, axis=0), 
                faces=mesh.faces, process=False)
    offset = np.array([0.0, -scale*np.min(this_mesh.vertices[:, 2])+1e-4, 0.0])
    pos += offset
    T = translation_matrix(pos)
    R = euler_matrix(rot[0], rot[1], rot[2])
    S = scale_matrix(scale, origin=[0, 0, 0])
    
    # The combined transform M = T * R * S. (Matrix multiplication order is from right to left)
    M = concatenate_matrices(T, R, S)
    
    # Apply the transformation
    this_mesh.apply_transform(M)
    return this_mesh


def sample_points_in_box(mesh, obb, num_points=1024, oversample_factor=10):
    # Sample a large number of points from the original mesh surface.
    sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points * oversample_factor)
    
    # Get the transformation matrix of the OBB. 
    box_transform_inv = np.linalg.inv(obb.primitive.transform)
    
    # Convert sampled points to homogeneous coordinates for transformation.
    sampled_points_h = np.hstack([sampled_points, np.ones((sampled_points.shape[0], 1))])
    
    # Transform points into the box's local coordinate system.
    points_local = (box_transform_inv @ sampled_points_h.T).T[:, :3]
    
    # The box in local coordinates is centered at the origin with extents given by obb.primitive.extents.
    half_extents = obb.primitive.extents / 2.0
    
    # Create a mask for points inside the box (all coordinates within half extents).
    inside_mask = np.all(np.abs(points_local) <= half_extents, axis=1)
    filtered_points = sampled_points[inside_mask]
    
    # Check if we have enough points. If yes, randomly sample num_points.
    if filtered_points.shape[0] >= num_points:
        indices = np.random.choice(filtered_points.shape[0], num_points, replace=False)
        return filtered_points[indices]
    elif filtered_points.shape[0] > 100:
        print(f"Warning: Only {filtered_points.shape[0]} points inside the bounding box. Duplicating...")
        # If not enough points, duplicate some to reach num_points.
        indices = np.random.choice(filtered_points.shape[0], num_points, replace=True)
        return filtered_points[indices]
    else:
        raise ValueError(f"Not enough points inside the bounding box: {filtered_points.shape[0]} < 101")


def save_point_cloud(pc, output_file):
    """
    Save a point cloud to a PLY file.
    
    Args:
        pc (np.ndarray): Point cloud array of shape [B, 3, N] or [3, N]. If batched, the first batch is used.
        output_file (str): Destination PLY file.
    """
    N = pc.shape[0]
    assert pc.shape == (N, 3), f"Expected point cloud shape (3, N), but got {pc.shape}"
    with open(output_file, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(pc.shape[0]))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(N):
            point = pc[i, :]
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved point cloud to {output_file}")

def init_finger_mesh(object_name, data_file, object_rot, scale):
    finger_len = 11
    finger_rot = np.pi/30
    finger_width = 0.08
    is_triangle = True
    finger_transform = None

    # can speed up by directly apply transforms but i'm lazy af
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = curr_dir + f"/../data_gen/{object_name}_frame850/rand/{data_file}.npz"
    data = np.load(file_path, allow_pickle=True)
    finger_transform_list = data['init_transform']
    finger_transform = [wp.transform(finger_transform_list[0][0:3], finger_transform_list[0][3:]), wp.transform(finger_transform_list[1][0:3], finger_transform_list[1][3:])]
    try:
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
    except Exception as e:
        print(f"Error initializing tendon: {e}")
        return None, None, None
    points = tendon.states[0].particle_q.numpy()

    return tendon, points, finger_transform

def get_all_files(dir_path, extension=".npz"):
    """
    Get all files in a directory with a specific extension.
    
    Args:
        dir_path (str): Path to the directory.
        extension (str): File extension to filter by.
        
    Returns:
        list: List of file paths with the specified extension.
    """
    files = []
    for root, dirs, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(extension):
                # get only the name without the extension
                name_without_ext = os.path.splitext(filename)[0]
                files.append(name_without_ext) 
    return files

def get_object_names(data_dir):
    """Get all object names from the data directory."""
    object_names = []
    for file in os.listdir(data_dir):
        name = file.split('_frame')[0]  # Extract name before '_frame'
        if name not in object_names:
            object_names.append(name)
    return object_names

def main():
    scale = 5.0
    r, p, y = [-math.pi/2, 0.0, 0.0]
    num_points = 1024

    args = parse_args()
    curr_dir = os.path.dirname(os.path.abspath(__file__))

    object_list = [args.object_name]
    if args.object_name == "all":
        object_list = get_object_names(curr_dir + '/../data_gen/')
    print("object_list:", object_list)

    for obj_name in object_list:
        file_names = get_all_files(curr_dir + f"/../data_gen/{obj_name}_frame850/rand/")

        save_dir = curr_dir + f"/../models/partial_pointclouds/{obj_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created {save_dir}")
        if not os.path.exists(save_dir + "/ply/"):
            os.makedirs(save_dir + "/ply/")
    
        count = 0
        for file_name in file_names:
            # check if this has been processed already
            if os.path.exists(save_dir + f"sample{num_points}_{file_name}.npz"):
                print(f"File {file_name} already processed, skipping...")
                continue

            init_finger, points, finger_transform = init_finger_mesh(obj_name, file_name, wp.quat_rpy(r, p, y), scale)
            if init_finger is None:
                print(f"Failed to initialize finger for {file_name}, skipping...")
                continue

            # Ensure points are in shape [N, 3]
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"Expected points of shape [N, 3], but got {points.shape}")
            
            # Create a point cloud from the points
            pc = trimesh.points.PointCloud(points)
            
            # Compute the minimum enclosing oriented bounding box
            obb = pc.bounding_box_oriented  # returns a trimesh.primitives.Box
            
            # Convert the oriented bounding box to a mesh
            box_mesh = obb.to_mesh()
            
            # Load the target mesh
            mesh = transform_and_scale_mesh(init_finger.obj_loader.mesh,
                                            np.zeros(3), [r, p, y], scale)

            try:
                points_in_box = sample_points_in_box(mesh, obb, num_points=num_points, oversample_factor=50)
            except ValueError as e:
                print(f"Error sampling points for {file_name}: {e}")
                continue

            np.savez(save_dir + f"sample{num_points}_{file_name}.npz", pc=points_in_box)

            if count % 50 == 0:
                save_point_cloud(points_in_box, save_dir + f"/ply/points{num_points}_{file_name}.ply")
                print(f"Processed {count} files, saved point cloud for {file_name}")
            count += 1
    

if __name__ == "__main__":
    main()