import os
import numpy as np
import warp as wp
import trimesh
import utils

class ObjectLoader:
    def __init__(self, obj_name='', scale=1.0):
        self.obj_name = obj_name
        self.scale = scale
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/ycb/')
        self.mid_height = 0.0

    def add_fix_box(self, builder, box_x, box_height, box_width, pos: np.array, rot: wp.quat, scale: float):
        self.obj_name = 'fix_box'
        builder.add_shape_box(
            body=-1, pos=pos,
            hx=box_x, hy=box_height/2, hz=box_width/2,
            mu=0.1, density=1e2)
        self.mid_height = box_height/3

    def add_box(self, builder, box_x, box_height, box_width, pos: np.array, rot: wp.quat, scale: float):
        self.obj_name = 'box'
        b = builder.add_body(
            # origin=wp.transform(pos, 
            origin=wp.transform(np.zeros(3), 
                                wp.quat_identity()))
        builder.add_shape_box(
            body=b, pos=pos, 
            hx=box_x, hy=box_height/2, hz=box_width/2, 
            density=1e1,
            ke=0.0e-5,
            kd=1.0e-5,
            kf=1e1,
            mu=1.0)
        self.mid_height = box_height / 2
      
    def add_ycb(self, builder, 
                pos: np.array, rot: wp.quat, 
                scale: float, 
                ke=1.0e-5, kd=1.0e-1,
                kf=1e1, mu=1.0,
                obj_name='006_mustard_bottle',
                use_simple_mesh=False, is_fix=False,
                density=1e1):
        # load a ycb object mesh file to the builder
        self.obj_name = obj_name
        # self.obj_name = '005_tomato_soup_can'
        # self.obj_name = '014_lemon'
        # self.obj_name = '012_strawberry'
        # self.obj_name = '025_mug'
        # self.obj_name = '013_apple'

        mesh_name = "nontextured.ply"
        if use_simple_mesh:
            mesh_name = "simple_nontextured.ply"
        obj_mesh = trimesh.load(os.path.join(self.data_dir, f'{self.obj_name}/google_16k/' + mesh_name))
        # obj_mesh = trimesh.load(os.path.join(self.data_dir, f'{self.obj_name}/google_16k/nontextured.ply'))
        shift_vs = obj_mesh.vertices - np.mean(obj_mesh.vertices, axis=0)
        mesh = wp.sim.Mesh(shift_vs, obj_mesh.faces)
        # print("object mesh vertices num: ", shift_vs.shape)
        self.mesh = obj_mesh

        # Get the bounding box
        offset = np.array([0.0, -scale*np.min(shift_vs[:, 2])+1e-4, 0.0])
        self.mid_height = scale * (np.max(shift_vs[:, 2]) - np.min(shift_vs[:, 2])) / 2

        # Get the oriented bounding box
        bbox = obj_mesh.bounding_box_oriented
        vertices = bbox.vertices
        v = None
        edges = [
            vertices[1] - vertices[0],
            vertices[2] - vertices[0],
            vertices[4] - vertices[0],
        ]
        edge_length = [np.linalg.norm(edge) for edge in edges]
        max_idx = np.argmax(edge_length)
        v = edges[max_idx]
        v = wp.transform_vector(wp.transform(pos, rot), wp.vec3(np.array(v)))
        v = wp.array(v).numpy()
        if v[1] < 0: v = -v
        object_com = wp.transform_point(wp.transform(pos + offset, rot), wp.vec3(np.array([0.0, 0.0, 0.0])))
        object_com = wp.array(object_com, dtype=wp.vec3)
        
        # very annoying, if penentrate the ground, the object will bounce super high
        b = builder.add_body(
            # origin=wp.transform(pos, 
            origin=wp.transform(
                        # pos.tolist(), 
                        pos + offset,
                        rot))
        # print("actual pos: ", pos + offset)
        if is_fix: 
            b = -1
            pos = pos + offset
        else:
            pos = wp.vec3(0.0, 0.0, 0.0)
            rot = wp.quat_identity()
        geo_id = builder.add_shape_mesh(
            body=b,
            mesh=mesh,
            pos=pos,
            rot=rot,
            scale=wp.vec3(scale, scale, scale),
            density=density,
            ke=ke,
            kd=kd,
            kf=kf,
            mu=mu,
        )
        # print(f"Added {self.obj_name} to the scene")
        return object_com, b, geo_id
