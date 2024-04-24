# Copyright 2021 by Haozhe Wu, Tsinghua University, Department of Computer Science and Technology.
# All rights reserved.
# This file is part of the pytorch-nicp,
# and is released under the "MIT License Agreement". Please see the LICENSE
# file that should have been included as part of this package.

import torch
import io3d
import render
import numpy as np
import json
from utils import normalize_mesh
from nicp import non_rigid_icp_mesh2mesh
import make_landmark
import data

# demo for registering mesh
# estimate landmark for target meshes
# the face must face toward z axis
# the mesh or point cloud must be normalized with normalize_mesh/normalize_pcl function before feed into the nicp process
device = torch.device("cuda:0")
meshes = io3d.load_obj_as_mesh("./test_data/0.obj", device=device)

with torch.no_grad():
    norm_meshes, norm_param = normalize_mesh(meshes)

    bfm_meshes = io3d.load_obj_as_mesh("./BFM/dnaraw.obj", device=device)
    bfm_meshes, bfm_meshes_norm = normalize_mesh(bfm_meshes)

    metahuman_landmark_data = make_landmark.read_4d_file("./landmark.txt")
    landmarks, weights = make_landmark.convert_4d_to_landmark(
        metahuman_landmark_data, bfm_meshes.faces_list()[0]
    )
    landmarks = [landmarks]
    bfm_lm_index_m = torch.from_numpy(np.array(landmarks)).to(device).long()
    # USE LESS
    target_lm_index_m = (
        torch.from_numpy(np.array(data.reconstruction_points)).to(device).long()
    )
    # USE LESS END
    # target_lm = make_landmark.read_4d_xyz_file("./lmshit.txt")
    target_lm_data = make_landmark.read_4d_file("./target_lm.txt")
    target_lm_ids, target_lm_weights = make_landmark.convert_4d_to_landmark_get(
        target_lm_data
    )
    target_lm = make_landmark.points_on_triangle_to_3d(
        norm_meshes.faces_list()[0],
        norm_meshes.verts_list()[0],
        torch.tensor(target_lm_ids, device=device).int(),
        torch.tensor(target_lm_weights, device=device),
    )
    print(make_landmark.toJson(target_lm))
    # target_lm = torch.from_numpy(np.array(([target_lm]))).to(device)
    target_lm = target_lm.unsqueeze(0)

fine_config = json.load(open("config/ultra_grain.json"))
registered_mesh = non_rigid_icp_mesh2mesh(
    bfm_meshes,
    norm_meshes,
    bfm_lm_index_m,
    target_lm_index_m,
    fine_config,
    target_lm=target_lm,
    point_landmark_data=metahuman_landmark_data,
)
io3d.save_meshes_as_objs(["final.obj"], registered_mesh, save_textures=False)
