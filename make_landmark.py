import json
import torch


def read_4d_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def read_4d_xyz_file(file_path):
    res = []
    with open(file_path, "r") as f:
        data = json.load(f)
        for i in data:
            res.append([i["x"], i["y"], i["z"]])
    return res


def convert_4d_to_landmark(data, face_info):
    """
    Args:
        data (list:list:float):
    Returns:
        landmarks (list:int):
        weights (list:float):
    """
    landmark_dict = {}
    face_info = face_info.cpu()
    for tp in data:
        face_id, r1, r2 = tp
        face_id = int(face_id)
        # r是重心坐标
        r3 = 1 - r1 - r2
        r_list = [r1, r2, r3]
        p_list = face_info[face_id].numpy().tolist()
        for i in range(len(r_list)):
            if p_list[i] in landmark_dict:
                landmark_dict[p_list[i]] += r_list[i]
            else:
                landmark_dict[p_list[i]] = r_list[i]
    landmarks = []
    weights = []
    for key, value in landmark_dict.items():
        landmarks.append(key)
        weights.append(value)
    return landmarks, weights


def convert_4d_to_landmark_get(data, face_info):
    face_info = face_info.cpu()
    ids = []
    weights = []
    for tp in data:
        face_id, r1, r2 = tp
        face_id = int(face_id)
        # r是重心坐标
        r3 = 1 - r1 - r2
        ids.append(face_id)
        weights.append([r1, r2, r3])
    return ids, weights


def points_on_triangle_to_3d(faces, vertices, ids, weights):
    if type(weights) != torch.Tensor:
        weights = torch.tensor(weights, device=vertices.device)
    target_vertices_index = faces[ids].int()
    target_vertices = vertices[target_vertices_index]
    weighted_vertices_transposed = target_vertices * weights.unsqueeze(-1).expand(
        -1, -1, 3
    )
    result = torch.sum(weighted_vertices_transposed, dim=2)
    return result
