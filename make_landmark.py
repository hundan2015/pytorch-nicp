import json
import torch
import io3d


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


def convert_4d_to_landmark_get(data):
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


def toJson(data):
    l = []
    for i in data:
        t = {}
        t["x"] = i[0].item()
        t["y"] = i[1].item()
        t["z"] = i[2].item()
        l.append(t)
    return l


def points_on_triangle_to_3d(faces, vertices, ids, weights):
    # if type(weights) != torch.Tensor:
    #     weights = torch.tensor(weights, device=vertices.device)
    # target_vertices_index = faces[ids].int()
    # target_vertices = vertices[target_vertices_index]
    # weighted_vertices_transposed = target_vertices * weights.unsqueeze(-1).expand(
    #     -1, -1, 3
    # )
    # result = torch.sum(weighted_vertices_transposed, dim=2)
    # return result
    return points_to_3d(faces, vertices, ids, weights)


def points_to_3d(
    faces_list,
    verts_list,
    ids,
    weights,
):
    final_res = torch.sum(
        (
            verts_list[(faces_list[ids]).int()] * weights.unsqueeze(-1).expand(-1, -1, 3)
        ).transpose(0, 1),
        dim=0,
    )
    return final_res


if __name__ == "__main__":
    mesh = io3d.load_obj_as_mesh("./test_data/0.obj")
    mesh_landmark_data = read_4d_file("./target_lm.txt")
    ids, weights = convert_4d_to_landmark_get(mesh_landmark_data)
    temp_face = mesh.faces_list()[0][27798]
    print(temp_face)
    temp_vert = mesh.verts_list()[0][16065]
    print(temp_vert)
    # FIXME: 面相关的顶点正确，顶点坐标正确，但是还有什么不正确？
    three_points = mesh.verts_list()[0][temp_face]
    print(weights[0])
    shit = torch.tensor(weights[0]).unsqueeze(-1).expand(-1, 3)
    print(shit)
    res = three_points * shit
    target_3d_point = torch.sum(res.transpose(0, 1), dim=1)
    # TODO: 把上面这个算法进一步向量化，写一个测试算法
    weights = torch.tensor(weights)

    verts_list = mesh.verts_list()[0]
    faces_list = mesh.faces_list()[0]
    points_to_3d(ids, weights, verts_list, faces_list)
    tmp = mesh.verts_list()[0] * weights.unsqueeze(-1).expand(-1, -1, 3)

    print(res)
    print(three_points)
    # print(ids)
    # print(weights)
