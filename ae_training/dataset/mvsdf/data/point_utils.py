import numpy as np
import open3d as o3d


def pack_colors(r, g, b):
    return np.uint32(r) * 16**4 + np.uint32(g) * 16**2 + np.uint32(b)


def load_points(points, color_format='packed', get_normals=False, dtype=np.float32):
    ply = o3d.io.read_point_cloud(points)
    points = np.asarray(ply.points, dtype=dtype)
    if ply.has_colors():
        colors = np.asarray(ply.colors)
        if color_format == 'packed':
            point_colors = pack_colors(colors[:, 0] * 255, colors[:, 1] * 255, colors[:, 2] * 255)
        elif color_format == 'float':
            point_colors = colors.astype(dtype)
    else:
        point_colors = None
    if get_normals:
        if ply.has_normals():
            normals = np.asarray(ply.normals, dtype=dtype)
        else:
            normals = None
        return points, point_colors, normals
    else:
        return points, point_colors


# def _pymesh_load_points(points, color_format='packed', get_normals=False):
#     ply = pymesh.load_mesh(points)
#     points = ply.vertices.astype(np.float32)
#     if 'vertex_red' in ply.attribute_names:
#         r, g, b = (ply.get_attribute(c).astype(np.uint8) for c in ['vertex_red', 'vertex_green', 'vertex_blue'])
#         if color_format == 'packed':
#             point_colors = pack_colors(r, g, b)
#         elif color_format == 'float':
#             point_colors = np.stack([r / 255, g / 255, b / 255], axis=-1)
#     else:
#         point_colors = None
#     if get_normals:
#         if 'vertex_nx' in ply.attribute_names:
#             normals = np.stack([ply.get_attribute(c) for c in ['vertex_nx', 'vertex_ny', 'vertex_nz']], axis=-1)
#         else:
#             normals = None
#         return points, point_colors, normals
#     else:
#         return points, point_colors
