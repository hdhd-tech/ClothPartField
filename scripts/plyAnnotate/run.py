import colorsys
import os
import random
from plyfile import PlyData, PlyElement
import numpy as np
from typing import List, Set
import torch
from PIL import Image
from render_utils import CameraSeed, PhongRenderer, MaskRenderer
from views import DatasetElem, CustomDatasetManager
import uuid
from tqdm import tqdm

NUM_SAMPLE_PER_CLOTH = 3
focal = [20000, 20000]
principal = [800 / 2, 1200 / 2]
target_pos = [0.0, 1.0, 0.0]
img_hw = [1200, 800]  # 假设主点在成像中心

current_file_path = os.path.abspath(__file__)

# INPUT_DIR = os.path.join(os.path.dirname(current_file_path), 'input')
INPUT_DIR = '/home/yaobao/workspace/haoda/pipeline/GarmentDataAnotator/GarmentCodeData/downloaded_garments/garments_5000_0/default_body'
OUTPUT_DIR = os.path.join(os.path.dirname(current_file_path), 'output')

target_list = [
    name for name in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, name))
]

target_list = target_list[:-1200]

print(f"Found {len(target_list)} target directories in {INPUT_DIR}.")

input_list = [f'{e}/{e}_sim' for e in target_list]

# input_list = ['rand_0A36YXPNV0_sim']

ply_files_path_list = [os.path.join(INPUT_DIR, e+'.ply') for e in input_list]
seg_files_path_list = [os.path.join(INPUT_DIR, e+'_segmentation.txt') for e in input_list]


TEXT_DIR = '/home/yaobao/workspace/haoda/pipeline/GarmentDataAnotator/GarmentCodeData/Textures/textures_pinterest' # 里面有很多JPG
# 获取 TEXT_DIR 下所有文件（仅文件，不包括子目录）
texture_files = [
    f for f in os.listdir(TEXT_DIR)
    if os.path.isfile(os.path.join(TEXT_DIR, f))
]
print(f"Found {len(texture_files)} texture files in {TEXT_DIR}.")

def get_random_texture(texture_files):
    """
    从给定的纹理文件列表中随机抽取一个纹理文件名。

    Args:
        texture_files (List[str]): 纹理文件名列表。

    Returns:
        str: 随机选中的纹理文件名。
    """
    if not texture_files:
        raise ValueError("纹理文件列表为空。")
    return random.choice(texture_files)

def rgb_to_class(rgb_img, color_map, threshold=0.0, background_class_idx=0):
    """
    高效地将RGB图像根据颜色图转换为类别索引图。

    此函数通过计算每个像素与颜色图中“最接近”的颜色，将其分配到对应的类别。
    如果像素与所有颜色的距离都超出了阈值，则将其划分为背景。

    Args:
        rgb_img (np.ndarray): 输入的RGB图像，形状为 (H, W, 3)，像素值范围应为 0-1。
        color_map (dict): 一个将类别名称(str)映射到RGB元组(0-255)的字典。
                          例如: {'panel_A': (255, 0, 0), 'panel_B': (0, 0, 255)}
        threshold (int): 判定颜色是否匹配的最大欧氏距离。在此距离内的才算有效匹配。
        background_class_idx (int): 用于背景或未匹配像素的类别索引，默认为 0。

    Returns:
        np.ndarray: 一个形状为 (H, W) 的 int32 类型类别索引图。
    """
    # 步骤 1: 验证输入并准备数据
    rgb_img *= 255
    
    # 将颜色图的键（panel名称）排序，以确保每次运行的类别ID分配一致
    panel_names = sorted(color_map.keys())
    
    # 从1开始为每个panel分配类别ID，0保留给背景
    # 这样可以从根本上解决“类别0”与“背景0”的冲突问题
    class_ids = np.arange(1, len(panel_names) + 1, dtype=np.int32)
    
    # 将颜色图中的RGB值转换为一个NumPy数组，形状为 (N, 3)，N为类别数
    colors_arr = np.array([color_map[name]*255 for name in panel_names], dtype=np.float32)

    # 步骤 2: 向量化距离计算
    # 将输入图像的数据类型转为float32以进行精确计算
    # 使用NumPy广播机制一次性计算所有像素到所有目标颜色的平方欧氏距离
    # 维度变化:
    #   rgb_img:    (H, W, 3)      -> (H, W, 1, 3)
    #   colors_arr: (N, 3)         -> (1, 1, N, 3)
    #   distances_sq: (H, W, N)
    distances_sq = np.sum((rgb_img.astype(np.float32)[:, :, np.newaxis, :] - colors_arr[np.newaxis, np.newaxis, :, :])**2, axis=3)

    # 步骤 3: 找出最近的颜色并应用阈值
    # 找到每个像素距离最近的颜色索引（沿N个颜色的轴）
    # closest_color_indices 的形状为 (H, W)
    closest_color_indices = np.argmin(distances_sq, axis=2)
    
    # 使用这些索引从 class_ids 数组中获取最终的类别ID
    class_img = class_ids[closest_color_indices]
    
    # 步骤 4: 处理未匹配的像素（背景）
    # 获取每个像素到其最近颜色的最小距离值
    min_distances = np.sqrt(np.min(distances_sq, axis=2))
    
    # 创建一个遮罩，标记那些与最近颜色的距离仍然大于阈值的像素
    background_mask = min_distances > threshold
    
    # 将这些像素的类别设置为背景索引
    class_img[background_mask] = background_class_idx
    
    return class_img

def rand_translation():
    RADIUS = 28.0
    fix_y = 1.0
    angle = random.uniform(0, 2 * np.pi)
    x = int(round(np.cos(angle) * RADIUS))
    z = int(round(np.sin(angle) * RADIUS))
    return [x, fix_y, z]
    
    # return [0.0, 1.0, 28.0]

def read_ply_file(file_path):
    """
    使用 plyfile 库读取 PLY 文件并提取点和面数据。

    Args:
        file_path (str): PLY 文件的路径。

    Returns:
        tuple: 包含 (points, faces) 的元组。
               points: NumPy 数组，形状为 (N, 3) 或 (N, K)，K 是属性数量。
               faces: NumPy 数组或列表，表示面的连接信息。
    """
    try:
        # 读取 PLY 文件
        plydata = PlyData.read(file_path)
        print(f"成功读取 PLY 文件: {file_path}")

        # --- 提取顶点数据 ---
        # 假设顶点数据在 'vertex' 元素中
        if 'vertex' in plydata:
            vertex_data = plydata['vertex']
            # 将顶点属性（x, y, z 等）转换为结构化 NumPy 数组
            # 或者转换为普通的 N x 3 数组（如果只有 x, y, z）
            points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
            uv = np.vstack([vertex_data['s'], vertex_data['t']]).T

        else:
            points = None
            uv = None
            print("文件中没有找到 'vertex' 元素。")

        # --- 提取面数据 ---
        # 假设面数据在 'face' 元素中
        if 'face' in plydata:
            face_data = plydata['face']
            # 面数据通常是一个列表，每个元素代表一个面，内部又是一个列表表示顶点的索引
            # 例如：[(4, [0, 1, 3, 2]), (4, [4, 5, 7, 6]), ...]
            # 需要提取其第二个元素 (顶点的索引列表)
            faces = np.array([f[0] for f in face_data.data])
            # print("前5个面:\n", faces[:5])
        else:
            faces = None
            print("文件中没有找到 'face' 元素。")
            
        return points, faces, uv

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在。")
        return None, None
    except Exception as e:
        print(f"读取 PLY 文件时发生错误: {e}")
        return None, None

def read_stitch_categories(file_path: str) -> List[Set[str]]:
    """
    读取指定格式的文本文件，将每行的类别信息转换为一个集合，
    并返回一个包含所有这些集合的列表。

    文件格式示例:
    stitch_0
    stitch_0,stitch_3,stitch_5
    stitch_1

    Args:
        file_path (str): 待读取的文本文件路径。

    Returns:
        List[Set[str]]: 一个列表，其中每个元素是一个集合，
                       包含对应行（或索引）的所有类别字符串。
                       如果文件不存在或读取失败，返回空列表。
    """
    categories_list: List[Set[str]] = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 移除行首尾的空白符（包括换行符）
                clean_line = line.strip()

                if clean_line:  # 确保行不为空
                    # 使用逗号分割字符串，得到一个类别字符串的列表
                    # 例如 "stitch_0,stitch_3" 会变成 ["stitch_0", "stitch_3"]
                    # "stitch_0" 会变成 ["stitch_0"]
                    category_strings = [s.strip() for s in clean_line.split(',')]

                    # 将类别字符串列表转换为一个集合
                    # 使用集合的好处是自动去重，且查找效率高
                    categories_set = set(category_strings)
                    categories_list.append(categories_set)
                else:
                    # 如果行是空的，我们可能想跳过或者添加一个空集合，取决于具体需求
                    # 这里选择跳过空行
                    pass

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请检查路径是否正确。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

    return categories_list
    
def save_obj(vertices: np.ndarray, faces: np.ndarray, file_path: str):
    """
    将顶点和面数据保存为 OBJ 文件。

    Args:
        vertices (np.ndarray): 顶点数组，形状为 (N, 3)。
        faces (np.ndarray): 面数组，形状为 (M, K)，K 通常为 3（三角面）或 4（四边面）。
        file_path (str): 输出 OBJ 文件路径。
    """
    with open(file_path, 'w') as f:
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 写入面（OBJ 索引从 1 开始）
        for face in faces:
            # 支持三角面和多边面
            idxs = [str(idx + 1) for idx in face]
            f.write(f"f {' '.join(idxs)}\n")

def merge_duplicate_vertices(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray) -> tuple:
    """
    Merges duplicate vertices in a 3D mesh and updates face indices accordingly,
    while preserving the relative order of the first occurrence of each unique vertex.
    Also merges the corresponding uv coordinates.

    Args:
        vertices (np.ndarray): Array of shape (N, 3) for vertex coordinates.
        faces (np.ndarray): Array of shape (M, K) for face indices.
        uv (np.ndarray): Array of shape (N, 2) for uv coordinates.

    Returns:
        tuple: (unique_vertices, new_faces, unique_uv)
    """
    if vertices.shape[1] != 3:
        raise ValueError("Vertices array must have a shape of (N, 3) for (x, y, z) coordinates.")
    if faces.dtype != np.int_:
        faces = faces.astype(np.int_)
        print("Warning: Faces array converted to integer type.")
    if uv.shape[0] != vertices.shape[0]:
        raise ValueError("UV array must have the same number of rows as vertices.")

    unique_vertex_map = {}
    new_vertex_list = []
    new_uv_list = []
    old_to_new_index_map = np.zeros(len(vertices), dtype=int)

    current_new_index = 0
    for i, vert in enumerate(vertices):
        vert_tuple = tuple(vert.tolist())
        if vert_tuple not in unique_vertex_map:
            unique_vertex_map[vert_tuple] = current_new_index
            new_vertex_list.append(vert)
            new_uv_list.append(uv[i])
            old_to_new_index_map[i] = current_new_index
            current_new_index += 1
        else:
            old_to_new_index_map[i] = unique_vertex_map[vert_tuple]

    unique_vertices = np.array(new_vertex_list)
    unique_uv = np.array(new_uv_list)
    new_faces = old_to_new_index_map[faces]

    return unique_vertices, new_faces, unique_uv

class Garment():
    def __init__(self, v, f, seg_sets):
        self.vertices = v
        self.faces = f
        self.seg_sets = seg_sets
        self.merged_class = self.merge_stitch_elements(Set.union(*seg_sets))
        self.dye_panel()

    def merge_stitch_elements(self, input_set: set) -> set:
        """
        将输入集合中所有以 'stitch_' 开头的元素合并为一个新的元素 'all_stitches'。

        Args:
            input_set (set): 包含字符串元素的原始集合。

        Returns:
            set: 合并后的新集合。
        """
        new_set = set()
        stitch_elements_present = False # 标志位，看是否有stitch_开头的元素

        for item in input_set:
            if item.startswith('stitch_'):
                stitch_elements_present = True
            else:
                new_set.add(item)

        if stitch_elements_present:
            new_set.add('all_stitches') # 如果存在stitch_开头的元素，则添加合并后的标识

        return new_set
    
    def _dispatch_a_color(self, p_idx, PANEL_NUM) -> np.ndarray:
        """
        为面板分配一个高对比度的颜色，使用黄金分割率在色相环上取色。
        这种方法能确保相邻索引的颜色也具有良好的视觉区分度。

        Args:
            p_idx (int): 当前面板的索引 (从 0 开始)。
            PANEL_NUM (int): 面板的总数 (此方法中该参数主要用于边界检查)。

        Returns:
            np.ndarray: 一个代表RGB颜色的Numpy数组，范围 [0, 255]。
        """
        if PANEL_NUM <= 0:
            return np.array([0, 0, 0])
        if p_idx >= PANEL_NUM:
            raise ValueError("p_idx 必须小于 PANEL_NUM")
            
        # 黄金分割率的共轭数，约为 0.618
        GOLDEN_RATIO_CONJUGATE = 0.61803398875

        # 1. 使用黄金分割率计算色相(H)，并用模1运算确保其在 [0, 1) 范围内
        hue = (p_idx * GOLDEN_RATIO_CONJUGATE) % 1

        # 2. 为了进一步增加区分度，可以交替改变饱和度和明度
        if p_idx % 2 == 0:
            saturation = 0.95
            value = 0.9
        else:
            saturation = 0.9
            value = 0.95

        # 3. 从 HSV 转换到 RGB (浮点数)
        rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)

        # 4. 映射到 [0, 255] 整数范围
        # rgb_int = [int(c * 255) for c in rgb_float]
        
        # 5. 返回 Numpy 数组
        return np.array(rgb_float)

    def dye_panel(self):
        PANEL_NUM = len(self.merged_class)
        # 在 rgb空间中 为每个panel分配一个颜色
        self.color_map = {}
        for p_idx, panel_name in enumerate(self.merged_class):
            self.color_map[panel_name] = self._dispatch_a_color(p_idx, PANEL_NUM)
        
        # 初始化 self.v_color 是一个numpy array，形状为（V，3） V == len(self.vertices)
        self.v_color = np.zeros((len(self.vertices), 3))
        
        for v_idx, panel_class in enumerate(self.seg_sets):
            panel_class = list(panel_class)
            if len(panel_class) != 1 or panel_class[0].startswith('stitch_'):
                color = [0,0,0] #self.color_map['all_stitches']
            else:
                assert len(panel_class) == 1
                color = self.color_map[panel_class[0]]
            self.v_color[v_idx] = color

def save_img(img_np, img_save_path):
    img_to_save = (img_np * 255).astype(np.uint8)
    if img_to_save.shape[0] == 3:  # (C, H, W) -> (H, W, C)
        img_to_save = np.transpose(img_to_save, (1, 2, 0))
    Image.fromarray(img_to_save).save(img_save_path)
    
phongRenderer = PhongRenderer()
maskRenderer = MaskRenderer()

dataset_manager = CustomDatasetManager(dataset_name="new_CustomGarmentDataset_label_with_texture_2") #new_CustomGarmentDataset

for idx, (mesh_path, seg_path) in tqdm(enumerate(zip(ply_files_path_list, seg_files_path_list)), total=len(ply_files_path_list), desc="Processing meshes"):
    mesh = read_ply_file(mesh_path)
    merged_mesh = merge_duplicate_vertices(mesh[0], mesh[1], mesh[2])
    v, f, uv = merged_mesh
    seg = read_stitch_categories(seg_path)
    
    g = Garment(v, f, seg)
    v_tensor = torch.tensor(g.vertices).to(device='cuda') / 100
    f_tensor = torch.tensor(g.faces).to(device='cuda')
    c_tensor = torch.tensor(g.v_color).to(device='cuda').to(dtype=torch.float32)
    
    text_path = os.path.join(TEXT_DIR, get_random_texture(texture_files))
    if not os.path.exists(text_path):
        print(f"Texture file {text_path} does not exist, skipping this garment.")
        continue
    # 读取纹理图片为RGB numpy数组
    texture_img = Image.open(text_path).convert('RGB')
    texture_np = np.array(texture_img) / 255.0  # 归一化到0-1

    # uv坐标归一化到纹理尺寸
    uv_normalized = uv.copy()
    uv_normalized[:, 0] = uv[:, 0] * (texture_np.shape[1] - 1)
    uv_normalized[:, 1] = (1 - uv[:, 1]) * (texture_np.shape[0] - 1)  # 注意y轴方向

    uv_int = np.round(uv_normalized).astype(int)
    uv_int = np.clip(uv_int, 0, [texture_np.shape[1] - 1, texture_np.shape[0] - 1])

    # 查询每个顶点的纹理颜色
    text_c_np = texture_np[uv_int[:, 1], uv_int[:, 0], :]  # shape: (V, 3)
    text_c_tensor = torch.tensor(text_c_np).to(device='cuda').to(dtype=torch.float32)
    
    for _ in range(NUM_SAMPLE_PER_CLOTH):
        rdm_trans = rand_translation()
        camera = CameraSeed(rdm_trans, focal, principal, target_pos, img_hw)
        
        rgb_mask = (
            maskRenderer.forward(v_tensor, f_tensor, camera, c_tensor[None])
            .detach()
            .cpu()
            .numpy()
        )
        real_img = (
            phongRenderer.forward(v_tensor, f_tensor, camera, text_c_tensor[None]).detach().cpu().numpy()
        )
        
        class_img = rgb_to_class(rgb_mask, g.color_map, 2, 0)
        # print("class_img unique values:", np.unique(class_img))
        
        uid = uuid.uuid4().hex[:10]
        tmp_elem = DatasetElem(uid, real_img, class_img)
        dataset_manager.add_data_elem(tmp_elem)
    
    if idx%5 == 0 :
        dataset_manager.save_to_disk(OUTPUT_DIR)