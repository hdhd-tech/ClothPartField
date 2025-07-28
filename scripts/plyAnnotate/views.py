import os
import json
import numpy as np
from PIL import Image
import shutil
from typing import List, Dict, Tuple, Any

# --- 辅助类型定义 ---
ImageRGB = np.ndarray # (H, W, 3)

class DatasetElem:
    """
    存储单个数据集元素，包含原始图像和标注图像。
    
    Attributes:
        uid (str): 数据的唯一标识符。
        raw_img (ImageRGB): 原始的RGB图像数据，是一个 (H, W, 3) 的Numpy数组。
        label_img (ImageRGB): 标注图像数据，是一个 (H, W, 3) 的Numpy数组。
    """
    def __init__(self, 
                 uid: str,
                 raw_img: ImageRGB,
                 label_img: ImageRGB,
                ):
        """初始化一个DatasetElem实例。"""
        self.uid = uid
        self.raw_img = raw_img
        self.label_img = label_img

    def to_dict(self) -> Dict[str, Any]:
        """将除了图像之外的元数据转换为字典，方便保存为JSON（如果需要的话）。"""
        # 在这个场景下，每个元素的唯一元数据就是其UID，这里仅作示例
        return {
            "uid": self.uid,
        }

def save_image_rgb(img_np: np.ndarray, save_path: str):
    """
    将RGB NumPy数组保存为图像文件。
    期望img_np是 (H, W, 3) 格式的，值范围0-1。
    """
    img_to_save = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_to_save).save(save_path)


class CustomDatasetManager:
    """
    管理合成数据集的类。
    
    这个类负责收集数据集元素（DatasetElem）并将其持久化到磁盘，
    按照JPEGImages和Annotations的结构组织。
    """
    def __init__(self, dataset_name: str):
        """
        初始化一个CustomDatasetManager实例。
        
        Args:
            dataset_name (str): 数据集的名称，将作为顶级文件夹的名称。
        """
        self.dataset_name = dataset_name
        self.data_elems: List[DatasetElem] = []
        print(f"初始化数据集管理器，数据集名称: '{self.dataset_name}'")

    def add_data_elem(self, data_elem: DatasetElem):
        """
        新增一个DatasetElem到当前实例中。
        
        Args:
            data_elem (DatasetElem): 一个新的数据集元素实例。
        """
        self.data_elems.append(data_elem)
        print(f"成功为数据集 '{self.dataset_name}' 添加了UID为 '{data_elem.uid}' 的数据元素。")

    def save_to_disk(self, base_dir: str):
        """
        将当前CustomDatasetManager维护的所有数据保存到本地文件夹。
        
        这包括：
        1. 在 base_dir 下创建一个以 dataset_name 命名的主文件夹。
        2. 在主文件夹内创建 'JPEGImages' 和 'Annotations' 两个子文件夹。
        3. 对于每个 DatasetElem：
            a. 在 'JPEGImages' 下创建一个以其UID命名的子文件夹，保存原始图像（JPG格式）。
            b. 在 'Annotations' 下创建一个以其UID命名的子文件夹，保存标注图像（PNG格式）。
        
        Args:
            base_dir (str): 指定的基文件夹路径，所有数据集将被保存在这里。
        """
        dataset_folder = os.path.join(base_dir, self.dataset_name)
        os.makedirs(dataset_folder, exist_ok=True)
        print(f"数据集 '{self.dataset_name}' 将被保存到: {dataset_folder}")

        jpeg_images_dir = os.path.join(dataset_folder, 'JPEGImages')
        annotations_dir = os.path.join(dataset_folder, 'Annotations')
        
        os.makedirs(jpeg_images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        print(f"已创建 '{jpeg_images_dir}' 和 '{annotations_dir}' 文件夹。")

        for elem in self.data_elems:
            # 创建原始图像的保存路径
            raw_img_sub_dir = os.path.join(jpeg_images_dir, elem.uid)
            os.makedirs(raw_img_sub_dir, exist_ok=True)
            raw_img_path = os.path.join(raw_img_sub_dir, f"00000.jpg")
            
            # 创建标注图像的保存路径
            label_img_sub_dir = os.path.join(annotations_dir, elem.uid)
            os.makedirs(label_img_sub_dir, exist_ok=True)
            label_img_path = os.path.join(label_img_sub_dir, f"00000.png") # 标注图通常用PNG以保留无损信息

            # 保存原始图像
            save_image_rgb(elem.raw_img, raw_img_path)
            
            # 保存标注图像
            save_image_rgb(elem.label_img, label_img_path)
            
            print(f"  - 已保存UID为 '{elem.uid}' 的图像和标注到对应文件夹。")

        print(f"--- 数据集 '{self.dataset_name}' 保存完成 ---")
        
        self.data_elems = []