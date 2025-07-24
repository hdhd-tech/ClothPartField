from sympy.core.facts import FactRules
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Import local modules
from render_utils import CameraSeed, MaskRenderer, HardRenderer
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class ClothSegmentationPipeline:    
    def __init__(self, config: Dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 初始化SAM2模型
        print("🔧 Loading SAM2 model...")
        self.sam_model = build_sam2(config['model_cfg'], config['checkpoint'])
        self.mask_gen = SAM2AutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.7,
            output_mode="binary_mask",
        )
        
        # 初始化渲染器
        self.mask_renderer = MaskRenderer()
        self.hard_renderer = HardRenderer()

    def _create_palette(self, k):
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(i) for i in np.linspace(0, 1, k)]
        return colors

    def load_mesh(self, obj_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """加载网格"""
        print(f"📦 Loading mesh from {obj_path}...")
        mesh = trimesh.load_mesh(obj_path, process=False)

        if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
            vertices = mesh.vertices / 100.0
            faces = mesh.faces
        else:
            raise ValueError(f"Mesh from {obj_path} does not have vertices and faces attributes")
        v = torch.tensor(vertices, dtype=torch.float32)
        f = torch.tensor(faces, dtype=torch.long)
        return v, f, vertices, faces

    def create_camera(self, view_type: str = 'front') -> CameraSeed:
        """创建相机"""
        img_h, img_w = self.config['img_h'], self.config['img_w']
        
        focal = [20000, 20000]
        principal = [img_w / 2, img_h / 2]
        target_pos = [0.0, 1.0, 0.0]
        img_hw = [img_h, img_w]
        
        if view_type == 'front':
            translation = [0.0, 1.0, 28.0]
        elif view_type == 'back':
            translation = [0.0, 1.0, -28.0]
        else:
            raise ValueError(f"Unknown view type: {view_type}")
        
        camera = CameraSeed(translation, focal, principal, target_pos, img_hw)
        print(f"📷 Created {view_type} camera with translation: {translation}")
        return camera

    def label_faces_by_camera(self, v_np: np.ndarray, f_np: np.ndarray, camera: 'CameraSeed') -> np.ndarray:
        """
        根据相机位置和法线方向为每个面分配标签
        返回: labels, shape=(num_faces,), front_label/back_label/-1
        """
        # 计算每个面的法线
        v0 = v_np[f_np[:, 0]]
        v1 = v_np[f_np[:, 1]]
        v2 = v_np[f_np[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

        # 相机朝向向量 (target - position)
        camera_pos = np.array(camera.translation)
        camera_target = np.array(camera.target_pos)
        camera_dir = camera_target - camera_pos
        camera_dir = camera_dir / (np.linalg.norm(camera_dir) + 1e-8)

        # 计算法线与相机朝向的点积
        dot_ = np.dot(face_normals, camera_dir)

        # 正面为正，背面为负
        labels = dot_ < 0

        return labels


    def label_faces_by_camera_sam(self, vertices: torch.Tensor, faces: torch.Tensor, camera: 'CameraSeed') -> List[np.ndarray]:
        """
        用渲染+SAM2分割辅助判断正反面，返回: labels, shape=(num_faces,), front_label/back_label/-1
        """
        # 1. 渲染 mesh 得到图片
        img, face_idx_map, _ = self.mask_renderer.forward(
            vertices,
            faces,
            camera
        )
        rendered_img = self.hard_renderer.forward(
            vertices,
            faces,
            camera
        )

        rendered_img = rendered_img.cpu().numpy()
        
        masks = self.mask_gen.generate(rendered_img)
        masks = self.filter_sam_masks(masks, min_area=self.config['min_mask_area'])
        masks = [m["segmentation"] for m in masks]
        print(f"🎨 Found {len(masks)} masks")

        import os
        from PIL import Image

        # 创建输出目录
        os.makedirs("debug_masks", exist_ok=True)

        # 可视化渲染图像
        rendered_vis = (rendered_img * 255).astype(np.uint8)
        Image.fromarray(rendered_vis).save(f"debug_masks/{camera.translation[2]:.0f}_rendered.png")

        # 可视化每个mask
        for i, m in enumerate(masks):
            mask_img = np.zeros_like(rendered_img[..., 0], dtype=np.uint8)
            mask_img[m] = 255
            plt.imsave(f"debug_masks/{camera.translation[2]:.0f}_mask_{i}.png", mask_img, cmap='gray', vmin=0, vmax=255)

            # 若想将 mask 叠加在渲染图上：
            overlay = rendered_vis.copy()
            overlay[m] = [255, 0, 0]  # 红色标出 mask 区域
            Image.fromarray(overlay).save(f"debug_masks/{camera.translation[2]:.0f}_overlay_{i}.png")

        face_direction_mask = self.label_faces_by_camera(
            vertices.cpu().numpy(), faces.cpu().numpy(), camera
        )

        mask_vertices = [
            self._pixels_to_vertices(
                faces=faces.cpu().numpy(),
                segmentation=m,
                face_idx_map=face_idx_map,
                face_idx_filter=face_direction_mask
            )
            for m in masks
        ]

        # filter by min_mask_area again
        mask_vertices = [m for m in mask_vertices if len(m) > self.config['min_vertex_count']]
        
        return mask_vertices

    def filter_sam_masks(self, masks: List[Dict], min_area: int) -> List[Dict]:
        """过滤SAM2分割结果"""
        masks = [m for m in masks if m['area'] > min_area]
        m2g = {i: i for i in range(len(masks))}
        g2m = {i: [i] for i in range(len(masks))}
        # check iou between masks
        for i in range(len(masks)):
            for j in range(i+1, len(masks)):
                group_i = m2g[i]
                group_j = m2g[j]
                if group_i == group_j:
                    continue
                iou = np.sum(masks[i]['segmentation'] & masks[j]['segmentation']) / np.sum(masks[i]['segmentation'] | masks[j]['segmentation'])
                if iou > 0.7:
                    # merge group_j into group_i
                    for mask_id in g2m[group_j]:
                        m2g[mask_id] = group_i
                    g2m[group_i].extend(g2m[group_j])
                    del g2m[group_j]

        # filter masks by groups
        filtered_masks = []
        for group_id in g2m:
            group_masks = [masks[mask_id] for mask_id in g2m[group_id]]
            segmentation = np.zeros_like(group_masks[0]['segmentation'])
            for mask in group_masks:
                segmentation |= mask['segmentation']
            filtered_masks.append({
                'segmentation': segmentation,
                'area': np.sum(segmentation)
            })
        return filtered_masks

    def _pixels_to_vertices(self, faces: np.ndarray, segmentation: np.ndarray, face_idx_map: np.ndarray, face_idx_filter: np.ndarray) -> np.ndarray:
        """将像素映射到面"""
        # map pixels to face ids
        fidxs = np.unique(face_idx_map[0][segmentation.nonzero()].reshape(-1))
        # remove -1
        fidxs = fidxs[fidxs > 0]
        # keep idxs that have face_idx_filter[idx] == True
        fidxs = fidxs[face_idx_filter[fidxs]]
        vidxs = faces[fidxs]
        vidxs = np.unique(vidxs.reshape(-1))

        return vidxs

    
    def save_labels(self, labels: np.ndarray, out_path: str):
        """保存标签"""
        print(f"📂 Saving labels to {out_path}...")
        np.save(out_path, labels)

        
    def run_complete_pipeline(self, obj_path: str) -> None:
        v,f, vertices, faces = self.load_mesh(obj_path)
        cameras = [
            self.create_camera(view_type=view_type)
            for view_type in ['front', 'back']
        ]
        masks = []
        for camera in cameras:
            mask_vertices = self.label_faces_by_camera_sam(
                vertices = v,
                faces = f,
                camera = camera
            )
            masks.extend(mask_vertices)
        print(f"🎨 Found {len(masks)} masks")

        labels_expanded = np.zeros((vertices.shape[0], len(masks)), dtype=np.bool)
        for i, mask in enumerate(masks):
            labels_expanded[mask, i] = 1

        labels = np.ones(vertices.shape[0], dtype=np.int8) * -1
        _mask = labels_expanded.sum(axis=1) == 1
        labels[_mask] = np.argmax(labels_expanded[_mask], axis=1)
        self.save_labels(labels, "labels.npy")
        print(f"🎨 Saved labels to labels.npy")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Improved Cloth Segmentation Pipeline")
    parser.add_argument("--obj_path", type=str, default="shirt.obj", 
                       help="Path to the OBJ file")
    parser.add_argument("--checkpoint", type=str, 
                       default="/root/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_CLOTH_finetune.yaml/checkpoints/checkpoint.pt",
                       help="Path to SAM2 checkpoint")
    parser.add_argument("--model_cfg", type=str, 
                       default="configs/sam2.1/sam2.1_hiera_b+.yaml",
                       help="Path to SAM2 model config")
    parser.add_argument("--img_h", type=int, default=1200, help="Image height")
    parser.add_argument("--img_w", type=int, default=800, help="Image width")
    parser.add_argument("--min_mask_area", type=int, default=2000, 
                       help="Minimum mask area threshold")
    parser.add_argument("--min_vertex_count", type=int, default=100, 
                       help="Minimum vertex count threshold")
    args = parser.parse_args()
    
    # 配置
    config = {
        'checkpoint': args.checkpoint,
        'model_cfg': args.model_cfg,
        'img_h': args.img_h,
        'img_w': args.img_w,
        'min_mask_area': args.min_mask_area,
        'min_vertex_count': args.min_vertex_count,
    }
    
    # 创建并运行管道
    pipeline = ClothSegmentationPipeline(config)
    pipeline.run_complete_pipeline(args.obj_path)


if __name__ == "__main__":
    main()