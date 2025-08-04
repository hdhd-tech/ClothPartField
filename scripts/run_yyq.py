import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import sys
from pathlib import Path
import os
# 将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sam2_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sam2'))
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)
# Import local modules
from utils.render_utils import CameraSeed, MaskRenderer, HardRenderer
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
            original_vertices = mesh.vertices
            faces = mesh.faces
            
            # Print original mesh info
            orig_bounds = {
                'min': np.min(original_vertices, axis=0),
                'max': np.max(original_vertices, axis=0),
                'center': np.mean(original_vertices, axis=0),
                'size': np.max(original_vertices, axis=0) - np.min(original_vertices, axis=0)
            }
            print(f"📏 Original mesh bounds: center={orig_bounds['center']}, size={orig_bounds['size']}")
            
            # Apply configurable scaling
            mesh_scale = self.config.get('mesh_scale', 0.01)  # Default to 0.01 instead of 0.01 (1/100)
            vertices = original_vertices * mesh_scale
            print(f"🔧 Applied mesh scaling: {mesh_scale}")
        else:
            raise ValueError(f"Mesh from {obj_path} does not have vertices and faces attributes")
        
        # Calculate mesh bounds and center for camera positioning
        self.mesh_bounds = {
            'min': np.min(vertices, axis=0),
            'max': np.max(vertices, axis=0),
            'center': np.mean(vertices, axis=0),
            'size': np.max(vertices, axis=0) - np.min(vertices, axis=0)
        }
        
        print(f"📏 Scaled mesh bounds: center={self.mesh_bounds['center']}, size={self.mesh_bounds['size']}")
        
        v = torch.tensor(vertices, dtype=torch.float32)
        f = torch.tensor(faces, dtype=torch.long)
        return v, f, vertices, faces

    def create_camera(self, view_type: str = 'front', mesh_bounds: dict = None) -> CameraSeed:
        """创建相机 - 自适应网格大小和位置"""
        img_h, img_w = self.config['img_h'], self.config['img_w']
        
        # Use mesh bounds if available, otherwise use stored bounds
        bounds = mesh_bounds if mesh_bounds is not None else getattr(self, 'mesh_bounds', None)
        
        if bounds is None:
            # Fallback to original hardcoded values
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
        else:
            # Adaptive camera positioning based on mesh bounds
            center = bounds['center']
            size = bounds['size']
            
            # Calculate appropriate distance based on mesh size
            # Use the maximum dimension to ensure the entire mesh is visible
            max_dim = np.max(size)
            
            # Improved distance calculation for small meshes
            # Use configurable parameters with better defaults for small meshes
            padding_factor = self.config.get('camera_distance_factor', 0.5)
            distance_scale = self.config.get('camera_distance_scale', 10.0)
            
            # For very small meshes, use a different calculation approach
            if max_dim < 0.1:  # Very small mesh
                # Use a base distance that works well for small meshes
                distance = max(max_dim * 100 * padding_factor, 1.0)
            else:
                # Use the original calculation for normal-sized meshes
                distance = max_dim * padding_factor * distance_scale
            
            # Ensure reasonable distance bounds
            distance = max(min(distance, 100.0), 0.5)  # Clamp between 0.5 and 100
            
            focal = [20000, 20000]
            principal = [img_w / 2, img_h / 2]
            target_pos = center.tolist()  # Look at mesh center
            img_hw = [img_h, img_w]
            
            if view_type == 'front':
                # Position camera in front of the mesh (positive Z direction from center)
                translation = [center[0], center[1], center[2] + distance]
            elif view_type == 'back':
                # Position camera behind the mesh (negative Z direction from center)
                translation = [center[0], center[1], center[2] - distance]
            elif view_type == 'left':
                # Position camera to the left of the mesh (negative X direction from center)
                translation = [center[0] - distance, center[1], center[2]]
            elif view_type == 'right':
                # Position camera to the right of the mesh (positive X direction from center)
                translation = [center[0] + distance, center[1], center[2]]
            elif view_type == 'top':
                # Position camera above the mesh (positive Y direction from center)
                translation = [center[0], center[1] + distance, center[2]]
            elif view_type == 'bottom':
                # Position camera below the mesh (negative Y direction from center)
                translation = [center[0], center[1] - distance, center[2]]
            else:
                raise ValueError(f"Unknown view type: {view_type}")
        
        camera = CameraSeed(translation, focal, principal, target_pos, img_hw)
        print(f"📷 Created {view_type} camera with translation: {translation}, target: {target_pos}")
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
        
        # Check if mesh is visible in the rendered image
        img_min, img_max = rendered_img.min(), rendered_img.max()
        non_zero_pixels = np.sum(rendered_img > 0.01)  # Count non-background pixels
        print(f"   🖼️  Rendered image stats: min={img_min:.3f}, max={img_max:.3f}, non-zero pixels={non_zero_pixels}")
        
        if non_zero_pixels < 100:  # Very few pixels, mesh might not be visible
            print(f"   ⚠️  Warning: Only {non_zero_pixels} visible pixels, mesh might be too small or camera too far")
        
        masks = self.mask_gen.generate(rendered_img)
        print(f"   🎭 SAM2 generated {len(masks)} raw masks")
        
        masks = self.filter_sam_masks(masks, min_area=self.config['min_mask_area'])
        masks = [m["segmentation"] for m in masks]
        print(f"   🎨 After filtering: {len(masks)} masks (min_area={self.config['min_mask_area']})")

        import os
        from PIL import Image

        # 创建输出目录
        os.makedirs("../debug/debug_masks", exist_ok=True)

        # 可视化渲染图像
        rendered_vis = (rendered_img * 255).astype(np.uint8)
        Image.fromarray(rendered_vis).save(f"../debug/debug_masks/{camera.translation[2]:.0f}_rendered.png")

        # 可视化每个mask
        for i, m in enumerate(masks):
            mask_img = np.zeros_like(rendered_img[..., 0], dtype=np.uint8)
            mask_img[m] = 255
            plt.imsave(f"../debug/debug_masks/{camera.translation[2]:.0f}_mask_{i}.png", mask_img, cmap='gray', vmin=0, vmax=255)

            # 若想将 mask 叠加在渲染图上：
            overlay = rendered_vis.copy()
            overlay[m] = [255, 0, 0]  # 红色标出 mask 区域
            Image.fromarray(overlay).save(f"../debug/debug_masks/{camera.translation[2]:.0f}_overlay_{i}.png")

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
        mask_vertices_filtered = [m for m in mask_vertices if len(m) > self.config['min_vertex_count']]
        print(f"   ✂️  After vertex count filtering: {len(mask_vertices_filtered)} masks (min_vertex_count={self.config['min_vertex_count']})")
        
        return mask_vertices_filtered

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

    def analyze_mesh_orientation(self, vertices: np.ndarray, faces: np.ndarray) -> Dict[str, np.ndarray]:
        """
        分析网格朝向，尝试自动确定前后面
        返回推荐的相机朝向
        """
        # 计算所有面的法线和面积
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        face_cross = np.cross(v1 - v0, v2 - v0)
        face_areas = np.linalg.norm(face_cross, axis=1) / 2.0  # Real face areas
        face_normals = face_cross / (np.linalg.norm(face_cross, axis=1, keepdims=True) + 1e-8)
        
        # 计算面的中心点
        face_centers = (v0 + v1 + v2) / 3.0
        
        # 分析法线在各个方向上的分布
        # 统计朝向各个方向的面的数量和面积
        directions = {
            'front_z': np.array([0, 0, 1]),   # 正Z方向
            'back_z': np.array([0, 0, -1]),   # 负Z方向
            'left_x': np.array([-1, 0, 0]),   # 负X方向
            'right_x': np.array([1, 0, 0]),   # 正X方向
            'top_y': np.array([0, 1, 0]),     # 正Y方向
            'bottom_y': np.array([0, -1, 0])  # 负Y方向
        }
        
        orientation_scores = {}
        for name, direction in directions.items():
            # 计算法线与方向的点积
            dots = np.dot(face_normals, direction)
            # 只考虑朝向该方向的面 (点积 > 0.3 表示角度小于约70度)
            aligned_faces = dots > 0.3
            
            if np.any(aligned_faces):
                # 使用预计算的面积
                total_area = np.sum(face_areas[aligned_faces])
                orientation_scores[name] = {
                    'total_area': total_area,
                    'face_count': np.sum(aligned_faces),
                    'avg_dot': np.mean(dots[aligned_faces])
                }
            else:
                orientation_scores[name] = {
                    'total_area': 0,
                    'face_count': 0,
                    'avg_dot': 0
                }
        
        print("🧭 Mesh orientation analysis:")
        for name, scores in orientation_scores.items():
            print(f"  {name}: area={scores['total_area']:.6f}, faces={scores['face_count']}, avg_dot={scores['avg_dot']:.3f}")
        
        return orientation_scores
    
    def get_optimal_camera_views(self, vertices: np.ndarray, faces: np.ndarray) -> List[str]:
        """
        基于网格分析推荐最佳的相机视角
        """
        orientation_scores = self.analyze_mesh_orientation(vertices, faces)
        
        # 找出面积最大的两个相对方向作为主要视角
        z_views = ['front_z', 'back_z']
        x_views = ['left_x', 'right_x']
        y_views = ['top_y', 'bottom_y']
        
        view_pairs = [
            (z_views, 'front', 'back'),
            (x_views, 'left', 'right'),
            (y_views, 'top', 'bottom')
        ]
        
        best_pair = None
        best_score = 0
        
        for pair_views, view1, view2 in view_pairs:
            score1 = orientation_scores[pair_views[0]]['total_area']
            score2 = orientation_scores[pair_views[1]]['total_area']
            total_score = score1 + score2
            
            if total_score > best_score:
                best_score = total_score
                # 选择面积较大的作为主视角
                if score1 >= score2:
                    best_pair = [view1, view2]
                else:
                    best_pair = [view2, view1]
        
        if best_pair is None:
            # 如果分析失败，使用默认的前后视角
            best_pair = ['front', 'back']
            
        print(f"🎯 Recommended camera views: {best_pair}")
        return best_pair

    
    def save_labels(self, labels: np.ndarray, out_path: str):
        """保存标签"""
        print(f"📂 Saving labels to {out_path}...")
        np.save(out_path, labels)

        
    def run_complete_pipeline(self, obj_path: str) -> None:
        # Load mesh and calculate bounds
        v, f, vertices, faces = self.load_mesh(obj_path)
        
        # Determine camera views
        if self.config.get('manual_views'):
            # Use manually specified views
            optimal_views = self.config['manual_views']
            print(f"🎯 Using manual camera views: {optimal_views}")
        elif self.config.get('auto_camera', True):
            # Analyze mesh orientation to determine optimal camera views
            optimal_views = self.get_optimal_camera_views(vertices, faces)
        else:
            # Fallback to default front/back views
            optimal_views = ['front', 'back']
            print(f"🎯 Using default camera views: {optimal_views}")
        
        # Create cameras based on optimal views
        cameras = []
        for view_type in optimal_views:
            camera = self.create_camera(view_type=view_type)
            cameras.append(camera)
        
        print(f"🎥 Using {len(cameras)} camera views: {optimal_views}")
        
        masks = []
        for i, camera in enumerate(cameras):
            print(f"🎬 Processing camera {i+1}/{len(cameras)} ({optimal_views[i]})...")
            mask_vertices = self.label_faces_by_camera_sam(
                vertices=v,
                faces=f,
                camera=camera
            )
            print(f"   Found {len(mask_vertices)} valid masks from this camera")
            masks.extend(mask_vertices)
        
        print(f"🎨 Found {len(masks)} masks across all views")
        
        if len(masks) == 0:
            print("⚠️  No masks found! This might be due to:")
            print("   - Mesh too small/large for current camera settings")
            print("   - SAM2 couldn't detect any segments")
            print("   - min_vertex_count threshold too high")
            print("   - Camera positioned incorrectly")
            print("   Try adjusting --mesh_scale, --camera_distance_factor, or --min_vertex_count")
            
            # Create a simple fallback segmentation based on vertex positions
            print("🔄 Creating fallback segmentation...")
            center = self.mesh_bounds['center']
            vertices_np = v.cpu().numpy()
            
            # Simple segmentation: front/back based on Z coordinate relative to center
            front_mask = vertices_np[:, 2] > center[2]
            back_mask = vertices_np[:, 2] <= center[2]
            
            labels = np.ones(vertices_np.shape[0], dtype=np.int8) * -1
            if np.any(front_mask):
                labels[front_mask] = 0
            if np.any(back_mask):
                labels[back_mask] = 1
                
            print(f"🔄 Fallback segmentation: {np.sum(front_mask)} front vertices, {np.sum(back_mask)} back vertices")
        else:
            # Create labels from masks
            labels_expanded = np.zeros((vertices.shape[0], len(masks)), dtype=bool)
            for i, mask in enumerate(masks):
                labels_expanded[mask, i] = 1

            labels = np.ones(vertices.shape[0], dtype=np.int8) * -1
            _mask = labels_expanded.sum(axis=1) == 1
            labels[_mask] = np.argmax(labels_expanded[_mask], axis=1)
        
        # Save results
        obj_basename = Path(obj_path).stem
        out_dir = Path(__file__).parent.parent / "outputs" / "labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{obj_basename}_labels.npy"

        self.save_labels(labels, str(out_path))
        print(f"🎨 Saved labels to {out_path}")
        
        # Save camera info for reference
        camera_info = {
            'optimal_views': optimal_views,
            'mesh_bounds': self.mesh_bounds,
            'camera_positions': [(cam.translation, cam.target_pos) for cam in cameras],
            'config': {
                'auto_camera': self.config.get('auto_camera', True),
                'manual_views': self.config.get('manual_views'),
                'camera_distance_factor': self.config.get('camera_distance_factor', 1.5),
                'camera_distance_scale': self.config.get('camera_distance_scale', 10.0)
            }
        }
        
        # camera_info_path = out_dir / f"{obj_basename}_camera_info.json"
        # with open(camera_info_path, 'w') as f:
        #     # Convert numpy arrays to lists for JSON serialization
        #     serializable_info = {
        #         'optimal_views': optimal_views,
        #         'mesh_bounds': {
        #             'min': self.mesh_bounds['min'].tolist(),
        #             'max': self.mesh_bounds['max'].tolist(), 
        #             'center': self.mesh_bounds['center'].tolist(),
        #             'size': self.mesh_bounds['size'].tolist()
        #         },
        #         'camera_positions': camera_info['camera_positions'],
        #         'config': camera_info['config']
        #     }
        #     json.dump(serializable_info, f, indent=2)
        # print(f"📋 Saved camera info to {camera_info_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Improved Cloth Segmentation Pipeline")
    parser.add_argument("--obj_path", type=str, default="/home/yang/ClothPartField/data/meshes/newDress.obj", 
                       help="Path to the OBJ file")
    parser.add_argument("--checkpoint", type=str, 
                       default="/data/models/sam/checkpoint_latest.pt",
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
    
    # Camera control arguments
    parser.add_argument("--auto_camera", action="store_true", default=True,
                       help="Automatically determine optimal camera views based on mesh analysis")
    parser.add_argument("--manual_views", type=str, nargs="+", 
                       choices=['front', 'back', 'left', 'right', 'top', 'bottom'],
                       help="Manually specify camera views (overrides auto_camera)")
    parser.add_argument("--camera_distance_factor", type=float, default=1.5,
                       help="Factor to multiply mesh size for camera distance (default: 1.5)")
    parser.add_argument("--camera_distance_scale", type=float, default=10.0,
                       help="Additional scaling factor for camera distance (default: 10.0)")
    
    # Mesh scaling argument
    parser.add_argument("--mesh_scale", type=float, default=0.05,
                       help="Scaling factor for mesh coordinates (default: 0.01)")
    
    args = parser.parse_args()
    
    # 配置
    config = {
        'checkpoint': args.checkpoint,
        'model_cfg': args.model_cfg,
        'img_h': args.img_h,
        'img_w': args.img_w,
        'min_mask_area': args.min_mask_area,
        'min_vertex_count': args.min_vertex_count,
        'camera_distance_factor': args.camera_distance_factor,
        'camera_distance_scale': args.camera_distance_scale,
        'auto_camera': args.auto_camera,
        'manual_views': args.manual_views,
        'mesh_scale': args.mesh_scale,
    }
    
    # 创建并运行管道
    pipeline = ClothSegmentationPipeline(config)
    pipeline.run_complete_pipeline(args.obj_path)


if __name__ == "__main__":
    main()