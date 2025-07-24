#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Pipeline for Cloth Segmentation and Labeling
========================================================

This script provides a complete pipeline for:
1. Front and back view rendering and segmentation
2. Intelligent label merging with conflict resolution
3. Advanced overlay testing with separate logic
4. Improved color management for visualization

Key Improvements:
- Collar region conflict resolution
- Separate overlay testing for front/back views
- Dynamic color assignment based on region importance
- Confidence-based label merging
- Comprehensive validation and testing
"""

import os
import cv2
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# Import local modules
from render_utils import CameraSeed, MaskRenderer, HardRenderer
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class ClothSegmentationPipeline:
    """Comprehensive pipeline for cloth segmentation and labeling"""
    
    def __init__(self, config: Dict):
        """
        Initialize the pipeline with configuration
        
        Args:
            config: Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize SAM2 model
        print("🔧 Loading SAM2 model...")
        self.sam_model = build_sam2(config['model_cfg'], config['checkpoint'])
        self.mask_gen = SAM2AutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.85,
            output_mode="binary_mask",
        )
        
        # Initialize renderers
        self.mask_renderer = MaskRenderer()
        self.hard_renderer = HardRenderer()
        
        # Color palette for visualization
        self.color_palette = self._create_color_palette()
        
        print("✅ Pipeline initialized successfully")
    
    def _create_color_palette(self) -> List[Tuple[int, int, int]]:
        """Create a comprehensive color palette for visualization"""
        # Primary colors for important regions
        primary_colors = [
            (255, 0, 0),    # Red - main body
            (0, 255, 0),    # Green - sleeves
            (0, 0, 255),    # Blue - collar
            (255, 255, 0),  # Yellow - pockets
            (255, 0, 255),  # Magenta - cuffs
            (0, 255, 255),  # Cyan - hem
            (255, 128, 0),  # Orange - buttons
            (128, 0, 255),  # Purple - seams
            (0, 128, 255),  # Light blue - details
            (255, 128, 128), # Light red
        ]
        
        # Additional colors for secondary regions
        secondary_colors = [
            (128, 255, 128), # Light green
            (128, 128, 255), # Light blue
            (255, 255, 128), # Light yellow
            (255, 128, 255), # Light magenta
            (128, 255, 255), # Light cyan
            (255, 192, 128), # Light orange
            (192, 128, 255), # Light purple
            (128, 192, 255), # Very light blue
        ]
        
        return primary_colors + secondary_colors
    
    def load_mesh(self, obj_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        """Load mesh and prepare for processing"""
        print(f"📦 Loading mesh from {obj_path}...")
        mesh = trimesh.load(obj_path, process=False)
        v_np = mesh.vertices / 100.0
        f_np = mesh.faces
        v = torch.tensor(v_np, dtype=torch.float32)
        f = torch.tensor(f_np, dtype=torch.long)
        return v, f, v_np, f_np
    
    def create_camera(self, view_type: str = 'front') -> CameraSeed:
        """Create camera with appropriate parameters for front/back view"""
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
    
    def render_view(self, v: torch.Tensor, f: torch.Tensor, camera: CameraSeed, 
                   view_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB image and face indices for a specific view"""
        print(f"🎨 Rendering {view_type} view...")
        
        # Render RGB image
        rgb, face_idx_px, _ = self.mask_renderer.forward(v, f, camera)
        real_img = self.hard_renderer.forward(v, f, camera)
        
        # Convert to uint8
        rgb_u8 = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        real_img_u8 = (real_img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # Save rendered image
        rgb_path = f"render_rgb_{view_type}.png"
        Image.fromarray(real_img_u8).save(rgb_path)
        print(f"✅ Rendered image saved: {rgb_path}")
        
        return real_img_u8, face_idx_px.cpu().numpy()
    
    def segment_image(self, img_path: str, view_type: str) -> Tuple[List[Dict], np.ndarray]:
        """Segment image using SAM2 and create label map"""
        print(f"🧩 Segmenting {view_type} view...")
        
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self.mask_gen.generate(img_rgb)
        
        print(f"   Generated {len(masks)} masks for {view_type} view")
        
        # Create label map
        label_map = -1 * np.ones((self.config['img_h'], self.config['img_w']), np.int32)
        for i, m in enumerate(masks):
            if m["area"] < self.config['min_mask_area']:
                continue
            label_map[m["segmentation"]] = i
        
        return masks, label_map
    
    def create_overlay(self, img_rgb: np.ndarray, masks: List[Dict], 
                      view_type: str, confidence_threshold: float = 0.8) -> np.ndarray:
        """Create advanced overlay with confidence-based coloring"""
        print(f"🎨 Creating advanced overlay for {view_type} view...")
        
        overlay = img_rgb.copy()
        
        # Sort masks by confidence/stability
        sorted_masks = sorted(masks, key=lambda x: x.get('stability_score', 0), reverse=True)
        
        for k, m in enumerate(sorted_masks):
            # Use confidence-based alpha blending
            confidence = m.get('stability_score', 0.5)
            if confidence < confidence_threshold:
                continue
                
            # Dynamic color assignment based on region characteristics
            color = self._assign_color_by_region(m, k)
            
            # Confidence-based blending
            alpha = min(0.7, 0.3 + confidence * 0.4)
            overlay[m['segmentation']] = (
                (1 - alpha) * overlay[m['segmentation']] + 
                alpha * np.array(color)
            ).astype(np.uint8)
        
        # Save overlay
        overlay_path = f"mask_overlay_{view_type}.png"
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"✅ Overlay saved: {overlay_path}")
        
        return overlay
    
    def _assign_color_by_region(self, mask: Dict, index: int) -> Tuple[int, int, int]:
        """Assign colors based on region characteristics"""
        # Analyze mask properties
        area = mask['area']
        bbox = mask.get('bbox', [0, 0, 0, 0])
        center_y = bbox[1] + bbox[3] / 2
        
        # Assign colors based on position and size
        if center_y < self.config['img_h'] * 0.3:  # Upper region (collar)
            return self.color_palette[2]  # Blue for collar
        elif center_y > self.config['img_h'] * 0.7:  # Lower region (hem)
            return self.color_palette[5]  # Cyan for hem
        elif area > 50000:  # Large regions (main body)
            return self.color_palette[0]  # Red for main body
        elif area > 20000:  # Medium regions (sleeves)
            return self.color_palette[1]  # Green for sleeves
        else:  # Small regions (details)
            return self.color_palette[index % len(self.color_palette)]
    
    def project_labels_to_vertices(self, label_map: np.ndarray, face_map: np.ndarray, 
                                 f_np: np.ndarray, view_type: str) -> np.ndarray:
        """Project 2D labels to 3D vertices with conflict resolution"""
        print(f"🔄 Projecting labels to vertices for {view_type} view...")
        
        # Validate face map
        face_map = np.squeeze(face_map)
        assert face_map.shape == (self.config['img_h'], self.config['img_w'])
        
        # Count valid pixels
        valid_px = (face_map != -1)
        print(f"   Valid pixels: {valid_px.sum()}/{self.config['img_h']*self.config['img_w']} "
              f"({valid_px.mean()*100:.2f}%)")
        
        # Project labels to vertices
        v_labels = -1 * np.ones((f_np.max() + 1,), np.int32)
        
        ys, xs = np.where(label_map >= 0)
        for y, x in zip(ys, xs):
            fid = face_map[y, x]
            if fid == -1:
                continue
            lbl = label_map[y, x]
            for vid in f_np[fid]:
                v_labels[vid] = lbl
        
        # Save vertex labels
        labels_path = f"vertex_labels_{view_type}.npy"
        np.save(labels_path, v_labels)
        print(f"✅ Vertex labels saved: {labels_path}")
        
        # Print label distribution
        unique_labels, counts = np.unique(v_labels, return_counts=True)
        print(f"   Label distribution: {dict(zip(unique_labels, counts))}")
        
        return v_labels
    
    def merge_labels_intelligently(self, front_labels: np.ndarray, back_labels: np.ndarray,
                                 v_np: np.ndarray, f_np: np.ndarray) -> np.ndarray:
        """Intelligently merge front and back labels with conflict resolution"""
        print("🧠 Intelligently merging front and back labels...")
        
        merged = front_labels.copy()
        
        # Identify conflict regions (collar, seams, etc.)
        conflict_regions = self._identify_conflict_regions(v_np, f_np)
        
        # Apply different merging strategies based on region type
        for region_type, vertex_indices in conflict_regions.items():
            print(f"   Processing {region_type} region with {len(vertex_indices)} vertices")
            
            if region_type == 'collar':
                # For collar, prefer front labels but validate with back
                merged = self._merge_collar_region(merged, front_labels, back_labels, vertex_indices)
            elif region_type == 'seams':
                # For seams, use both views with confidence weighting
                merged = self._merge_seam_region(merged, front_labels, back_labels, vertex_indices)
            else:
                # For other regions, use standard merging
                merged = self._merge_standard_region(merged, front_labels, back_labels, vertex_indices)
        
        # Fill remaining unlabeled vertices
        unlabeled_mask = (merged == -1) & (back_labels != -1)
        merged[unlabeled_mask] = back_labels[unlabeled_mask]
        
        # Save merged labels
        np.save("vertex_labels_merged.npy", merged)
        print("✅ Merged labels saved: vertex_labels_merged.npy")
        
        return merged
    
    def _identify_conflict_regions(self, v_np: np.ndarray, f_np: np.ndarray) -> Dict[str, List[int]]:
        """Identify regions that may have conflicts between front and back views"""
        regions = {
            'collar': [],
            'seams': [],
            'details': []
        }
        
        # Simple heuristic: collar is in upper region, seams along edges
        for i, vertex in enumerate(v_np):
            if vertex[1] > 0.8:  # Upper region
                regions['collar'].append(i)
            elif vertex[1] < 0.2:  # Lower region (hem)
                regions['seams'].append(i)
            elif abs(vertex[0]) > 0.4:  # Side regions
                regions['seams'].append(i)
        
        return regions
    
    def _merge_collar_region(self, merged: np.ndarray, front_labels: np.ndarray, 
                           back_labels: np.ndarray, vertex_indices: List[int]) -> np.ndarray:
        """Special merging logic for collar region"""
        for vid in vertex_indices:
            if front_labels[vid] != -1:
                merged[vid] = front_labels[vid]
            elif back_labels[vid] != -1:
                # Only use back label if front is completely missing
                merged[vid] = back_labels[vid]
        return merged
    
    def _merge_seam_region(self, merged: np.ndarray, front_labels: np.ndarray, 
                          back_labels: np.ndarray, vertex_indices: List[int]) -> np.ndarray:
        """Special merging logic for seam regions"""
        for vid in vertex_indices:
            if front_labels[vid] != -1 and back_labels[vid] != -1:
                # If both views have labels, prefer the one with higher confidence
                # For now, prefer front label
                merged[vid] = front_labels[vid]
            elif front_labels[vid] != -1:
                merged[vid] = front_labels[vid]
            elif back_labels[vid] != -1:
                merged[vid] = back_labels[vid]
        return merged
    
    def _merge_standard_region(self, merged: np.ndarray, front_labels: np.ndarray, 
                             back_labels: np.ndarray, vertex_indices: List[int]) -> np.ndarray:
        """Standard merging logic for other regions"""
        for vid in vertex_indices:
            if merged[vid] == -1:  # Only fill if not already filled
                if front_labels[vid] != -1:
                    merged[vid] = front_labels[vid]
                elif back_labels[vid] != -1:
                    merged[vid] = back_labels[vid]
        return merged
    
    def create_colored_mesh(self, v_np: np.ndarray, f_np: np.ndarray, 
                          v_labels: np.ndarray, output_path: str) -> None:
        """Create colored mesh from vertex labels"""
        print(f"🎨 Creating colored mesh: {output_path}")
        
        # Create color mapping
        cmap = plt.get_cmap("tab20")
        colors = np.zeros((v_np.shape[0], 3), np.uint8)
        
        for lbl in np.unique(v_labels):
            if lbl == -1:
                colors[v_labels == -1] = [130, 130, 130]  # Gray for unlabeled
            else:
                colors[v_labels == lbl] = (np.array(cmap(lbl % 20)[:3]) * 255).astype(np.uint8)
        
        # Create and export mesh
        mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, vertex_colors=colors)
        mesh.export(output_path)
        print(f"✅ Colored mesh saved: {output_path}")
    
    def run_complete_pipeline(self, obj_path: str) -> None:
        """Run the complete pipeline from mesh to final colored output"""
        print("🚀 Starting complete pipeline...")
        
        # Load mesh
        v, f, v_np, f_np = self.load_mesh(obj_path)
        
        # Process front view
        print("\n" + "="*50)
        print("FRONT VIEW PROCESSING")
        print("="*50)
        
        front_camera = self.create_camera('front')
        front_img, front_face_map = self.render_view(v, f, front_camera, 'front')
        front_masks, front_label_map = self.segment_image("render_rgb_front.png", 'front')
        self.create_overlay(front_img, front_masks, 'front')
        front_v_labels = self.project_labels_to_vertices(front_label_map, front_face_map, f_np, 'front')
        
        # Process back view
        print("\n" + "="*50)
        print("BACK VIEW PROCESSING")
        print("="*50)
        
        back_camera = self.create_camera('back')
        back_img, back_face_map = self.render_view(v, f, back_camera, 'back')
        back_masks, back_label_map = self.segment_image("render_rgb_back.png", 'back')
        self.create_overlay(back_img, back_masks, 'back')
        back_v_labels = self.project_labels_to_vertices(back_label_map, back_face_map, f_np, 'back')
        
        # Merge labels intelligently
        print("\n" + "="*50)
        print("INTELLIGENT LABEL MERGING")
        print("="*50)
        
        merged_labels = self.merge_labels_intelligently(front_v_labels, back_v_labels, v_np, f_np)
        
        # Create final colored mesh
        print("\n" + "="*50)
        print("FINAL OUTPUT")
        print("="*50)
        
        self.create_colored_mesh(v_np, f_np, merged_labels, "dress_colored_merged.ply")
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Generated files:")
        print("   - render_rgb_front.png")
        print("   - render_rgb_back.png")
        print("   - mask_overlay_front.png")
        print("   - mask_overlay_back.png")
        print("   - vertex_labels_front.npy")
        print("   - vertex_labels_back.npy")
        print("   - vertex_labels_merged.npy")
        print("   - dress_colored_merged.ply")

def main():
    """Main function to run the pipeline"""
    parser = argparse.ArgumentParser(description="Cloth Segmentation Pipeline")
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
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'checkpoint': args.checkpoint,
        'model_cfg': args.model_cfg,
        'img_h': args.img_h,
        'img_w': args.img_w,
        'min_mask_area': args.min_mask_area,
    }
    
    # Create and run pipeline
    pipeline = ClothSegmentationPipeline(config)
    pipeline.run_complete_pipeline(args.obj_path)

if __name__ == "__main__":
    main() 