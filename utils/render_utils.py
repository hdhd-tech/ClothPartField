import torch
import torch.nn as nn
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    MeshRenderer,
    TexturesUV,
    TexturesVertex,
    BlendParams,
)

from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from pytorch3d.renderer.mesh.shader import ShaderBase

DEVICE = 'cpu'

class SimpleShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        
        quantized_bary_coords = torch.zeros_like(fragments.bary_coords).to(device=fragments.bary_coords.device)
        max_indices = torch.argmax(fragments.bary_coords, dim=-1, keepdim=True)
        quantized_bary_coords.scatter_(-1, max_indices, 1.0)

        # 获取每个像素的面索引
        # fragments.idx 的形状通常是 (N, H, W, K)，其中 K 是每个像素重叠的面数（通常取第一个）
        # 如果你只需要最前面的面，可以直接取 fragments.idx[..., 0]
        pixel_face_indices = fragments.pix_to_face[..., 0] # (N, H, W)

        # 你可以选择返回图像和面索引，或者只返回面索引，这取决于你的后续需求。
        # 这里我们假设你可能需要同时获取渲染图像和面索引。
        # 你可以将面索引作为额外的信息返回，例如作为元组或字典。
        return images, pixel_face_indices, quantized_bary_coords

class CameraSeed:
    def __init__(
        self,
        translation: list,
        focal: list,
        principal_point: list,
        target_pos: list,
        img_hw: list,
    ):
        """
        初始化 CameraSeed 对象。

        Args:
            translation (list): 相机在世界坐标系中的位置 [x, y, z]。
            focal (list): 相机焦距 [fx, fy]。
            principal_point (list): 相机主点 [px, py]。
            target_pos (list): 相机看向的目标点 [x, y, z]。
            img_hw (list): 图像高度和宽度 [height, width]。
        """

        self.device = DEVICE

        self.translation = torch.tensor(translation, dtype=torch.float32, device=self.device)
        self.focal = torch.tensor(focal, dtype=torch.float32, device=self.device)
        self.principal_point = torch.tensor(principal_point, dtype=torch.float32, device=self.device)
        self.img_hw = img_hw  # img_hw 不需要是 tensor，保持list或tuple

        # 如果提供了 target_pos，则计算 R 和 T
        if target_pos is not None and self.translation is not None:
            _target_pos = torch.tensor(
                target_pos, dtype=torch.float32, device=self.device
            )
            self.R, self.T = look_at_view_transform(
                eye=self.translation.unsqueeze(0), at=_target_pos.unsqueeze(0)
            )
        else:
            self.R = None
            self.T = None
        self.target_pos = target_pos

        # PyTorch3D 相机对象
        self.pytorch3d_camera = self._create_pytorch3d_camera()

    def _create_pytorch3d_camera(self):
        """
        根据内部参数创建 PyTorch3D 的 PerspectiveCameras 对象。
        """
        # 如果缺少关键参数，返回 None 或抛出错误
        if self.focal is None or self.R is None or self.T is None:
            raise ValueError("创建PyTorch3D相机所需的焦距、旋转或平移信息不完整。")

        # 如果未提供 principal_point，则根据 img_hw 计算图像中心作为主点
        current_principal_point = self.principal_point
        if current_principal_point is None and self.img_hw is not None:
            current_principal_point = torch.tensor(
                [self.img_hw[1] * 0.5, self.img_hw[0] * 0.5],
                dtype=torch.float32,
                device=self.device,
            )
        elif current_principal_point is None and self.img_hw is None:
            raise ValueError(
                "无法计算主点：未提供 principal_point 且 img_hw 也未提供。"
            )

        # PyTorch3D 的 cameras 期望 fx, fy 分别作为焦距，px, py 作为主点
        # 并且 R 和 T 应该是批处理维度 (N, 3, 3) 和 (N, 3)
        # 确保它们有批处理维度
        self.R_batch = self.R.unsqueeze(0) if self.R.dim() == 2 else self.R
        self.T_batch = self.T.unsqueeze(0) if self.T.dim() == 1 else self.T
        focal_batch = self.focal.unsqueeze(0) if self.focal.dim() == 1 else self.focal
        principal_point_batch = (
            current_principal_point.unsqueeze(0)
            if current_principal_point.dim() == 1
            else current_principal_point
        )

        return PerspectiveCameras(
            focal_length=focal_batch,
            principal_point=principal_point_batch,
            R=self.R_batch,
            T=self.T_batch,
            in_ndc=False,
            image_size=(
                torch.tensor([self.img_hw], dtype=torch.float32, device=self.device)
                if self.img_hw
                else None
            ),
            device=self.device,
        )

    def get_pytorch3d_camera(self):
        """
        返回 PyTorch3D 的相机对象。
        """
        return self.pytorch3d_camera

class MaskRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = DEVICE

        # 定义光栅化设置
        raster_settings = RasterizationSettings(
            blur_radius=0,  
            faces_per_pixel=1,
            cull_backfaces=False,  # 禁用背面剔除，方便渲染薄片状物体如衣服
            # bin_size=10,
        )

        # 初始化相机，实际的相机参数会在 forward 中更新
        self.cameras = PerspectiveCameras(device=self.device)

        # 初始化光栅器
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=raster_settings
        )
        
        # 初始化混合参数，用于设置背景颜色
        blend_params = BlendParams(
            background_color=[0.0, 0.0, 0.0]
        )

        # 初始化光源
        self.lights = PointLights(
            device=self.device,
            location=torch.tensor(
                [[1.0, 5.0, 3.0]], device=self.device
            ),
        )

        self.vertex_shader = SimpleShader(blend_params=blend_params)

        # 组合光栅器和着色器来创建渲染器
        self.mesh_renderer = MeshRenderer(
            rasterizer=self.rasterizer, shader=self.vertex_shader
        )

        self.to(self.device)

    def to(self, device):
        self.device = device
        self.cameras.to(device)
        self.rasterizer.to(device)
        self.lights.to(device)
        self.vertex_shader.to(device)
        self.mesh_renderer.to(device)

    def forward(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        camera_seed: CameraSeed,
        verts_rgb=None,
    ):
        """
        Args:
            vertices (torch.Tensor): 形状为 (B, N_v, 3) 的顶点张量。
            faces (torch.Tensor): 形状为 (B, N_f, 3) 的面张量。
            camera_seed (CameraSeed): 包含相机参数和（可选的）顶点颜色的 CameraSeed 对象。

        Returns:
            torch.Tensor: 渲染出的图像，形状为 (B, H, W, 3)，包含 RGB 通道。
        """
        device = self.device

        # 从 CameraSeed 获取 PyTorch3D 相机对象
        pytorch3d_camera = camera_seed.get_pytorch3d_camera()
        if pytorch3d_camera is None:
            raise ValueError(
                "无法从 CameraSeed 获取有效的 PyTorch3D 相机对象。请检查 CameraSeed 参数。"
            )

        # 更新渲染器和着色器中的相机对象
        self.rasterizer.cameras = pytorch3d_camera

        img_size_hw = (camera_seed.img_hw[0], camera_seed.img_hw[1])
        self.rasterizer.raster_settings.image_size = img_size_hw

        if verts_rgb is None:
            # 如果没有提供顶点颜色，使用默认的灰色
            _verts = vertices[0] if vertices.ndim == 3 else vertices
            verts_rgb = torch.ones_like(_verts)[None] * torch.tensor(
                [0.8, 0.5, 0.5], device=device
            )  # 灰色
        else:
            verts_rgb = verts_rgb.to(device)  # 确保颜色在正确的设备上

        textures = TexturesVertex(verts_features=verts_rgb)

        # 创建 Meshes 对象
        # 注意：PyTorch3D 的 Meshes 期望 verts 和 faces 作为列表，即使只有一个网格
        meshes = Meshes(verts=[vertices], faces=[faces], textures=textures)

        # 调用渲染器进行渲染
        images, _pixel_map, quantized_bary_coords = self.mesh_renderer(
            meshes, cameras=pytorch3d_camera, lights=self.lights
        )

        return images[0, ..., :3], _pixel_map, quantized_bary_coords

class HardRenderer(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = DEVICE
    

        # sigma = 1e-4

        # 定义光栅化设置
        raster_settings = RasterizationSettings(
            blur_radius= 0, #np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=32,
            cull_backfaces=False, # 禁用背面剔除，方便渲染薄片状物体如衣服
            bin_size=0
        )

        # 初始化相机，实际的相机参数会在 forward 中更新
        self.cameras = PerspectiveCameras(device=self.device)

        # 初始化光栅器
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)

        # 初始化光源
        self.lights = PointLights(device=self.device, location=torch.tensor([[1.0, 5.0, 3.0]], device=self.device))

        # 初始化混合参数，用于设置背景颜色
        blend_params = BlendParams(background_color=[0.0, 0.0, 0.0])

        # 初始化 SoftPhongShader
        self.phong_shader = HardPhongShader(
            device=self.device,
            cameras=self.cameras, # shader 也需要相机信息
            lights=self.lights,   # shader 需要光源信息
            blend_params=blend_params, # 将混合参数传入 shader
        )

        # 组合光栅器和着色器来创建渲染器
        self.mesh_renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.phong_shader
        )

        self.to(self.device)

    def to(self, device):
        self.device = device
        self.cameras.to(device)
        self.rasterizer.to(device)
        self.lights.to(device)
        self.phong_shader.to(device)
        self.mesh_renderer.to(device)

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor, camera_seed: CameraSeed, verts_rgb = None):
        """
        Args:
            vertices (torch.Tensor): 形状为 (B, N_v, 3) 的顶点张量。
            faces (torch.Tensor): 形状为 (B, N_f, 3) 的面张量。
            camera_seed (CameraSeed): 包含相机参数和（可选的）顶点颜色的 CameraSeed 对象。

        Returns:
            torch.Tensor: 渲染出的图像，形状为 (B, H, W, 3)，包含 RGB 通道。
        """
        device = DEVICE

        # 从 CameraSeed 获取 PyTorch3D 相机对象
        pytorch3d_camera = camera_seed.get_pytorch3d_camera()
        if pytorch3d_camera is None:
            raise ValueError("无法从 CameraSeed 获取有效的 PyTorch3D 相机对象。请检查 CameraSeed 参数。")

        # 更新渲染器和着色器中的相机对象
        self.rasterizer.cameras = pytorch3d_camera
        self.phong_shader.cameras = pytorch3d_camera

        img_size_hw = (camera_seed.img_hw[0], camera_seed.img_hw[1]) 
        self.rasterizer.raster_settings.image_size = img_size_hw

        if verts_rgb is None:
            # 如果没有提供顶点颜色，使用默认的灰色
            _verts = vertices[0] if vertices.ndim == 3 else vertices
            verts_rgb = torch.ones_like(_verts)[None] * torch.tensor([0.8, 0.5, 0.5], device=device) # 灰色
        else:
            verts_rgb = verts_rgb.to(device) # 确保颜色在正确的设备上

        textures = TexturesVertex(verts_features=verts_rgb)

        # 创建 Meshes 对象
        # 注意：PyTorch3D 的 Meshes 期望 verts 和 faces 作为列表，即使只有一个网格
        meshes = Meshes(verts=[vertices], faces=[faces], textures=textures)

        # 调用渲染器进行渲染
        images = self.mesh_renderer(meshes, cameras=pytorch3d_camera, lights=self.lights)

        # images 的形状为 (B, H, W, 4)，包含 RGBA
        # return  # 返回 RGB 通道，去除 Alpha

        return images[0, ..., :3]