import base64
import json
import xmlrpc.client
from pathlib import Path

# 读取并 base64 编码参考图像
ref_img_path = Path("../debug/debug_masks/1_rendered.png")
with open(ref_img_path, "rb") as f_img:
    reference_image = base64.b64encode(f_img.read()).decode("ascii")
    print(f"已加载reference_image")

specification_path = Path("../outputs/sim_json/test_spec.json")
with open(specification_path, "r") as f_json:
    specification_json = json.load(f_json)
    print(f"已加载specification_json")

print("开始调用仿真服务...")
rpc_client = xmlrpc.client.ServerProxy("http://localhost:9081")
result = rpc_client.run_simulation(specification_json, reference_image)
# 判断微服务返回内容
if isinstance(result, dict) and "error" in result.keys():
    # 只有error字段，直接返回0分和错误信息
    print(f"仿真服务返回错误: {result['error']}")
    raise ValueError(f"仿真服务返回错误: {result['error']}")

# 否则按原有方式解包
elif isinstance(result, (list, tuple)) and len(result) == 4:
    loss, _, sim_image, static_image = result
    print(f"仿真服务返回loss: {loss}")

    assert sim_image, "sim_image is None"
    assert static_image, "static_image is None"

    def get_base64_bytes(data):
        import xmlrpc.client

        if isinstance(data, xmlrpc.client.Binary):
            data = data.data
        if isinstance(data, str):
            return base64.b64decode(data)
        elif isinstance(data, bytes):
            return base64.b64decode(data)
        else:
            return None

    img_data = get_base64_bytes(sim_image)
    static_img_data = get_base64_bytes(static_image)

    sim_image_path = Path("../outputs/sim_outputs/sim_image.png")
    static_image_path = Path("../outputs/sim_outputs/static_image.png")

    with open(sim_image_path, "wb") as f_img:
        f_img.write(img_data)
        print(f"成功保存仿真图片到: {sim_image_path}")
    with open(static_image_path, "wb") as f_img:
        f_img.write(static_img_data)
        print(f"成功保存静态图片到: {static_image_path}")
else:
    raise ValueError("仿真结果解析失败")