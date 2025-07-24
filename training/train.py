import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import trimesh
import matplotlib.pyplot as plt

# ---------- ① 读取数据 ----------
X_all = np.load("data/embeddings/shirt/part_feat_shirt_0_batch.npy")      # (N, D)
y_all = np.load("data/labels/labels.npy")              # (N,)  -1 表示未标注
unique, counts = np.unique(y_all, return_counts=True)
for u, c in zip(unique, counts):
    print(f"Label {u}: {c} samples")
points = trimesh.load("data/meshes/shirt.obj")              # (N, 3)
mesh = trimesh.load("data/meshes/shirt.obj", process=False)           # 保持原顺序
V = mesh.vertices                                         # (N, 3)
F = mesh.faces                                            # (M, 3)
points = V                                                # 直接把顶点当作 point cloud

print("X_all:", X_all.shape, " y_all:", y_all.shape, " V:", V.shape, " F:", F.shape, " points:", points.shape)
assert len(X_all) == len(y_all) == len(points), "点数不一致"

# ---------- ② 拆分已标注 / 未标注 ----------
mask_labeled   = y_all != -1
mask_unlabeled = y_all == -1

X_labeled   = X_all[mask_labeled]
y_labeled   = y_all[mask_labeled]
X_unlabeled = X_all[mask_unlabeled]

# ---------- ③ 连续化标签 ----------
le = LabelEncoder()
y_labeled_enc = le.fit_transform(y_labeled)     # 如 [0,2,3,4,5,6,7,8] → [0,1,2,3,4,5,6,7]
num_classes   = len(le.classes_)                # K

# ---------- ④ 训练 / 验证划分 ----------
X_train, X_val, y_train_enc, y_val_enc = train_test_split(
    X_labeled, y_labeled_enc, test_size=0.2, stratify=y_labeled_enc, random_state=42
)


# ---------- ⑤ 标准化 ----------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_unlab_s = scaler.transform(X_unlabeled)

# ---------- ⑥ 自训练 ----------
print("\n✅ 开始自训练")
current_X_train = X_train_s
current_y_train = y_train_enc
current_X_unlab = X_unlab_s

threshold = 0.5
max_iter  = 15
min_add   = 8

for it in range(max_iter):
    print(f"\n--- 迭代 {it+1}/{max_iter} ---")
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_classes,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    clf.fit(current_X_train, current_y_train)

    # 验证集
    y_val_pred = np.argmax(clf.predict_proba(X_val_s), axis=1)
    print("验证准确率:", accuracy_score(y_val_enc, y_val_pred))

    if len(current_X_unlab) == 0:
        break

    # 未标注样本高置信度筛选
    probs = clf.predict_proba(current_X_unlab)
    max_p = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    hi_idx = np.where(max_p >= threshold)[0]

    if len(hi_idx) < min_add and it > 0:
        break

    current_X_train = np.vstack([current_X_train, current_X_unlab[hi_idx]])
    current_y_train = np.hstack([current_y_train, preds[hi_idx]])
    current_X_unlab = np.delete(current_X_unlab, hi_idx, axis=0)

# ---------- ⑦ 训练最终模型 ----------
final_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=num_classes,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
final_clf.fit(current_X_train, current_y_train)

# ---------- ⑧ 预测未标注点 ----------
y_unlab_pred_enc = final_clf.predict(X_unlab_s)
y_unlab_pred     = le.inverse_transform(y_unlab_pred_enc)   # 反映射回原编号

# ---------- ⑨ 合并全部标签 ----------
y_final = np.copy(y_all)
y_final[mask_unlabeled] = y_unlab_pred
np.save("outputs/labels/vertex_labels_full.npy", y_final.astype(np.int32))
print("✔ 已保存 vertex_labels_full.npy")
combined = np.concatenate([X_all, y_final.reshape(-1, 1)], axis=1)  # (N, D+1)
np.save("outputs/labels/embeddings/embedding_with_labels.npy", combined)
print("✔ 已保存 embedding_with_labels.npy")




# ---------- ⑩ 上色并保存 PLY ----------
# colors = (plt.get_cmap("tab20")(y_final % 20)[:, :3] * 255).astype(np.uint8)
# colored_pcd = trimesh.points.PointCloud(points, colors=colors)
# colored_pcd.export("cloth/dress_colored_pred.ply")

# print("\n🎉 已保存上色文件: cloth/dress_colored_pred.ply")
colors = (plt.get_cmap("tab20")(y_final % 20)[:, :3] * 255).astype(np.uint8)

# 重新组装带颜色的网格
mesh_colored = trimesh.Trimesh(vertices=V,
                               faces=F,
                               vertex_colors=colors,
                               process=True)

mesh_colored.export("outputs/colored_meshes_outputs/dress_colored_vert.ply")

print("已保存: outputs/colored_meshes_outputs/dress_colored_vert.ply")