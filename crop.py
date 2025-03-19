import cv2
import numpy as np


# 基于规则 (Rule-Based): 裁剪的逻辑是基于一组预定义的规则，而不是通过学习（例如，训练一个目标检测模型）得到的
# 去除边缘边框 (固定像素数)。
# 找到像素值大于阈值 (20) 的区域。
# 找到最大的连通区域。
# 根据最大连通区域的边界框进行裁剪。
# 简单 (Simple): 这些规则相对简单，易于理解和实现。它们不涉及复杂的算法或模型。
# 裁剪 (Crop): 最终目的是从原始图像中提取出一个子区域（感兴趣区域），即裁剪操作。

def fit_image(fname):
    X = cv2.imread(fname)

    # 去除边缘边框
    X = X[5:-5, 5:-5]

    # 连通组件分析
    output = cv2.connectedComponentsWithStats((X > 20).astype(np.uint8)[:, :, 0], 8, cv2.CV_32S)
    stats = output[2]

    # 找到最大连通区域
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h

    # 裁剪图像
    X_fit = X[y1:y2, x1:x2]

    # 调整大小为 2:1 (1024 x 512)
    X_resized = cv2.resize(X_fit, (512, 1024))

    im_id = 'test'  # 可以根据需要修改
    print(f'图片裁剪并调整大小后的尺寸: {X_resized.shape}')
    cv2.imwrite(f'{im_id}2.png', X_resized[:, :, 0]) #只保存一个通道