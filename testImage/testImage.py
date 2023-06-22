import cv2
import numpy as np

gt = cv2.imread("testImage/VT5000_gt_75.jpg")
rgb = cv2.imread("testImage/VT5000_rgb_75.jpg")
# it = cv2.imread("./testImage/VT821_t_20.jpg")


def visualize_heatmap(image, saliency_map):
    # 归一化显著性分数图像到0到1的范围
    normalized_map = (saliency_map - saliency_map.min()) / \
        (saliency_map.max() - saliency_map.min())

    # 应用颜色映射，将归一化后的显著性分数图像转换为热力图
    heatmap = cv2.applyColorMap(
        (normalized_map * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 可选：将热力图与原始图像叠加，以便更好地理解目标的位置和边界
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    # 显示热力图
    cv2.imshow("Heatmap", heatmap)
    cv2.imshow("overlay", overlay)
    # cv2.imwrite("testImage/VT821_heatmap_20.jpg", heatmap)
    cv2.imwrite("testImage/VT5000_heatmaponrgb_75.jpg", overlay)



# 将图像转换为浮点型
image = gt.astype(np.float32) / 255.0

# 应用双边滤波
smoothed = cv2.bilateralFilter(image, 128, 256, 16)

# 将图像还原为8位无符号整数类型
smoothed = (smoothed * 255).astype('uint8')

cv2.imshow('Original Image', gt)
cv2.imshow('Blurred Image', smoothed)
visualize_heatmap(rgb, smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()

