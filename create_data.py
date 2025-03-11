import cv2
import matplotlib.pyplot as plt
import  torch
import numpy as np
def add_mosaic_to_region(image, top_left, bottom_right, block_size=10):
    """
    给图片的指定区域添加马赛克效果。
    :param image: 输入图片（NumPy数组格式）
    :param top_left: 指定区域左上角坐标 (x1, y1)
    :param bottom_right: 指定区域右下角坐标 (x2, y2)
    :param block_size: 马赛克块的大小
    :return: 加了马赛克的图片
    """

        # 否则，假设数据已经是 PyTorch 张量，直接返回
        # 这里可以直接将其转换为张量，如果不是张量则会抛出类型错误
    x1, y1 = top_left
    x2, y2 = bottom_right

    # 提取指定区域

    newimage = []
    for i in range(image.shape[0]):
       new_image = image[i].numpy()
       numpy_image = np.transpose(new_image, (1, 2, 0))  # 转换为 (H, W, C)
       numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
       roi = numpy_image[y1:y2, x1:x2]
       roi_mosaic = add_mosaic(roi, block_size)
       numpy_image[y1:y2, x1:x2] = roi_mosaic
       resized_image = torch.from_numpy(numpy_image).permute(2, 0, 1)/255
       newimage.append(resized_image.unsqueeze(0))
       # 4. 注意：如果张量是浮点数，请确保范围为 [0, 255] 并转换为 uint8
    newimage = torch.cat(newimage,dim=0)

    # 对提取区域加马赛克


    # 将加了马赛克的区域放回原图
    return newimage

def add_mosaic(image, block_size=10):
    h, w = image.shape[:2]
    image_small = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    image_mosaic = cv2.resize(image_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return image_mosaic

# 测试用例
''''
if __name__ == "__main__":
    image = cv2.imread(r"D:\ai picture\75a502c2-a854-40ae-9044-057d2265efa4.png")  # 读取图片
    plt.subplot(2,1,1)
    # 指定马赛克区域
    top_left = (200, 200)  # 左上角坐标
    bottom_right = (400, 400)  # 右下角坐标
    mosaic_image = add_mosaic_to_region(image, top_left, bottom_right, block_size=20)
    cv2.imwrite("mosaic_output.jpg", mosaic_image)  # 保存结果
    plt.subplot(2, 1, 1)
    cv2.imshow("Mosaic Region Image", mosaic_image)  # 展示结果
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''