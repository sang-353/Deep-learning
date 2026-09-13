from pathlib import Path
from skimage import io, color, transform, exposure, util
import argparse

def preprocess_image(path, size=224, do_equalize=False):
    """把任意图像整理成模型能吃的输入。

    参数:
        path: 输入图像路径
        size: 输出图像的边长，输出为 size x size
        do_equalize: 是否做直方图均衡化
    返回:
        float32 的二维数组（灰度图），取值范围 0~1
    """
    # 读取图像
    image = io.imread(path)
    # 转换为灰度图像
    if image.ndim == 3:
        if image.shape[-1] == 4:
            # 如果是 RGBA 图像，去掉 alpha 通道
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    # 归一化：按 dtype 满量程折算到 0~1 的 float32
    image = util.img_as_float32(image)

    # 缩放，preserve_range=True 让 resize 只改尺寸
    image = transform.resize(image, (size, size), preserve_range=True, anti_aliasing=True)

    # 直方图均衡化
    if do_equalize:
        image = exposure.equalize_hist(image)

    return image.astype('float32')

def main():
    parser = argparse.ArgumentParser(description="Preprocess images for deep learning")
    parser.add_argument("image_path", help="Path to the image to preprocess")
    parser.add_argument("--output", required=True, help="处理后的图像保存路径")
    parser.add_argument("--size", type=int, default=224, help="Size of the output image")
    parser.add_argument("--equalize", action="store_true", help="Apply histogram equalization")
    args = parser.parse_args()

    if not Path(args.image_path).is_file():
        print(f"找不到输入文件: {args.image_path}")
        return
    raw = io.imread(args.image_path)
    print(f"处理前: shape={raw.shape}, dtype={raw.dtype}, 取值范围={raw.min()}~{raw.max()}")
    result = preprocess_image(args.image_path, size=args.size, do_equalize=args.equalize)
    print(f"处理后: shape={result.shape}, dtype={result.dtype}, "
          f"取值范围={result.min():.4f}~{result.max():.4f}")

    # PNG 只能存 0~255 的整数，显式转回 uint8 再保存，避免精度损失警告
    io.imsave(args.output, util.img_as_ubyte(result))
    print(f"预处理后的图像已保存到 {args.output}")

if __name__ == "__main__":
    main()
