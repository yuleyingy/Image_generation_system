"""合成多尺度图像数据集。

每张图像由“两色渐变背景 + 1~3 个彩色几何体（圆 / 方块 / 三角形）”组成，
颜色、位置、大小均随机，用于检验低分辨率预览在“构图与色彩”上是否与高分辨率结果一致。
图像以 2 倍超采样渲染后平均，起到抗锯齿作用；像素值范围为 [-1, 1]，格式为 NCHW。
"""
import numpy as np


def _render(rng, n, size, ss=2):
    """随机渲染 n 张 size×size 的合成场景图（渐变背景 + 1~3 个几何体），ss 为超采样倍数。"""
    H = size * ss
    yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, H), indexing="ij")
    imgs = np.empty((n, 3, H, H), np.float32)
    for b in range(n):
        c0, c1 = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
        ang = rng.uniform(0, 2 * np.pi)
        g = (np.cos(ang) * (xx - .5) + np.sin(ang) * (yy - .5)) + .5
        img = c0[:, None, None] * (1 - g) + c1[:, None, None] * g
        for _ in range(rng.integers(1, 4)):
            col = rng.uniform(-1, 1, 3)
            cx, cy = rng.uniform(.15, .85, 2)
            r = rng.uniform(.08, .25)
            kind = rng.integers(3)
            if kind == 0:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2
            elif kind == 1:
                m = (np.abs(xx - cx) < r) & (np.abs(yy - cy) < r)
            else:
                m = (yy - cy < r) & (yy - cy > -r) & (np.abs(xx - cx) < (yy - cy + r) / 2)
            img = np.where(m[None], col[:, None, None], img)
        imgs[b] = img
    imgs = np.clip(imgs, -1, 1)
    return imgs.reshape(n, 3, size, ss, size, ss).mean((3, 5))  # 超采样抗锯齿


def avgpool(x, s=2):
    """面积平均下采样（作为“真实”的低分辨率参考）。"""
    n, c, h, w = x.shape
    return x.reshape(n, c, h // s, s, w // s, s).mean((3, 5))


def sample_batch(rng, n, hr=32):
    """返回一批高分辨率图像 (n,3,hr,hr)。"""
    return _render(rng, n, hr)


def save_samples(out_dir="assets/samples", n=16, seed=2026):
    """导出 n 张高分辨率样例图像及其低分辨率版本（PNG），作为代码附带的数据素材。"""
    import os
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    x = sample_batch(np.random.default_rng(seed), n)
    lr = avgpool(x)
    for i in range(n):
        for tag, arr in (("hr32", x[i]), ("lr16", lr[i])):
            im = ((arr.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(im).save(f"{out_dir}/sample_{i:02d}_{tag}.png")
    print("saved", n, "samples to", out_dir)


if __name__ == "__main__":
    save_samples()
