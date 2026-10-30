"""快速演示：用训练好的模型对若干随机种子生成高分辨率结果与各种低分辨率预览，并拼成对比图。

用法：python demo.py --seeds 0 1 2 3 4 5 --out demo.png
每一列是一个随机种子；各行依次为：高分辨率结果、参考（面积平均）、直接低分辨率生成、
朴素下采样、论文方法（全局 D* + CG）、本文方法（逐块 D + CG）。在普通 CPU 上约 10 秒完成。
"""
import argparse, pickle
import jax, jax.numpy as jnp, numpy as np
from PIL import Image, ImageDraw
from preview import hr_sample, direct_lr, preview, avgpool


def to_uint8(x):
    """把 (3,H,W)、范围 [-1,1] 的图像转换为 (H,W,3) 的 uint8 数组。"""
    return (np.clip((np.asarray(x).transpose(1, 2, 0) + 1) / 2, 0, 1) * 255).astype(np.uint8)


def main():
    """解析参数，运行各方法并保存对比图。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt.pkl")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--m", type=int, default=5, help="交换子归零引导的步数")
    ap.add_argument("--alpha", type=float, default=0.1, help="交换子归零引导的步长")
    ap.add_argument("--out", default="demo.png")
    a = ap.parse_args()

    P = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(a.ckpt, "rb"))["params"])
    x0 = jnp.asarray(np.stack([np.random.default_rng(s).standard_normal((3, 32, 32)) for s in a.seeds]).astype(np.float32))
    x0l = jnp.asarray(np.stack([np.random.default_rng(s).standard_normal((3, 16, 16)) for s in a.seeds]).astype(np.float32))

    hr, _ = hr_sample(P, x0)
    rows = [("HR 32x32", hr), ("Reference", avgpool(hr)),
            ("Direct LR", direct_lr(P, x0l)[0]),
            ("Naive down", preview(P, x0, select="nearest")[0]),
            ("Paper: global D*+CG", preview(P, x0, select="global", cg=True, m=a.m, alpha=a.alpha)[0]),
            ("Ours: block D+CG", preview(P, x0, select="block", cg=True, m=a.m, alpha=a.alpha)[0])]

    cell, lab_w = 96, 170
    canvas = Image.new("RGB", (lab_w + cell * len(a.seeds), cell * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, (name, imgs) in enumerate(rows):
        draw.text((6, r * cell + cell // 2 - 6), name, fill="black")
        for c in range(len(a.seeds)):
            im = Image.fromarray(to_uint8(imgs[c])).resize((cell - 4, cell - 4), Image.NEAREST)
            canvas.paste(im, (lab_w + c * cell + 2, r * cell + 2))
    canvas.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
