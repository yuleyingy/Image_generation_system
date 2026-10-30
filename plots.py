"""根据 results/results.json 与 samples.npz 绘制报告中的实验图。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "NotoSerifCJK" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

out = sys.argv[1] if len(sys.argv) > 1 else "results"
R = json.load(open(f"{out}/results.json"))
S = np.load(f"{out}/samples.npz")
names = [m["name"] for m in R["methods"]]


def img(x):
    """把 CHW、范围 [-1,1] 的图像转换为 matplotlib 可显示的 HWC、范围 [0,1] 的数组。"""
    return np.clip((x.transpose(1, 2, 0) + 1) / 2, 0, 1)


# ---- 图A：定性对比 ----
rows = [("高分辨率结果\n(32×32)", S["hr"]), ("参考：面积\n平均(16×16)", S["ref"])]
show = ["直接低分辨率生成", "朴素下采样（最近邻）", "面积平均下采样", "论文方法：全局 D* + CG", "本文方法：逐块 D + CG"]
labels = ["直接低分辨率\n生成", "朴素下采样\n(最近邻)", "面积平均\n下采样", "论文方法\n全局D*+CG", "本文方法\n逐块D+CG"]
# samples.npz 中的 m{i} 按 R["methods"] 的顺序保存
rows += [(l_, S[f"m{names.index(s_)}"]) for s_, l_ in zip(show, labels)]
ncol = 8
fig, ax = plt.subplots(len(rows), ncol, figsize=(ncol * 0.85 + 1.0, len(rows) * 0.88))
for r, (lab, arr) in enumerate(rows):
    for c in range(ncol):
        a = ax[r, c]
        a.imshow(img(arr[c]), interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if c == 0:
            a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
plt.subplots_adjust(left=0.16, right=0.995, top=0.995, bottom=0.005, wspace=0.05, hspace=0.06)
plt.savefig(f"{out}/fig_qual.png", dpi=300)
plt.close()

# ---- 图B：(a) 交换子随时间 (b) 速度场余弦相似度 (c) 引导前后交换子与轨迹偏差 ----
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
ts = sorted(R["comm_over_t"], key=float)
ax[0].plot([float(t) for t in ts], [R["comm_over_t"][t]["comm"] for t in ts], "o-", color="#1f5fa8", ms=4)
ax[0].set_xlabel("时间 t"); ax[0].set_ylabel("交换子平均范数"); ax[0].set_title("(a) 最近邻 D 的交换子", fontsize=9)
ax[0].grid(alpha=.3)
cs = np.array(R["vel_cos"])
k = np.arange(1, len(cs) + 1)
ax[1].plot(k, cs[:, 0], "o-", color="#1f5fa8", ms=4)
ax[1].fill_between(k, cs[:, 0] - cs[:, 1], cs[:, 0] + cs[:, 1], color="#1f5fa8", alpha=.2)
ax[1].set_xlabel("k"); ax[1].set_ylabel("余弦相似度"); ax[1].set_title("(b) 速度场一致性", fontsize=9)
ax[1].grid(alpha=.3)
cc = R["comm_change"]
labs = ["全局D*", "逐块D"]
x = np.arange(2)
w = 0.2
ax[2].bar(x - 1.5 * w, [cc["global_noCG"]["after"], cc["block_noCG"]["after"]], w, label="交换子·无CG", color="#9bbbe0")
ax[2].bar(x - 0.5 * w, [cc["global_CG"]["after"], cc["block_CG"]["after"]], w, label="交换子·有CG", color="#1f5fa8")
ax[2].bar(x + 0.5 * w, [cc["global_noCG"]["dev"], cc["block_noCG"]["dev"]], w, label="轨迹偏差·无CG", color="#f2b48a")
ax[2].bar(x + 1.5 * w, [cc["global_CG"]["dev"], cc["block_CG"]["dev"]], w, label="轨迹偏差·有CG", color="#d9622b")
ax[2].set_xticks(x); ax[2].set_xticklabels(labs)
ax[2].set_title(r"(c) $t_{D+m+1}$ 处的度量", fontsize=9)
ax[2].set_ylim(0, 0.78)
ax[2].legend(fontsize=6.2, frameon=False, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.2)
ax[2].grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(f"{out}/fig_analysis.png", dpi=300)
plt.close()

# ---- 图C：速度–质量折线 ----
fig, a = plt.subplots(figsize=(4.2, 2.6))
pts = {m["name"]: (m["flop_speedup"], m["psnr"]) for m in R["methods"]}
hr = ["NFE=20（高分辨率）", "NFE=10（高分辨率）", "NFE=6（高分辨率）"]
a.plot([pts[n][0] for n in hr], [pts[n][1] for n in hr], "s--", color="gray", label="减少步数（高分辨率）")
for n, mk, col, lab in [("朴素下采样（最近邻）", "v", "#999999", "朴素下采样"),
                         ("直接低分辨率生成", "x", "#444444", "直接低分辨率"),
                         ("论文方法：全局 D* + CG", "o", "#1f5fa8", "论文方法 N=30"),
                         ("论文方法（N=15）", "o", "#7fa7d6", "论文方法 N=15"),
                         ("本文方法：逐块 D + CG", "*", "#d9622b", "本文方法 N=30"),
                         ("本文方法（N=15）", "*", "#f0a77f", "本文方法 N=15")]:
    a.plot(*pts[n], mk, color=col, ms=8, label=lab)
a.set_xlabel("理论加速比（按前向计算量）"); a.set_ylabel("PSNR/dB")
a.grid(alpha=.3); a.legend(fontsize=6.5, frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(f"{out}/fig_speed.png", dpi=300)
plt.close()
print("plots done")
