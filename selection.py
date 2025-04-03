import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ================== 数据准备 ==================
speeds = ['Normal Speed', '1.5x Speed', '2x Speed']

# 生成模拟数据 (6000×3)
np.random.seed(42)
sample_size = 6000

before = np.column_stack([
    np.random.normal(0.1747, 0.02, sample_size),
    np.random.normal(0.1486, 0.02, sample_size),
    np.random.normal(0.1718, 0.02, sample_size)
])

after = np.column_stack([
    np.random.normal(0.1290, 0.015, sample_size),
    np.random.normal(0.1172, 0.014, sample_size),
    np.random.normal(0.1508, 0.016, sample_size)
])

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False
})

# ================== 统计检验 ==================
p_values = []
for i in range(3):
    t_stat, p_val = stats.ttest_rel(before[:, i], after[:, i])  # 修复缩进
    p_values.append(p_val)
    print(f"Speed {i}: t = {t_stat:.2f}, p = {p_val:.2e}")

# ================== 可视化设置 ==================
before_means = np.mean(before, axis=0)
after_means = np.mean(after, axis=0)
before_std = np.std(before, axis=0)
after_std = np.std(after, axis=0)

group_num = len(speeds)
bar_width = 0.25
index = np.arange(group_num)

fig, ax = plt.subplots(figsize=(7, 4), dpi=100)

# 定义统一的误差线样式
error_kw = {'elinewidth': 1.5, 'capsize': 10, 'capthick': 1.5}  # 提前定义

# 绘制柱状图（修正参数顺序）
rects1 = ax.bar(
    x=index - bar_width/2,
    height=before_means,
    width=bar_width,
    yerr=before_std,
    label='Before',
    color='#A8DADB',
    edgecolor='black',
    error_kw=error_kw  # 直接使用定义好的样式
)

rects2 = ax.bar(
    x=index + bar_width/2,
    height=after_means,
    width=bar_width,
    yerr=after_std,
    label='After',
    color='#f2b6b6',
    edgecolor='black',
    error_kw=error_kw
)

# ================== 统一化标注系统 ==================
def add_significance(i):
    """优化线宽一致性的标注方法"""
    # 动态高度计算
    max_height = max(before_means[i] + before_std[i],
                    after_means[i] + after_std[i])
    
    # 统一布局参数
    line_gap = max_height * 0.08  # 单一定义比例
    line_y = max_height + line_gap
    y_base = max_height + line_gap
    # 绘制单一横线（去除冗余装饰线）
    ax.plot([index[i]-bar_width/2, index[i]+bar_width/2],
            [line_y, line_y],
            color='#404040',
            lw=1.5,  # 统一线宽
            solid_capstyle="butt",
            zorder=2)  # 确保在文字下方
    for x_pos in [index[i]-bar_width/2, index[i]+bar_width/2]:
        ax.plot([x_pos, x_pos],  # 垂直装饰线
                [y_base - line_gap*0.3, y_base],
                color='dimgray',
                lw=1.2)
    
    # 优化符号定位
    symbol, color = get_symbol(p_values[i])
    ax.text(index[i],
            line_y - 0.002,  # 微调垂直位置
            symbol,
            ha='center',
            va='bottom',
            fontsize=14,
            color=color,
            fontweight='bold',
            zorder=3)  # 确保文字在线上方

# 保留get_symbol函数
def get_symbol(p):
    if p < 0.001:
        return '*', '#2F4F4F'
    elif p < 0.01:
        return '*', '#556B2F'
    elif p < 0.05:
        return '*', '#6B8E23'
    else:
        return 'n.s.', 'gray'

# 应用标注
for i in range(group_num):
    add_significance(i)



# ================== 图表美化 ==================
# 调整y轴范围
max_y = max(np.concatenate([before_means + before_std, 
                           after_means + after_std]))
ax.set_ylim(0, max_y * 1.45)

# 坐标轴标签设置
ax.set_xticks(index)
ax.set_xticklabels([s.replace(' ', '\n') for s in speeds],
                  fontsize=13,
                  linespacing=0.8)
ax.set_ylabel('RMSE', fontsize=14)

# ================== 标题设置 ==================
# ax.set_title('Statistical Analysis Results Before and After Channel Selection', 
#              fontsize=14,

#              pad=10,
#              y=1.01)  # 标题位于图表上方1.05倍高度处

# 网格和图例
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(fontsize=12, loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()
