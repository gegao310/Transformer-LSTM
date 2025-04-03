import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# 数据准备
errors_same = {
    "Shoulder Flex./Ret.": {
        "Normal": [0.1052, 0.1171, 0.1230,  0.1086, 0.1096, 0.0815],
        "1.5x":   [0.1934, 0.1417, 0.1408,  0.1931, 0.1955, 0.1003],
        "2x":     [0.2435, 0.1761, 0.1739,  0.2257, 0.2308, 0.1453],
    },
    "Forearm Pron./Sup.": {
        "Normal": [0.1576, 0.1979, 0.2072,  0.1596, 0.1496, 0.1363],
        "1.5x":   [0.3507, 0.5076, 0.5438,  0.3862, 0.3495, 0.3113],
        "2x":     [0.1946, 0.2732, 0.3312,  0.1912, 0.1728, 0.1512],
    },
    "Shoulder Abd./Add.": {
        "Normal": [0.0792, 0.0857, 0.0850,  0.0822, 0.0923, 0.0806],
        "1.5x":   [ 0.1245, 0.1495, 0.1228, 0.1445, 0.1858, 0.1212],
        "2x":     [ 0.1670, 0.1610, 0.1713, 0.1636, 0.2258, 0.1344],
    },
}

errors_across = {
    "Shoulder Flex./Ret.": {
        "Normal": [0.1526,0.1789, 0.1523,  0.1619, 0.1497, 0.1290],
        "1.5x":   [0.1515, 0.1533, 0.1534, 0.1565, 0.1580, 0.1172],
        "2x":     [0.1899, 0.1890, 0.1891, 0.1952, 0.1920, 0.1508],
    },
    "Forearm Pron./Sup.": {
        "Normal": [0.3967, 0.3909, 0.4281, 0.4594, 0.3780, 0.2850],
        "1.5x":   [0.3033, 0.4497, 0.5140, 0.3435, 0.3016, 0.2541],
        "2x":     [0.3238, 0.4279, 0.4954, 0.3132, 0.2814, 0.2226],
    },
    "Shoulder Abd./Add.": {
        "Normal": [0.2007,  0.2084, 0.2338, 0.1930, 0.2006, 0.1621],
        "1.5x":   [0.2097,  0.2167, 0.2276, 0.2129, 0.1998, 0.1976],
        "2x":     [0.2090,  0.2142, 0.2199, 0.2139, 0.2198, 0.1944],
    },
}


models = ["CNN-LSTM","LSTM", "GRU" , "BiLSTM-ATT", "TCN", "Proposed"]
speed = list(errors_same["Shoulder Flex./Ret."].keys())
#colors = ['#4B6587', '#C9C5BA', '#F0E5CF', '#D1A827',  '#7C909A',  '#3C4F5C' ]
#colors = ['#D89D6A',  '#7C909A',  '#D4B483',  '#826C5B',  '#A3A3A3',  '#5C4B51']
#colors = ['#405A7A',  '#88A0B8', '#C4D1DF', '#5B8C9D', '#2E4E5F', '#9FB4C7' ]
#colors = ['#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D','#59A14F', '#8CD17D']
colors = ['#F0FAEF','#A8DADB','#457BB3','#1D3557','#f2b6b6','#E73847']
# colors = [
#     '#8fd0fc',  # 天青蓝
#     '#a3e6a3',  # 樱花粉
#     '#c7a3c0',  # 嫩芽绿
#     '#f7c682',  # 杏仁白
#     '#f2b6b6',   # 冰湖蓝
#     '#c0c0c0'
# ]
# 全局字体设置（关键修改点）


# 设置全局样式和高级配色
# plt.style.use('seaborn')
# mpl.rcParams['font.family'] = 'Arial'

plt.rcParams.update({
    'font.family': 'Times New Roman',  # 主字体
    'mathtext.fontset': 'stix',        # 数学公式字体（与Times New Roman兼容）
    'axes.unicode_minus': False        # 解决负号显示异常
})


# 定义莫兰迪配色（低饱和度高级配色）
#COLORS = ['#6F8FAF', '#B4869F', '#C5AFA0']  # 对应三种速度：Normal, 1.5x, 2x
COLORS = ['#E73847', '#1D3557', '#457BB3']  # 对应三种速度：Normal, 1.5x, 2x
MARKERS = ['o', 's', 'D']                    # 不同速度的标记形状
LINE_STYLES = ['-', '--', ':']               # 不同速度的线型

# 定义绘图函数
def plot_errors(data_dict, title):
    metrics = list(data_dict.keys())  # 获取指标名称
    
    # 创建画布和子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle(title, fontsize=14, y=1.05)
    
    # 遍历每个指标（肩关节、前臂等）
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        speed_data = data_dict[metric]
        
        # 绘制每种速度的折线
        for speed_idx, speed in enumerate(speed_data.keys()):
            values = speed_data[speed]
            ax.plot(
                models, values,
                color=COLORS[speed_idx],
                linestyle=LINE_STYLES[speed_idx],
                marker=MARKERS[speed_idx],
                markersize=8,
                linewidth=2,
                alpha=0.9,
                label=f'{speed} speed'
            )
        
        # 设置子图格式
        ax.set_title(metric, fontsize=14, pad=10)
        ax.set_xlabel('Models', fontsize=14)
        ax.set_ylabel('Error Value', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=0, labelsize=13)
        ax.legend(frameon=True, loc='upper left', fontsize=12)
    
    plt.tight_layout()
    return fig

# ----------- 绘制相同主题内的误差 -----------
fig_same = plot_errors(
    errors_same, 
    "Error Comparison Within Same Subject (Three Metrics)"
)

# 调整图例位置到左下角
for ax in fig_same.axes:
    legend = ax.get_legend()
    if legend:
        legend.set_loc('upper left')
        legend.get_frame().set_alpha(0.5)

fig_same.savefig('same_subject_errors.png', bbox_inches='tight')

# ----------- 绘制不同主题间的误差 -----------
fig_across = plot_errors(
    errors_across, 
    "Error Comparison Across Different Subjects (Three Metrics)"
)
# 调整图例位置到左下角
for ax in fig_across.axes:
    legend = ax.get_legend()
    if legend:
        legend.set_loc('lower left')
        legend.get_frame().set_alpha(0.5)
fig_across.savefig('across_subjects_errors.png', bbox_inches='tight')
plt.show()