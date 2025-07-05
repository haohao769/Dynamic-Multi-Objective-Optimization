# 智能无人运输车与无人机协同优化系统 - 安装与运行指南

## 1. 环境准备

### 系统要求
- Python 3.8 或更高版本
- 至少 8GB RAM
- 建议使用虚拟环境

### 创建虚拟环境
```bash
# 创建虚拟环境
python -m venv drone_env

# 激活虚拟环境
# Windows:
drone_env\Scripts\activate
# Linux/Mac:
source drone_env/bin/activate
```

## 2. 安装依赖

### 必需的Python包
```bash
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install matplotlib==3.6.3
pip install scikit-learn==1.2.2
pip install scipy==1.10.1
pip install seaborn==0.12.2
```

### 可选包（用于性能优化）
```bash
pip install numba  # 加速计算
pip install joblib  # 并行处理
```

## 3. 文件组织

确保所有文件在同一目录下：
```
project_directory/
│
├── config_enhanced.py          # 配置文件
├── utilities.py               # 工具函数
├── baseline_algorithms.py     # 基线算法
├── drone_payload_model.py     # 载荷模型
├── multi_uav_coordination.py  # 多机协调
├── stage1_clustering_enhanced.py   # 聚类算法
├── stage2_truck_routing.py    # 卡车路由
├── stage3_drone_planning_3d.py     # 3D路径规划
├── visualizer_enhanced.py     # 可视化
├── main_enhanced.py          # 原始主程序
└── run_experiment.py         # 优化的运行脚本
```

## 4. 运行实验

### 快速测试（推荐先运行）
```python
# 运行优化的实验脚本
python run_experiment.py

# 选择运行快速测试
# 当提示"是否运行完整实验?"时，输入'n'进行快速测试
```

### 完整实验
```python
# 运行完整实验（耗时较长）
python run_experiment.py

# 当提示"是否运行完整实验?"时，输入'y'
```

### 自定义实验
```python
# 修改config_enhanced.py中的参数
# 例如：
# SCALE_TEST_POINTS = [20, 50]  # 减少测试规模
# ROBUSTNESS_TEST_RUNS = 5      # 减少鲁棒性测试次数
```

## 5. 查看结果

### 结果文件位置
```
experiment_results/
├── experiment_results.csv     # 详细实验数据
├── experiment_report.json     # 实验报告
└── report.html               # HTML格式报告

# 可视化图表
├── 3d_solution_visualization.png
├── algorithm_efficiency_comparison.png
└── algorithm_robustness_analysis.png
```

### 查看报告
1. 打开 `experiment_results/report.html` 查看综合报告
2. 使用Excel或Python打开 `.csv` 文件进行详细分析
3. 查看生成的可视化图表

## 6. 常见问题解决

### 问题1：导入错误
```python
# 如果出现模块导入错误，检查：
# 1. 所有文件是否在同一目录
# 2. Python路径是否正确
import sys
print(sys.path)  # 查看Python路径
```

### 问题2：内存不足
```python
# 减少测试规模
# 在config_enhanced.py中修改：
SCALE_TEST_POINTS = [20, 50]  # 减少测试点数
GA_POPULATION_SIZE = 50       # 减少种群大小
```

### 问题3：运行时间过长
```python
# 调整参数加快运行
# 在config_enhanced.py中：
GA_GENERATIONS = 50           # 减少遗传算法代数
ROBUSTNESS_TEST_RUNS = 5      # 减少测试次数
```

### 问题4：可视化错误
```bash
# 如果matplotlib显示错误，尝试：
# Linux系统：
export MPLBACKEND=Agg

# 或在代码开头添加：
import matplotlib
matplotlib.use('Agg')
```

## 7. 性能优化建议

### 并行处理
```python
# 在run_experiment.py中添加并行处理
from joblib import Parallel, delayed

# 并行运行多个算法
results = Parallel(n_jobs=-1)(
    delayed(algo_func)(size, seed, params) 
    for algo_func in algorithms
)
```

### 缓存优化
```python
# 使用缓存避免重复计算
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_distance_cached(p1, p2):
    return np.linalg.norm(p1 - p2)
```

## 8. 扩展开发

### 添加新算法
1. 在 `baseline_algorithms.py` 中添加新算法类
2. 在 `run_experiment.py` 的 `algorithms` 字典中注册
3. 确保返回格式与其他算法一致

### 添加新的性能指标
1. 在算法返回结果中添加新指标
2. 在 `_calculate_summary_stats` 中添加统计
3. 在可视化函数中添加新图表

## 9. 论文撰写建议

### 实验数据使用
- 使用 `experiment_results.csv` 中的数据制作表格
- 使用生成的图表作为论文插图
- 引用 `experiment_report.json` 中的统计数据

### 结果描述模板
```
在[X]个需求点的测试场景中，所提出的算法相比[基线算法]
实现了[Y]%的效率提升，计算时间复杂度为O(n^[Z])。
在[N]次鲁棒性测试中，算法的变异系数为[CV]，
表明算法具有良好的稳定性。
```

## 10. 联系支持

如有问题，请检查：
1. Python版本是否正确
2. 所有依赖是否安装
3. 文件路径是否正确
4. 配置参数是否合理

祝实验顺利！