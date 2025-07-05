# --- FILE: test_modules.py ---
# 测试所有模块是否正常工作

import sys
import os
import numpy as np

print("="*60)
print("模块测试脚本")
print("="*60)

# 测试导入
modules_to_test = [
    'config_enhanced',
    'utilities',
    'baseline_algorithms',
    'drone_payload_model',
    'multi_uav_coordination',
    'stage1_clustering_enhanced',
    'stage2_truck_routing',
    'stage3_drone_planning_3d',
    'visualizer_enhanced'
]

print("\n1. 测试模块导入...")
all_imports_ok = True
for module in modules_to_test:
    try:
        exec(f"import {module}")
        print(f"✓ {module} 导入成功")
    except Exception as e:
        print(f"✗ {module} 导入失败: {e}")
        all_imports_ok = False

if not all_imports_ok:
    print("\n存在导入错误，请检查文件是否完整！")
    sys.exit(1)

print("\n2. 测试基本功能...")

# 测试配置
try:
    import config_enhanced as cfg
    print(f"✓ 配置加载成功 - 仿真区域: {cfg.AREA_LIMIT}km")
except Exception as e:
    print(f"✗ 配置加载失败: {e}")

# 测试需求点生成
try:
    from stage1_clustering_enhanced import generate_demand_points_with_priority
    points, priorities = generate_demand_points_with_priority(10, 42)
    print(f"✓ 需求点生成成功 - 生成了{len(points)}个点")
except Exception as e:
    print(f"✗ 需求点生成失败: {e}")

# 测试聚类
try:
    from stage1_clustering_enhanced import DynamicThreatModel, optimize_clusters_with_threats
    threat_model = DynamicThreatModel()
    threat_model.threat_zones.append({
        'center': np.array([5, 5]),
        'radius': 2,
        'threat_level': 'medium',
        'mobility': 'static'
    })
    kmeans = optimize_clusters_with_threats(points, priorities, threat_model, 42)
    print(f"✓ 聚类优化成功 - {cfg.NUM_CLUSTERS}个聚类")
except Exception as e:
    print(f"✗ 聚类优化失败: {e}")

# 测试TSP求解
try:
    from stage2_truck_routing import solve_tsp_with_ga
    test_points = np.random.rand(5, 2) * 10
    route, dist = solve_tsp_with_ga(test_points, [], 42)
    print(f"✓ TSP求解成功 - 路径长度: {dist:.2f}km")
except Exception as e:
    print(f"✗ TSP求解失败: {e}")

# 测试3D地形
try:
    from stage3_drone_planning_3d import TerrainModel
    terrain = TerrainModel()
    elevation = terrain.get_elevation(0, 0)
    print(f"✓ 地形模型成功 - 原点高程: {elevation:.3f}km")
except Exception as e:
    print(f"✗ 地形模型失败: {e}")

# 测试载荷模型
try:
    from drone_payload_model import DronePayloadEnduranceModel
    payload_model = DronePayloadEnduranceModel()
    params = payload_model.calculate_flight_parameters(2.5, 'medical')
    print(f"✓ 载荷模型成功 - 2.5kg医疗物资续航: {params['flight_time_h']:.2f}小时")
except Exception as e:
    print(f"✗ 载荷模型失败: {e}")

# 测试多机协调
try:
    from multi_uav_coordination import MultiUAVCoordinator
    coordinator = MultiUAVCoordinator()
    print(f"✓ 多机协调模块成功")
except Exception as e:
    print(f"✗ 多机协调模块失败: {e}")

# 测试可视化（不显示图形）
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    from visualizer_enhanced import plot_efficiency_comparison
    print(f"✓ 可视化模块成功")
except Exception as e:
    print(f"✗ 可视化模块失败: {e}")

print("\n3. 性能测试...")

# 测试不同规模的计算时间
import time
test_sizes = [10, 20, 50]
for size in test_sizes:
    start = time.time()
    points, _ = generate_demand_points_with_priority(size, 42)
    elapsed = time.time() - start
    print(f"  生成{size}个点耗时: {elapsed:.3f}秒")

print("\n" + "="*60)
print("测试完成！")
print("="*60)

# 系统信息
print("\n系统信息:")
print(f"Python版本: {sys.version}")
print(f"NumPy版本: {np.__version__}")

try:
    import pandas as pd
    print(f"Pandas版本: {pd.__version__}")
except:
    pass

try:
    import matplotlib
    print(f"Matplotlib版本: {matplotlib.__version__}")
except:
    pass

try:
    import sklearn
    print(f"Scikit-learn版本: {sklearn.__version__}")
except:
    pass

print("\n如果所有测试都通过，您可以运行 run_experiment.py 开始实验！")