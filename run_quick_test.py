#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --- FILE: run_quick_test.py ---
# 简化的测试脚本，只运行快速测试

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# 确保正确的导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'

# 尝试导入run_experiment模块
try:
    from run_experiment import OptimizedExperiment
    print("成功导入实验模块")
except Exception as e:
    print(f"导入错误: {e}")
    sys.exit(1)

def main():
    """主函数"""
    print("="*70)
    print("智能无人运输车与无人机协同优化 - 快速测试")
    print("="*70)
    
    # 创建实验对象
    print("初始化实验框架...")
    start_time = time.time()
    experiment = OptimizedExperiment()
    
    # 只运行快速测试
    experiment.run_quick_test()
    
    print("\n" + "="*70)
    print(f"测试完成！总耗时: {time.time() - start_time:.2f}秒")
    print("="*70)

if __name__ == "__main__":
    main() 