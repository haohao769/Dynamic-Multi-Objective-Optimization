# --- FILE: fix_imports.py ---
# 这个文件包含所有必要的导入修复

# 1. 修复 multi_uav_coordination.py - 添加缺失的time导入
import time

# 2. 修复 stage3_drone_planning_3d.py - 修正utilities导入
# 将 from utilities import is_line_segment_intersecting_cylinder_3d
# 改为正确的导入路径

# 3. 修复 main_enhanced.py - 确保所有模块都能正确导入
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 4. 安装缺失的依赖
"""
pip install numpy pandas matplotlib scikit-learn scipy seaborn
"""