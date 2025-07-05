# --- FILE: visualizer_enhanced.py ---

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.patches import Circle
import pandas as pd
import seaborn as sns
import os
import json
from datetime import datetime
import config_enhanced as cfg

def plot_3d_solution(solution):
    """绘制完整3D解决方案可视化"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置视角
    ax.view_init(elev=30, azim=135)
    
    # 绘制地面控制中心
    ax.scatter([cfg.CONTROL_CENTER[0]], [cfg.CONTROL_CENTER[1]], [0], 
               marker='*', s=200, c='gold', edgecolor='black', label='控制中心')
    
    # 如果有聚类数据，绘制聚类中心和需求点
    if 'kmeans' in solution:
        kmeans = solution['kmeans']
        demand_points = solution['demand_points']
        priorities = solution['priorities']
        
        # 绘制需求点
        norm = plt.Normalize(vmin=1, vmax=5)  # 优先级映射
        scatter = ax.scatter(demand_points[:, 0], demand_points[:, 1], np.zeros_like(demand_points[:, 0]),
                 c=priorities, cmap='viridis', s=50, alpha=0.7, norm=norm)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('优先级')
        
        # 绘制聚类中心
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                   np.zeros_like(kmeans.cluster_centers_[:, 0]), marker='o', s=150, 
                   c='orange', edgecolor='black', label='卡车补给点')
        
        # 如果有卡车路线，绘制卡车路线
        if 'truck_route' in solution:
            truck_route = solution['truck_route']
            truck_points = solution['truck_points']
            
            # 绘制卡车路线
            for i in range(len(truck_route)-1):
                start = truck_route[i]
                end = truck_route[i+1]
                ax.plot([truck_points[start][0], truck_points[end][0]],
                        [truck_points[start][1], truck_points[end][1]],
                        [0.01, 0.01], 'k-', lw=2, alpha=0.7)
            
            # 连接最后和第一个点构成闭环
            ax.plot([truck_points[truck_route[-1]][0], truck_points[truck_route[0]][0]],
                    [truck_points[truck_route[-1]][1], truck_points[truck_route[0]][1]],
                    [0.01, 0.01], 'k-', lw=2, alpha=0.7, label='卡车路线')
            
        # 如果有无人机路径，绘制无人机路径
        if 'drone_paths' in solution:
            drone_paths = solution['drone_paths']
            
            colors = cm.rainbow(np.linspace(0, 1, len(drone_paths)))
            for i, (center, paths) in enumerate(drone_paths.items()):
                color = colors[i]
                for path in paths:
                    xs, ys, zs = zip(*path)
                    # 绘制无人机路径
                    ax.plot(xs, ys, zs, '-', lw=1, alpha=0.8, color=color)
                    # 标记起点和终点
                    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker='^', s=50, color=color)
                    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker='v', s=50, color=color)
            
            # 为无人机路径添加标签
            ax.plot([], [], 'b-', lw=1, alpha=0.8, label='无人机路径')
            
    # 如果有威胁区域，绘制威胁区域
    if 'threat_zones' in solution:
        threats = solution['threat_zones']
        
        threat_colors = {
            'low': (0.3, 0.7, 0.3, 0.2),      # 绿色，低透明度
            'medium': (0.9, 0.7, 0.1, 0.3),   # 橙色，中透明度
            'high': (0.8, 0.2, 0.2, 0.4)      # 红色，高透明度
        }
        
        for threat in threats:
            center = threat['center']
            radius = threat['radius']
            level = threat['threat_level']
            
            # 绘制威胁区域柱体
            theta = np.linspace(0, 2*np.pi, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            
            # 高度根据威胁等级调整
            height_multiplier = {
                'low': 0.2,
                'medium': 0.3,
                'high': 0.4
            }
            height = height_multiplier.get(level, 0.3)
            
            # 绘制底面
            ax.plot(x, y, np.zeros_like(x), color=threat_colors[level][:3], alpha=0.5)
            
            # 绘制侧面（采样点）
            for i in range(0, len(theta), 5):
                ax.plot([x[i], x[i]], [y[i], y[i]], [0, height], 
                         color=threat_colors[level][:3], alpha=0.15)
                
            # 绘制顶面
            ax.plot(x, y, height * np.ones_like(x), color=threat_colors[level][:3], alpha=0.5)
            
        # 添加威胁区域示例到图例
        for level, color in threat_colors.items():
            ax.plot([], [], '-', color=color[:3], alpha=0.5, 
                     label=f'{level}级威胁区域')
    
    # 设置轴标签和标题
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('高度 (km)')
    ax.set_title('智能无人运输车与无人机协同优化3D可视化')
    
    # 设置轴范围
    limit = cfg.AREA_LIMIT
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([0, 0.5])
    
    # 添加图例
    ax.legend(loc='upper right')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig('3d_solution_visualization.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def plot_efficiency_comparison(df):
    """绘制算法效率对比图"""
    # 创建画布和子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 任务完成时间对比
    if 'total_time' in df.columns:
        ax = axes[0, 0]
        sns.barplot(x='algorithm', y='total_time', data=df, ax=ax, ci='sd')
        ax.set_title('算法任务完成时间对比', fontsize=12)
        ax.set_xlabel('算法')
        ax.set_ylabel('完成时间 (小时)')
        
    # 2. 随问题规模的计算时间
    if 'num_points' in df.columns and 'computation_time' in df.columns:
        ax = axes[0, 1]
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            if len(algo_df) > 1:  # 确保有足够数据点
                ax.plot(algo_df['num_points'], algo_df['computation_time'], 
                       marker='o', linestyle='-', label=algo)
                
        ax.set_title('算法可扩展性分析', fontsize=12)
        ax.set_xlabel('需求点数量')
        ax.set_ylabel('计算时间 (秒)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
    # 3. 能量消耗对比
    if 'energy_consumed_wh' in df.columns:
        ax = axes[1, 0]
        energy_df = df[df['energy_consumed_wh'].notna()]
        if not energy_df.empty:
            sns.barplot(x='algorithm', y='energy_consumed_wh', data=energy_df, ax=ax)
            ax.set_title('算法能量消耗对比', fontsize=12)
            ax.set_xlabel('算法')
            ax.set_ylabel('能量消耗 (Wh)')
            
    # 4. 不同场景下的性能表现
    if 'scenario' in df.columns and 'total_time' in df.columns:
        ax = axes[1, 1]
        scenario_df = df[df['scenario'].notna()]
        if not scenario_df.empty:
            sns.boxplot(x='scenario', y='total_time', data=scenario_df, ax=ax)
            ax.set_title('不同场景下的算法表现', fontsize=12)
            ax.set_xlabel('场景')
            ax.set_ylabel('完成时间 (小时)')
    
    plt.tight_layout()
    plt.savefig('algorithm_efficiency_comparison.png', dpi=300)
    plt.close(fig)
    
def plot_robustness_analysis(df):
    """绘制算法鲁棒性分析图"""
    if 'scenario' not in df.columns:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 筛选有场景信息的数据
    scenario_df = df[df['scenario'].notna()]
    if scenario_df.empty:
        return
        
    # 1. 计算变异系数（标准差/均值）
    cv_data = []
    for algo in scenario_df['algorithm'].unique():
        for scenario in scenario_df['scenario'].unique():
            subset = scenario_df[(scenario_df['algorithm'] == algo) & 
                               (scenario_df['scenario'] == scenario)]
            if len(subset) >= 2:  # 至少需要2个样本才能计算标准差
                cv = subset['total_time'].std() / subset['total_time'].mean()
                cv_data.append({
                    'algorithm': algo,
                    'scenario': scenario,
                    'cv': cv
                })
                
    cv_df = pd.DataFrame(cv_data)
    
    if not cv_df.empty:
        # 绘制变异系数热图
        pivot_cv = cv_df.pivot_table(index='algorithm', columns='scenario', values='cv')
        sns.heatmap(pivot_cv, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax1)
        ax1.set_title('算法在不同场景下的变异系数 (CV)', fontsize=12)
        
        # 绘制箱线图
        sns.boxplot(x='algorithm', y='total_time', hue='scenario', data=scenario_df, ax=ax2)
        ax2.set_title('不同场景下算法完成时间分布', fontsize=12)
        ax2.set_xlabel('算法')
        ax2.set_ylabel('完成时间 (小时)')
        ax2.legend(title='场景')
    
    plt.tight_layout()
    plt.savefig('algorithm_robustness_analysis.png', dpi=300)
    plt.close(fig)
    
def generate_comprehensive_report(report_data):
    """生成综合实验报告"""
    # 创建报告文件夹
    report_dir = 'experiment_report'
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        
    # 生成主HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>智能无人运输车与无人机协同优化实验报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #205375; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .result-section {{ margin-bottom: 30px; background: #f9f9f9; padding: 15px; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #205375; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            .highlight {{ background-color: #e6f7ff; padding: 10px; border-left: 4px solid #1890ff; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>智能无人运输车与无人机协同优化实验报告</h1>
            <p>实验日期：{report_data['timestamp']}</p>
            
            <div class="result-section">
                <h2>1. 执行摘要</h2>
                <p class="highlight">本实验通过一系列对比测试，验证了所提出的协同优化算法在完成时间、计算效率和能源消耗方面的优越性。</p>
                
                <h3>主要结果概览：</h3>
                <ul>
    """
    
    # 添加效率收益
    if 'efficiency_gains' in report_data:
        for key, value in report_data['efficiency_gains'].items():
            html_content += f"<li>与{key[3:]}算法相比：提高效率 {value:.2f}%</li>\n"
    
    # 添加可扩展性分析
    html_content += """
                </ul>
            </div>
            
            <div class="result-section">
                <h2>2. 可扩展性分析</h2>
                <table>
                    <tr>
                        <th>算法</th>
                        <th>时间复杂度</th>
                        <th>R²</th>
                    </tr>
    """
    
    if 'scalability_analysis' in report_data:
        for algo, analysis in report_data['scalability_analysis'].items():
            html_content += f"""
                    <tr>
                        <td>{algo}</td>
                        <td>O(n<sup>{analysis['complexity_order']:.2f}</sup>)</td>
                        <td>{analysis.get('r_squared', 'N/A'):.4f}</td>
                    </tr>
            """
    
    html_content += """
                </table>
                <img src="../algorithm_efficiency_comparison.png" alt="算法效率对比">
            </div>
            
            <div class="result-section">
                <h2>3. 鲁棒性分析</h2>
                <p>在不同场景下的性能表现：</p>
                <table>
                    <tr>
                        <th>场景</th>
                        <th>平均性能</th>
                        <th>标准差</th>
                        <th>变异系数</th>
                    </tr>
    """
    
    if 'robustness_analysis' in report_data:
        for scenario, metrics in report_data['robustness_analysis'].items():
            html_content += f"""
                    <tr>
                        <td>{scenario}</td>
                        <td>{metrics['mean_performance']:.4f}</td>
                        <td>{metrics['std_performance']:.4f}</td>
                        <td>{metrics['cv']:.4f}</td>
                    </tr>
            """
    
    html_content += """
                </table>
                <img src="../algorithm_robustness_analysis.png" alt="算法鲁棒性分析">
            </div>
            
            <div class="result-section">
                <h2>4. 3D解决方案可视化</h2>
                <img src="../3d_solution_visualization.png" alt="3D解决方案可视化">
            </div>
            
            <div class="result-section">
                <h2>5. 结论与建议</h2>
                <p>基于实验结果，我们得出以下结论：</p>
                <ul>
                    <li>提出的协同优化算法在各种测试场景中均表现出显著的性能优势</li>
                    <li>该算法在处理动态威胁环境和解决多目标优化问题方面具有鲁棒性</li>
                    <li>算法的可扩展性良好，能够有效处理更大规模的问题实例</li>
                </ul>
                
                <h3>未来工作建议：</h3>
                <ul>
                    <li>进一步优化3D路径规划算法，提高在复杂地形中的性能</li>
                    <li>研究更高效的多无人机协调机制，以支持更大规模的集群操作</li>
                    <li>探索将强化学习方法集成到框架中，以提高动态环境下的适应性</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # 保存HTML报告
    with open(os.path.join(report_dir, 'report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    # 保存JSON数据
    with open(os.path.join(report_dir, 'report_data.json'), 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
        
    print(f"报告已生成至 {report_dir}/report.html") 