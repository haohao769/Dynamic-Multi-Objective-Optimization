# --- FILE: enhanced_visualizer.py ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from performance_optimizers import EconomicAnalyzer, MultiObjectiveOptimizer
import time
from typing import List, Dict, Tuple, Optional
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedVisualizationEngine:
    """高级可视化引擎"""
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8', dpi=300):
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use(style)
        
        # 颜色配置
        self.colors = {
            'truck_route': '#2E8B57',      # 海绿色
            'drone_path': '#4169E1',       # 皇室蓝
            'threat_zone': '#FF6347',      # 番茄红
            'demand_point': '#FFD700',     # 金色
            'cluster_center': '#8B0000',   # 暗红色
            'control_center': '#000000',   # 黑色
            'restricted_area': '#FF4500'   # 橙红色
        }
        
        # 3D可视化配置
        self.elevation_colormap = 'terrain'
        self.animation_frames = []
        
    def create_comprehensive_solution_plot(self, solution_data, save_path=None):
        """创建综合解决方案图表"""
        fig = plt.figure(figsize=(20, 12))
        
        # 主要2D解决方案图
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        self._plot_2d_solution(ax1, solution_data)
        
        # 3D解决方案图
        ax2 = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2, projection='3d')
        self._plot_3d_solution(ax2, solution_data)
        
        # 性能指标雷达图
        ax3 = plt.subplot2grid((3, 4), (2, 0), polar=True)
        self._plot_performance_radar(ax3, solution_data)
        
        # 经济效益分析
        ax4 = plt.subplot2grid((3, 4), (2, 1))
        self._plot_economic_analysis(ax4, solution_data)
        
        # 收敛历史
        ax5 = plt.subplot2grid((3, 4), (2, 2))
        self._plot_convergence_history(ax5, solution_data)
        
        # 威胁分析热力图
        ax6 = plt.subplot2grid((3, 4), (2, 3))
        self._plot_threat_heatmap(ax6, solution_data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"综合图表已保存至: {save_path}")
        
        return fig
    
    def _plot_2d_solution(self, ax, solution_data):
        """绘制2D解决方案"""
        # 绘制需求点
        demand_points = solution_data['demand_points']
        priorities = solution_data.get('priorities', np.ones(len(demand_points)))
        
        scatter = ax.scatter(demand_points[:, 0], demand_points[:, 1], 
                           c=priorities, s=100, cmap='viridis', alpha=0.7,
                           edgecolors='black', linewidth=1)
        plt.colorbar(scatter, ax=ax, label='优先级')
        
        # 绘制卡车路线
        if 'truck_route' in solution_data:
            truck_route = solution_data['truck_route']
            route_points = demand_points[truck_route]
            # 添加返回起点的线
            route_points = np.vstack([route_points, route_points[0]])
            
            ax.plot(route_points[:, 0], route_points[:, 1], 
                   color=self.colors['truck_route'], linewidth=3, 
                   label='卡车路线', alpha=0.8)
            
            # 标记路线顺序
            for i, point in enumerate(route_points[:-1]):
                ax.annotate(str(i), (point[0], point[1]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, fontweight='bold')
        
        # 绘制无人机路径
        if 'drone_missions' in solution_data:
            for i, mission in enumerate(solution_data['drone_missions']):
                if 'path' in mission:
                    path = np.array(mission['path'])
                    ax.plot(path[:, 0], path[:, 1], 
                           color=self.colors['drone_path'], 
                           linewidth=2, alpha=0.6,
                           linestyle='--', label=f'无人机{i+1}' if i == 0 else "")
        
        # 绘制威胁区域
        if 'threat_zones' in solution_data:
            for threat in solution_data['threat_zones']:
                circle = Circle(threat['center'], threat['radius'], 
                              color=self.colors['threat_zone'], alpha=0.3)
                ax.add_patch(circle)
        
        # 绘制聚类中心
        if 'cluster_centers' in solution_data:
            centers = solution_data['cluster_centers']
            ax.scatter(centers[:, 0], centers[:, 1], 
                      color=self.colors['cluster_center'], 
                      s=200, marker='X', label='聚类中心')
        
        # 绘制控制中心
        ax.scatter(0, 0, color=self.colors['control_center'], 
                  s=300, marker='*', label='控制中心')
        
        ax.set_xlabel('X坐标 (km)')
        ax.set_ylabel('Y坐标 (km)')
        ax.set_title('智能协同优化解决方案')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_3d_solution(self, ax, solution_data):
        """绘制3D解决方案"""
        # 绘制地形
        if 'terrain_model' in solution_data:
            terrain = solution_data['terrain_model']
            X, Y = np.meshgrid(terrain['x'], terrain['y'])
            Z = terrain['elevation']
            
            surf = ax.plot_surface(X, Y, Z, cmap=self.elevation_colormap, 
                                 alpha=0.3, antialiased=True)
        
        # 绘制3D无人机路径
        if 'drone_missions' in solution_data:
            for i, mission in enumerate(solution_data['drone_missions']):
                if 'path_3d' in mission:
                    path_3d = np.array(mission['path_3d'])
                    ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2],
                           color=self.colors['drone_path'], linewidth=2,
                           label=f'无人机{i+1}路径' if i == 0 else "")
        
        # 绘制3D威胁区域（圆柱体）
        if 'threat_zones_3d' in solution_data:
            for threat in solution_data['threat_zones_3d']:
                center = threat['center']
                radius = threat['radius']
                height = threat.get('height', 0.5)
                
                # 创建圆柱体表面
                theta = np.linspace(0, 2*np.pi, 20)
                z_cyl = np.linspace(0, height, 10)
                theta_mesh, z_mesh = np.meshgrid(theta, z_cyl)
                
                x_cyl = center[0] + radius * np.cos(theta_mesh)
                y_cyl = center[1] + radius * np.sin(theta_mesh)
                
                ax.plot_surface(x_cyl, y_cyl, z_mesh, 
                              color=self.colors['threat_zone'], alpha=0.3)
        
        ax.set_xlabel('X坐标 (km)')
        ax.set_ylabel('Y坐标 (km)')
        ax.set_zlabel('高度 (km)')
        ax.set_title('3D路径规划与威胁环境')
        ax.legend()
    
    def _plot_performance_radar(self, ax, solution_data):
        """绘制性能指标雷达图"""
        metrics = solution_data.get('performance_metrics', {})
        
        # 性能指标
        categories = ['时间效率', '路径优化', '安全性', '燃料消耗', '任务成功率']
        values = [
            metrics.get('time_efficiency', 0.7) * 100,
            metrics.get('path_optimization', 0.8) * 100,
            metrics.get('safety_score', 0.9) * 100,
            metrics.get('fuel_efficiency', 0.75) * 100,
            metrics.get('success_rate', 0.95) * 100
        ]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('算法性能雷达图')
        ax.grid(True)
    
    def _plot_economic_analysis(self, ax, solution_data):
        """绘制经济效益分析"""
        economic_data = solution_data.get('economic_analysis', {})
        
        categories = ['燃料成本', '电力成本', '维护成本', '人员成本', '保险成本']
        costs = [
            economic_data.get('fuel_cost', 1000),
            economic_data.get('electricity_cost', 200),
            economic_data.get('maintenance_cost', 500),
            economic_data.get('operator_cost', 800),
            economic_data.get('insurance_cost', 200)
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        wedges, texts, autotexts = ax.pie(costs, labels=categories, autopct='%1.1f%%',
                                         startangle=90, colors=colors)
        
        ax.set_title('成本构成分析')
    
    def _plot_convergence_history(self, ax, solution_data):
        """绘制收敛历史"""
        if 'algorithm_performance' in solution_data:
            history = solution_data['algorithm_performance'].get('convergence_history', [])
            if history:
                generations = range(len(history))
                ax.plot(generations, history, 'b-', linewidth=2, alpha=0.7)
                ax.fill_between(generations, history, alpha=0.3)
                ax.set_xlabel('代数')
                ax.set_ylabel('适应度值')
                ax.set_title('算法收敛历史')
                ax.grid(True, alpha=0.3)
    
    def _plot_threat_heatmap(self, ax, solution_data):
        """绘制威胁分析热力图"""
        # 创建威胁强度网格
        x_range = np.linspace(-20, 20, 50)
        y_range = np.linspace(-20, 20, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        threat_intensity = np.zeros_like(X)
        
        if 'threat_zones' in solution_data:
            for threat in solution_data['threat_zones']:
                center = threat['center']
                radius = threat['radius']
                level_multiplier = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                multiplier = level_multiplier.get(threat.get('threat_level', 'medium'), 2)
                
                # 计算到威胁中心的距离
                distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
                # 添加威胁强度（高斯分布）
                threat_intensity += multiplier * np.exp(-(distances / radius)**2)
        
        im = ax.imshow(threat_intensity, extent=[-20, 20, -20, 20], 
                      cmap='Reds', alpha=0.7, origin='lower')
        plt.colorbar(im, ax=ax, label='威胁强度')
        ax.set_xlabel('X坐标 (km)')
        ax.set_ylabel('Y坐标 (km)')
        ax.set_title('威胁环境热力图')
    
    def create_interactive_3d_plot(self, solution_data, save_path=None):
        """创建交互式3D图表"""
        fig = go.Figure()
        
        # 添加需求点
        demand_points = solution_data['demand_points']
        priorities = solution_data.get('priorities', np.ones(len(demand_points)))
        
        fig.add_trace(go.Scatter3d(
            x=demand_points[:, 0],
            y=demand_points[:, 1],
            z=np.zeros(len(demand_points)),
            mode='markers',
            marker=dict(
                size=8,
                color=priorities,
                colorscale='Viridis',
                colorbar=dict(title="优先级"),
                line=dict(width=2, color='black')
            ),
            name='需求点',
            text=[f'点{i}: 优先级{p}' for i, p in enumerate(priorities)],
            hovertemplate='<b>%{text}</b><br>坐标: (%{x:.2f}, %{y:.2f})<extra></extra>'
        ))
        
        # 添加卡车路线
        if 'truck_route' in solution_data:
            truck_route = solution_data['truck_route']
            route_points = demand_points[truck_route]
            # 添加返回起点
            route_points = np.vstack([route_points, route_points[0]])
            
            fig.add_trace(go.Scatter3d(
                x=route_points[:, 0],
                y=route_points[:, 1],
                z=np.zeros(len(route_points)),
                mode='lines+markers',
                line=dict(color='green', width=6),
                marker=dict(size=4),
                name='卡车路线'
            ))
        
        # 添加无人机路径
        if 'drone_missions' in solution_data:
            for i, mission in enumerate(solution_data['drone_missions']):
                if 'path_3d' in mission:
                    path_3d = np.array(mission['path_3d'])
                    fig.add_trace(go.Scatter3d(
                        x=path_3d[:, 0],
                        y=path_3d[:, 1],
                        z=path_3d[:, 2],
                        mode='lines',
                        line=dict(color='blue', width=4, dash='dash'),
                        name=f'无人机{i+1}路径'
                    ))
        
        # 设置布局
        fig.update_layout(
            title='交互式3D协同优化解决方案',
            scene=dict(
                xaxis_title='X坐标 (km)',
                yaxis_title='Y坐标 (km)',
                zaxis_title='高度 (km)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=700
        )
        
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            print(f"交互式3D图表已保存至: {save_path}")
        
        return fig
    
    def create_algorithm_comparison_dashboard(self, comparison_data, save_path=None):
        """创建算法对比仪表板"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('执行时间对比', '解决方案质量', '收敛性分析', 
                          '鲁棒性测试', '可扩展性分析', '综合评分'),
            specs=[[{"type": "bar"}, {"type": "box"}, {"type": "scatter"}],
                   [{"type": "violin"}, {"type": "heatmap"}, {"type": "bar"}]]
        )
        
        algorithms = list(comparison_data.keys())
        colors = px.colors.qualitative.Set1[:len(algorithms)]
        
        # 1. 执行时间对比
        execution_times = [comparison_data[alg].get('avg_execution_time', 0) for alg in algorithms]
        fig.add_trace(
            go.Bar(x=algorithms, y=execution_times, marker_color=colors, name='执行时间'),
            row=1, col=1
        )
        
        # 2. 解决方案质量箱线图
        for i, alg in enumerate(algorithms):
            quality_scores = comparison_data[alg].get('quality_scores', [])
            fig.add_trace(
                go.Box(y=quality_scores, name=alg, marker_color=colors[i]),
                row=1, col=2
            )
        
        # 3. 收敛性分析
        for i, alg in enumerate(algorithms):
            convergence = comparison_data[alg].get('convergence_curve', [])
            if convergence:
                fig.add_trace(
                    go.Scatter(y=convergence, mode='lines', name=alg, 
                             line=dict(color=colors[i])),
                    row=1, col=3
                )
        
        # 4. 鲁棒性测试（小提琴图）
        for i, alg in enumerate(algorithms):
            robustness_scores = comparison_data[alg].get('robustness_scores', [])
            fig.add_trace(
                go.Violin(y=robustness_scores, name=alg, 
                         line_color=colors[i], fillcolor=colors[i]),
                row=2, col=1
            )
        
        # 5. 可扩展性分析（热力图）
        scale_data = []
        scale_algorithms = []
        scale_sizes = []
        scale_times = []
        
        for alg in algorithms:
            scalability = comparison_data[alg].get('scalability_test', {})
            for size, time_val in scalability.items():
                scale_algorithms.append(alg)
                scale_sizes.append(size)
                scale_times.append(time_val)
        
        if scale_times:
            # 创建热力图数据矩阵
            unique_algs = list(set(scale_algorithms))
            unique_sizes = sorted(list(set(scale_sizes)))
            
            heatmap_data = []
            for alg in unique_algs:
                row = []
                for size in unique_sizes:
                    # 查找对应的时间值
                    time_val = 0
                    for i, (a, s, t) in enumerate(zip(scale_algorithms, scale_sizes, scale_times)):
                        if a == alg and s == size:
                            time_val = t
                            break
                    row.append(time_val)
                heatmap_data.append(row)
            
            fig.add_trace(
                go.Heatmap(z=heatmap_data, x=unique_sizes, y=unique_algs,
                          colorscale='Viridis'),
                row=2, col=2
            )
        
        # 6. 综合评分
        overall_scores = [comparison_data[alg].get('overall_score', 0) for alg in algorithms]
        fig.add_trace(
            go.Bar(x=algorithms, y=overall_scores, marker_color=colors, name='综合评分'),
            row=2, col=3
        )
        
        # 更新布局
        fig.update_layout(
            title_text="算法性能对比仪表板",
            showlegend=False,
            height=800
        )
        
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            print(f"算法对比仪表板已保存至: {save_path}")
        
        return fig
    
    def create_animation_sequence(self, solution_process, save_path=None):
        """创建算法过程动画"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        def animate(frame):
            ax.clear()
            
            # 获取当前帧数据
            current_data = solution_process[frame]
            
            # 绘制当前状态
            self._plot_2d_solution(ax, current_data)
            
            # 添加进度信息
            ax.text(0.02, 0.98, f'步骤: {frame+1}/{len(solution_process)}', 
                   transform=ax.transAxes, fontsize=12, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=len(solution_process),
                                     interval=500, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"动画已保存至: {save_path}")
        
        return anim
    
    def generate_comprehensive_report(self, all_results, save_path=None):
        """生成综合报告"""
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>智能协同优化系统 - 性能分析报告</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .metrics {{ display: flex; justify-content: space-around; }}
                .metric {{ text-align: center; padding: 10px; }}
                .metric h3 {{ color: #3498db; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>智能协同优化系统性能分析报告</h1>
                <p>生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>执行摘要</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>{all_results.get('total_experiments', 0)}</h3>
                        <p>总实验次数</p>
                    </div>
                    <div class="metric">
                        <h3>{all_results.get('avg_improvement', 0):.1f}%</h3>
                        <p>平均性能提升</p>
                    </div>
                    <div class="metric">
                        <h3>{all_results.get('success_rate', 0):.1f}%</h3>
                        <p>成功率</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>算法性能对比</h2>
                <table>
                    <tr>
                        <th>算法</th>
                        <th>平均执行时间(s)</th>
                        <th>解决方案质量</th>
                        <th>收敛代数</th>
                        <th>综合评分</th>
                    </tr>
        """
        
        # 添加算法对比表格
        for alg_name, results in all_results.get('algorithm_comparison', {}).items():
            report_html += f"""
                    <tr>
                        <td>{alg_name}</td>
                        <td>{results.get('avg_execution_time', 0):.2f}</td>
                        <td>{results.get('avg_quality', 0):.2f}</td>
                        <td>{results.get('avg_convergence', 0):.0f}</td>
                        <td>{results.get('overall_score', 0):.2f}</td>
                    </tr>
            """
        
        report_html += """
                </table>
            </div>
            
            <div class="section">
                <h2>经济效益分析</h2>
                <p>基于当前解决方案的成本效益分析显示：</p>
                <ul>
        """
        
        # 添加经济分析
        economic_data = all_results.get('economic_analysis', {})
        for key, value in economic_data.items():
            if isinstance(value, (int, float)):
                report_html += f"<li>{key}: ¥{value:.2f}</li>"
        
        report_html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            print(f"综合报告已保存至: {save_path}")
        
        return report_html

# 使用示例和测试
if __name__ == "__main__":
    # 创建测试数据
    test_solution_data = {
        'demand_points': np.random.rand(20, 2) * 20 - 10,
        'priorities': np.random.randint(1, 6, 20),
        'truck_route': np.arange(20),
        'threat_zones': [
            {'center': [5, 5], 'radius': 3, 'threat_level': 'high'},
            {'center': [-5, -5], 'radius': 2, 'threat_level': 'medium'}
        ],
        'performance_metrics': {
            'time_efficiency': 0.85,
            'path_optimization': 0.78,
            'safety_score': 0.92,
            'fuel_efficiency': 0.73,
            'success_rate': 0.95
        }
    }
    
    # 测试可视化引擎
    viz_engine = AdvancedVisualizationEngine()
    
    # 创建综合图表
    fig = viz_engine.create_comprehensive_solution_plot(test_solution_data, 
                                                       'test_comprehensive_plot.png')
    plt.show()
    
    print("高级可视化引擎测试完成！")