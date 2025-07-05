# --- FILE: main_enhanced.py ---

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
import json

os.environ['OMP_NUM_THREADS'] = '1'

# 导入增强的模块
import config as cfg
import utilities as utils
from stage1_clustering_enhanced import (
    DynamicThreatModel, 
    generate_demand_points_with_priority,
    optimize_clusters_with_threats,
    generate_dynamic_restricted_airspace
)
from stage2_truck_routing import solve_tsp_with_ga
from stage3_drone_planning_3d import (
    TerrainModel,
    DronePathPlanner3D,
    plan_drone_missions_3d
)
from drone_payload_model import DronePayloadEnduranceModel
from multi_uav_coordination import MultiUAVCoordinator
from visualizer_enhanced import (
    plot_3d_solution,
    plot_efficiency_comparison,
    plot_robustness_analysis,
    generate_comprehensive_report
)

# 对比算法导入
from baseline_algorithms import (
    GreedyTSP,
    NearestNeighborClustering,
    SimpleGeneticAlgorithm
)

class ComprehensiveExperiment:
    """综合实验框架"""
    
    def __init__(self):
        self.results = []
        self.algorithms = {
            'proposed': self.run_proposed_algorithm,
            'greedy': self.run_greedy_baseline,
            'nearest_neighbor': self.run_nn_baseline,
            'simple_ga': self.run_simple_ga_baseline
        }
        
    def run_proposed_algorithm(self, num_points, seed, scenario_params):
        """运行提出的算法"""
        print(f"\n运行提出的算法: {num_points}个需求点, 种子={seed}")
        start_time = time.time()
        
        # 1. 场景生成
        demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
        threat_model = DynamicThreatModel()
        
        # 添加初始威胁
        for _ in range(num_points // 5):
            threat_model.threat_zones.append({
                'center': np.random.randn(2) * 10,
                'radius': np.random.uniform(0.5, 2.0),
                'threat_level': np.random.choice(['low', 'medium', 'high']),
                'mobility': np.random.choice(['static', 'mobile'])
            })
            
        # 2. 聚类优化
        kmeans = optimize_clusters_with_threats(demand_points, priorities, threat_model, seed)
        restricted_areas = generate_dynamic_restricted_airspace(kmeans, demand_points, threat_model)
        
        # 3. 卡车路径规划
        truck_points = np.vstack([cfg.CONTROL_CENTER, kmeans.cluster_centers_])
        truck_route, truck_cost = solve_tsp_with_ga(truck_points, restricted_areas, seed)
        truck_dist = utils.calculate_path_dist(truck_route, truck_points)
        truck_time = truck_dist / cfg.TRUCK_SPEED_KMH
        
        # 4. 3D地形和无人机规划
        terrain = TerrainModel()
        
        # 转换禁飞区为3D
        restricted_areas_3d = []
        for area in restricted_areas:
            restricted_areas_3d.append({
                'center': (*area['center'], 0.2),  # 200m高度中心
                'radius': area['radius'],
                'bottom': 0.0,
                'top': 0.4  # 400m高度
            })
            
        # 5. 载荷模型和无人机任务规划
        payload_model = DronePayloadEnduranceModel()
        coordinator = MultiUAVCoordinator()
        
        max_drone_time = 0
        total_energy_consumed = 0
        
        for i in range(cfg.NUM_CLUSTERS):
            cluster_pts = demand_points[kmeans.labels_ == i]
            cluster_priorities = priorities[kmeans.labels_ == i]
            
            if len(cluster_pts) == 0:
                continue
                
            center = kmeans.cluster_centers_[i]
            
            # 为每个需求点生成补给需求
            supply_requirements = []
            for j, (pt, pri) in enumerate(zip(cluster_pts, cluster_priorities)):
                supply_type = np.random.choice(['medical', 'ammunition', 'food', 'general'],
                                             p=[0.2, 0.3, 0.3, 0.2])
                weight = np.random.uniform(1.0, 4.0)  # 1-4kg
                
                supply_requirements.append({
                    'weight': weight,
                    'type': supply_type,
                    'priority': pri
                })
                
            # 优化载荷分配
            missions = payload_model.optimize_payload_distribution(cluster_pts, supply_requirements)
            
            # 3D路径规划
            missions_3d, cluster_time = plan_drone_missions_3d(
                center, cluster_pts, restricted_areas_3d, terrain, 
                np.mean([m['payload'] for m in missions])
            )
            
            max_drone_time = max(max_drone_time, cluster_time)
            
            # 计算能耗
            for mission in missions:
                params = payload_model.calculate_flight_parameters(
                    mission['payload'], mission['supply_type']
                )
                total_energy_consumed += params['power_consumption_w'] * params['flight_time_h']
                
        # 6. 动态威胁更新模拟
        threat_model.update_threats(300, [])  # 5分钟后的威胁更新
        
        # 7. 结果汇总
        total_time = truck_time + max_drone_time
        computation_time = time.time() - start_time
        
        return {
            'algorithm': 'proposed',
            'num_points': num_points,
            'seed': seed,
            'truck_time': truck_time,
            'drone_time': max_drone_time,
            'total_time': total_time,
            'energy_consumed_wh': total_energy_consumed,
            'computation_time': computation_time,
            'num_threats': len(threat_model.threat_zones),
            'avg_priority': np.mean(priorities)
        }
        
    def run_greedy_baseline(self, num_points, seed, scenario_params):
        """运行贪心算法基线"""
        print(f"运行贪心算法基线: {num_points}个需求点")
        start_time = time.time()
        
        demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
        greedy = GreedyTSP()
        
        # 简单的贪心路径
        all_points = np.vstack([cfg.CONTROL_CENTER, demand_points])
        route = greedy.solve(all_points)
        
        total_dist = utils.calculate_path_dist(route, all_points)
        total_time = total_dist / cfg.TRUCK_SPEED_KMH
        
        return {
            'algorithm': 'greedy',
            'num_points': num_points,
            'seed': seed,
            'total_time': total_time,
            'computation_time': time.time() - start_time
        }
        
    def run_nn_baseline(self, num_points, seed, scenario_params):
        """运行最近邻基线"""
        print(f"运行最近邻基线: {num_points}个需求点")
        start_time = time.time()
        
        from baseline_algorithms import NearestNeighborClustering, GreedyTSP
        
        # 生成需求点
        demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
        
        # 使用最近邻聚类
        nn_clustering = NearestNeighborClustering(n_clusters=cfg.NUM_CLUSTERS)
        nn_clustering.fit(demand_points)
        
        # 使用贪心算法规划卡车路径
        truck_points = np.vstack([cfg.CONTROL_CENTER, nn_clustering.cluster_centers_])
        greedy = GreedyTSP()
        truck_route = greedy.solve(truck_points)
        
        # 计算总时间（卡车时间 + 估计的无人机时间）
        truck_dist = utils.calculate_path_dist(truck_route, truck_points)
        truck_time = truck_dist / cfg.TRUCK_SPEED_KMH
        
        # 简单估计无人机时间
        max_cluster_dist = 0
        for i in range(cfg.NUM_CLUSTERS):
            cluster_pts = demand_points[nn_clustering.labels_ == i]
            if len(cluster_pts) > 0:
                center = nn_clustering.cluster_centers_[i]
                distances = [np.linalg.norm(pt - center) for pt in cluster_pts]
                max_cluster_dist = max(max_cluster_dist, max(distances) if distances else 0)
        
        drone_time = (max_cluster_dist * 2) / cfg.DRONE_BASE_SPEED_KMH  # 往返时间
        total_time = truck_time + drone_time
        
        return {
            'algorithm': 'nearest_neighbor',
            'num_points': num_points,
            'seed': seed,
            'truck_time': truck_time,
            'drone_time': drone_time,
            'total_time': total_time,
            'computation_time': time.time() - start_time
        }
        
    def run_simple_ga_baseline(self, num_points, seed, scenario_params):
        """运行简单遗传算法基线"""
        print(f"运行简单遗传算法基线: {num_points}个需求点")
        start_time = time.time()
        
        from baseline_algorithms import SimpleGeneticAlgorithm, NearestNeighborClustering
        
        # 生成需求点
        demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
        
        # 使用最近邻聚类（为了公平比较）
        clustering = NearestNeighborClustering(n_clusters=cfg.NUM_CLUSTERS)
        clustering.fit(demand_points)
        
        # 使用简单GA优化卡车路径
        truck_points = np.vstack([cfg.CONTROL_CENTER, clustering.cluster_centers_])
        simple_ga = SimpleGeneticAlgorithm(pop_size=50, generations=50, mutation_rate=0.1)
        truck_route, truck_dist = simple_ga.optimize_tsp(truck_points)
        
        truck_time = truck_dist / cfg.TRUCK_SPEED_KMH
        
        # 估计无人机时间
        max_cluster_dist = 0
        for i in range(cfg.NUM_CLUSTERS):
            cluster_pts = demand_points[clustering.labels_ == i]
            if len(cluster_pts) > 0:
                center = clustering.cluster_centers_[i]
                distances = [np.linalg.norm(pt - center) for pt in cluster_pts]
                max_cluster_dist = max(max_cluster_dist, max(distances) if distances else 0)
        
        drone_time = (max_cluster_dist * 2) / cfg.DRONE_BASE_SPEED_KMH
        total_time = truck_time + drone_time
        
        return {
            'algorithm': 'simple_ga',
            'num_points': num_points,
            'seed': seed,
            'truck_time': truck_time,
            'drone_time': drone_time,
            'total_time': total_time,
            'computation_time': time.time() - start_time
        }
        
    def run_scalability_test(self):
        """可扩展性测试"""
        print("\n" + "="*60)
        print("开始可扩展性测试")
        print("="*60)
        
        test_sizes = [20, 50, 100, 200, 500]
        
        for size in test_sizes:
            for algo_name, algo_func in self.algorithms.items():
                if size > 100 and algo_name in ['greedy', 'nearest_neighbor']:
                    continue  # 跳过大规模的简单算法
                    
                result = algo_func(size, cfg.RANDOM_SEED, {})
                self.results.append(result)
                
    def run_robustness_test(self):
        """鲁棒性测试"""
        print("\n" + "="*60)
        print("开始鲁棒性测试")
        print("="*60)
        
        num_runs = 30  # 增加运行次数
        base_size = 50
        
        for run in range(num_runs):
            seed = cfg.RANDOM_SEED + run
            
            # 不同场景参数
            scenarios = [
                {'name': 'normal', 'threat_density': 0.1},
                {'name': 'high_threat', 'threat_density': 0.3},
                {'name': 'dynamic', 'threat_mobility': 0.8}
            ]
            
            for scenario in scenarios:
                result = self.run_proposed_algorithm(base_size, seed, scenario)
                result['scenario'] = scenario['name']
                self.results.append(result)
                
    def run_comparison_study(self):
        """综合对比研究"""
        print("\n" + "="*60)
        print("开始综合对比研究")
        print("="*60)
        
        # 运行所有测试
        self.run_scalability_test()
        self.run_robustness_test()
        
        # 数据分析
        df = pd.DataFrame(self.results)
        
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary_statistics': df.groupby('algorithm').agg({
                'total_time': ['mean', 'std', 'min', 'max'],
                'computation_time': ['mean', 'std'],
                'energy_consumed_wh': ['mean', 'std']
            }).to_dict(),
            'scalability_analysis': self._analyze_scalability(df),
            'robustness_analysis': self._analyze_robustness(df),
            'efficiency_gains': self._calculate_efficiency_gains(df)
        }
        
        # 保存结果
        with open('experiment_results.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # 生成可视化
        plot_efficiency_comparison(df)
        plot_robustness_analysis(df)
        plot_3d_solution(self.results[0])  # 展示一个3D解决方案
        generate_comprehensive_report(report)
        
        return df, report
        
    def _analyze_scalability(self, df):
        """分析可扩展性"""
        scalability = {}
        
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            if 'num_points' in algo_df.columns:
                # 拟合时间复杂度
                sizes = algo_df['num_points'].values
                times = algo_df['computation_time'].values
                
                if len(sizes) > 2:
                    # 简单的多项式拟合
                    coeffs = np.polyfit(np.log(sizes), np.log(times), 1)
                    scalability[algo] = {
                        'complexity_order': coeffs[0],
                        'r_squared': np.corrcoef(np.log(sizes), np.log(times))[0,1]**2
                    }
                    
        return scalability
        
    def _analyze_robustness(self, df):
        """分析鲁棒性"""
        robustness = {}
        
        if 'scenario' in df.columns:
            for scenario in df['scenario'].unique():
                scenario_df = df[df['scenario'] == scenario]
                robustness[scenario] = {
                    'mean_performance': scenario_df['total_time'].mean(),
                    'std_performance': scenario_df['total_time'].std(),
                    'cv': scenario_df['total_time'].std() / scenario_df['total_time'].mean()
                }
                
        return robustness
        
    def _calculate_efficiency_gains(self, df):
        """计算效率提升"""
        gains = {}
        
        proposed_times = df[df['algorithm'] == 'proposed']['total_time'].values
        
        for algo in df['algorithm'].unique():
            if algo != 'proposed':
                algo_times = df[df['algorithm'] == algo]['total_time'].values
                if len(algo_times) > 0 and len(proposed_times) > 0:
                    avg_gain = np.mean((algo_times[:len(proposed_times)] - proposed_times) / algo_times[:len(proposed_times)] * 100)
                    gains[f'vs_{algo}'] = avg_gain
                    
        return gains

def main():
    """主函数"""
    print("智能无人运输车与无人机协同优化研究 - 增强版")
    print("="*70)
    
    # 创建实验框架
    experiment = ComprehensiveExperiment()
    
    # 运行综合对比研究
    results_df, report = experiment.run_comparison_study()
    
    print("\n" + "="*70)
    print("实验完成！")
    print("="*70)
    
    # 打印关键结果
    print("\n效率提升总结:")
    for key, value in report['efficiency_gains'].items():
        print(f"  {key}: {value:.2f}%")
        
    print("\n可扩展性分析:")
    for algo, analysis in report['scalability_analysis'].items():
        print(f"  {algo}: O(n^{analysis['complexity_order']:.2f})")
        
    print("\n结果已保存至:")
    print("  - experiment_results.json")
    print("  - 各类可视化图表")
    
if __name__ == "__main__":
    main()