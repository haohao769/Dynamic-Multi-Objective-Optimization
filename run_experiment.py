# --- FILE: run_experiment.py ---
# 优化的主运行脚本，包含所有修复

import numpy as np
import pandas as pd
import time
import os
import sys
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 确保正确的导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'

# 导入所有必要的模块
try:
    import config_enhanced as cfg
    from utilities import calculate_path_dist, is_line_segment_intersecting_cylinder_3d
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
    from baseline_algorithms import (
        GreedyTSP,
        NearestNeighborClustering,
        SimpleGeneticAlgorithm
    )
    print("所有模块导入成功！")
except Exception as e:
    print(f"导入错误: {e}")
    sys.exit(1)

class OptimizedExperiment:
    """优化的实验框架，包含性能改进和错误修复"""
    
    def __init__(self):
        self.results = []
        self.algorithms = {
            'proposed': self.run_proposed_algorithm,
            'greedy': self.run_greedy_baseline,
            'nearest_neighbor': self.run_nn_baseline,
            'simple_ga': self.run_simple_ga_baseline
        }
        # 预创建结果目录
        self.results_dir = "experiment_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
    def run_proposed_algorithm(self, num_points, seed, scenario_params):
        """运行提出的算法（优化版本）"""
        print(f"\n运行提出的算法: {num_points}个需求点, 种子={seed}")
        start_time = time.time()
        
        try:
            # 1. 场景生成
            demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
            
            # 2. 动态威胁模型
            threat_model = DynamicThreatModel()
            num_threats = max(1, num_points // 5)  # 确保至少有一个威胁
            
            for _ in range(num_threats):
                threat_model.threat_zones.append({
                    'center': np.random.randn(2) * 10,
                    'radius': np.random.uniform(0.5, 2.0),
                    'threat_level': np.random.choice(['low', 'medium', 'high']),
                    'mobility': np.random.choice(['static', 'mobile'])
                })
                
            # 3. 聚类优化
            kmeans = optimize_clusters_with_threats(demand_points, priorities, threat_model, seed)
            restricted_areas = generate_dynamic_restricted_airspace(kmeans, demand_points, threat_model)
            
            # 4. 卡车路径规划
            truck_points = np.vstack([cfg.CONTROL_CENTER, kmeans.cluster_centers_])
            truck_route, truck_cost = solve_tsp_with_ga(truck_points, restricted_areas, seed)
            truck_dist = calculate_path_dist(truck_route, truck_points)
            truck_time = truck_dist / cfg.TRUCK_SPEED_KMH
            
            # 5. 3D地形和无人机规划
            terrain = TerrainModel()
            
            # 转换禁飞区为3D
            restricted_areas_3d = []
            for area in restricted_areas:
                restricted_areas_3d.append({
                    'center': (*area['center'], 0.2),
                    'radius': area['radius'],
                    'bottom': 0.0,
                    'top': 0.4
                })
                
            # 6. 载荷模型和无人机任务规划
            payload_model = DronePayloadEnduranceModel()
            coordinator = MultiUAVCoordinator()
            
            max_drone_time = 0
            total_energy_consumed = 0
            all_missions = []
            
            for i in range(cfg.NUM_CLUSTERS):
                cluster_pts = demand_points[kmeans.labels_ == i]
                cluster_priorities = priorities[kmeans.labels_ == i]
                
                if len(cluster_pts) == 0:
                    continue
                    
                center = kmeans.cluster_centers_[i]
                
                # 生成补给需求
                supply_requirements = []
                for j, (pt, pri) in enumerate(zip(cluster_pts, cluster_priorities)):
                    supply_type = np.random.choice(['medical', 'ammunition', 'food', 'general'],
                                                 p=[0.2, 0.3, 0.3, 0.2])
                    weight = np.random.uniform(1.0, 4.0)
                    
                    supply_requirements.append({
                        'weight': weight,
                        'type': supply_type,
                        'priority': int(pri)
                    })
                    
                # 优化载荷分配
                missions = payload_model.optimize_payload_distribution(cluster_pts, supply_requirements)
                
                # 3D路径规划
                avg_payload = np.mean([m['payload'] for m in missions]) if missions else 2.5
                missions_3d, cluster_time = plan_drone_missions_3d(
                    center, cluster_pts, restricted_areas_3d, terrain, avg_payload
                )
                
                max_drone_time = max(max_drone_time, cluster_time)
                all_missions.extend(missions_3d)
                
                # 计算能耗
                for mission in missions:
                    params = payload_model.calculate_flight_parameters(
                        mission['payload'], mission['supply_type']
                    )
                    total_energy_consumed += params['power_consumption_w'] * params['flight_time_h']
                    
            # 7. 多无人机冲突解决
            if all_missions:
                uav_missions = []
                for i, mission in enumerate(all_missions):
                    uav_missions.append({
                        'uav_id': f'UAV_{i}',
                        'path': mission.get('path_3d', []),
                        'priority': np.random.randint(1, 6)
                    })
                    
                # 解决冲突
                adjusted_missions = coordinator.resolve_conflicts(uav_missions)
                
            # 8. 动态威胁更新模拟
            threat_model.update_threats(300, [])
            
            # 9. 计算总结果
            total_time = truck_time + max_drone_time
            computation_time = time.time() - start_time
            
            # 保存详细解决方案用于可视化
            solution = {
                'demand_points': demand_points,
                'priorities': priorities,
                'kmeans': kmeans,
                'truck_route': truck_route,
                'truck_points': truck_points,
                'threat_zones': threat_model.threat_zones,
                'drone_paths': {}  # 简化的无人机路径存储
            }
            
            # 存储部分无人机路径用于可视化
            for i in range(min(3, len(all_missions))):  # 只存储前3个任务的路径
                if all_missions[i].get('path_3d'):
                    solution['drone_paths'][i] = [all_missions[i]['path_3d']]
                    
            self.latest_solution = solution
            
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
                'avg_priority': np.mean(priorities),
                'solution': solution  # 用于可视化
            }
            
        except Exception as e:
            print(f"算法执行错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                'algorithm': 'proposed',
                'num_points': num_points,
                'seed': seed,
                'total_time': float('inf'),
                'computation_time': time.time() - start_time,
                'error': str(e)
            }
            
    def run_greedy_baseline(self, num_points, seed, scenario_params):
        """运行贪心算法基线"""
        print(f"运行贪心算法基线: {num_points}个需求点")
        start_time = time.time()
        
        try:
            demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
            greedy = GreedyTSP()
            
            # 简单的贪心路径
            all_points = np.vstack([cfg.CONTROL_CENTER, demand_points])
            route = greedy.solve(all_points)
            
            total_dist = calculate_path_dist(route, all_points)
            total_time = total_dist / cfg.TRUCK_SPEED_KMH
            
            return {
                'algorithm': 'greedy',
                'num_points': num_points,
                'seed': seed,
                'total_time': total_time,
                'computation_time': time.time() - start_time,
                'truck_time': total_time,
                'drone_time': 0
            }
        except Exception as e:
            print(f"贪心算法错误: {e}")
            return {
                'algorithm': 'greedy',
                'num_points': num_points,
                'seed': seed,
                'total_time': float('inf'),
                'computation_time': time.time() - start_time,
                'error': str(e)
            }
            
    def run_nn_baseline(self, num_points, seed, scenario_params):
        """运行最近邻基线"""
        print(f"运行最近邻基线: {num_points}个需求点")
        start_time = time.time()
        
        try:
            demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
            
            # 使用最近邻聚类
            nn_clustering = NearestNeighborClustering(n_clusters=cfg.NUM_CLUSTERS)
            nn_clustering.fit(demand_points)
            
            # 使用贪心算法规划卡车路径
            truck_points = np.vstack([cfg.CONTROL_CENTER, nn_clustering.cluster_centers_])
            greedy = GreedyTSP()
            truck_route = greedy.solve(truck_points)
            
            # 计算总时间
            truck_dist = calculate_path_dist(truck_route, truck_points)
            truck_time = truck_dist / cfg.TRUCK_SPEED_KMH
            
            # 估计无人机时间
            max_cluster_dist = 0
            for i in range(cfg.NUM_CLUSTERS):
                cluster_pts = demand_points[nn_clustering.labels_ == i]
                if len(cluster_pts) > 0:
                    center = nn_clustering.cluster_centers_[i]
                    distances = [np.linalg.norm(pt - center) for pt in cluster_pts]
                    max_cluster_dist = max(max_cluster_dist, max(distances) if distances else 0)
            
            drone_time = (max_cluster_dist * 2) / cfg.DRONE_BASE_SPEED_KMH
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
        except Exception as e:
            print(f"最近邻算法错误: {e}")
            return {
                'algorithm': 'nearest_neighbor',
                'num_points': num_points,
                'seed': seed,
                'total_time': float('inf'),
                'computation_time': time.time() - start_time,
                'error': str(e)
            }
            
    def run_simple_ga_baseline(self, num_points, seed, scenario_params):
        """运行简单遗传算法基线"""
        print(f"运行简单遗传算法基线: {num_points}个需求点")
        start_time = time.time()
        
        try:
            demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
            
            # 使用最近邻聚类
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
        except Exception as e:
            print(f"简单GA算法错误: {e}")
            return {
                'algorithm': 'simple_ga',
                'num_points': num_points,
                'seed': seed,
                'total_time': float('inf'),
                'computation_time': time.time() - start_time,
                'error': str(e)
            }
            
    def run_quick_test(self):
        """快速测试所有算法是否正常工作 (优化版)"""
        print("\n" + "="*60)
        print("开始快速测试")
        print("="*60)
        
        # 减少测试规模
        test_size = 10  # 从20减少到10个点
        test_seed = cfg.RANDOM_SEED
        
        # 添加超时控制
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("算法执行超时")
        
        for algo_name, algo_func in self.algorithms.items():
            print(f"\n测试 {algo_name}...")
            
            # 设置超时 (30秒)
            try:
                # 仅在Unix系统上使用信号超时
                if os.name != 'nt':  # Windows系统跳过超时设置
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)
                
                # 执行算法
                result = algo_func(test_size, test_seed, {})
                
                # 取消超时
                if os.name != 'nt':
                    signal.alarm(0)
                    
                if 'error' not in result:
                    print(f"✓ {algo_name} 运行成功! 总时间: {result.get('total_time', 'N/A'):.2f}小时")
                    self.results.append(result)  # 保存成功的结果
                else:
                    print(f"✗ {algo_name} 运行失败: {result['error']}")
                    
            except TimeoutError as e:
                if os.name != 'nt':
                    signal.alarm(0)  # 取消超时
                print(f"✗ {algo_name} 执行超时 (>30秒)")
                
            except Exception as e:
                if os.name != 'nt':
                    signal.alarm(0)  # 取消超时
                print(f"✗ {algo_name} 执行错误: {str(e)}")
                
        print("\n快速测试完成!")
        
        # 如果提出算法成功，创建一个简单可视化
        if any(r.get('algorithm') == 'proposed' for r in self.results):
            proposed_result = next(r for r in self.results if r.get('algorithm') == 'proposed')
            if 'solution' in proposed_result:
                try:
                    print("生成简单可视化...")
                    plot_3d_solution(proposed_result['solution'])
                    print("✓ 可视化生成成功!")
                except Exception as e:
                    print(f"✗ 可视化生成失败: {str(e)}")
            
    def run_full_experiment(self):
        """运行完整实验"""
        print("\n" + "="*60)
        print("开始完整实验")
        print("="*60)
        
        # 1. 可扩展性测试（减少测试规模以加快速度）
        print("\n--- 可扩展性测试 ---")
        test_sizes = [20, 50, 100]  # 减少测试规模
        
        for size in test_sizes:
            print(f"\n测试规模: {size}个需求点")
            for algo_name, algo_func in self.algorithms.items():
                if size > 50 and algo_name == 'greedy':
                    continue  # 跳过大规模的贪心算法
                    
                result = algo_func(size, cfg.RANDOM_SEED, {})
                if 'error' not in result:
                    self.results.append(result)
                    print(f"  {algo_name}: {result['total_time']:.2f}小时")
                    
        # 2. 鲁棒性测试（减少运行次数）
        print("\n--- 鲁棒性测试 ---")
        num_runs = 10  # 减少运行次数
        base_size = 50
        
        scenarios = [
            {'name': 'normal', 'threat_density': 0.1},
            {'name': 'high_threat', 'threat_density': 0.3}
        ]
        
        for scenario in scenarios:
            print(f"\n场景: {scenario['name']}")
            for run in range(num_runs):
                seed = cfg.RANDOM_SEED + run
                result = self.run_proposed_algorithm(base_size, seed, scenario)
                if 'error' not in result:
                    result['scenario'] = scenario['name']
                    self.results.append(result)
                    
        # 3. 生成报告和可视化
        self._generate_results()
        
    def _generate_results(self):
        """生成结果报告和可视化"""
        print("\n--- 生成报告 ---")
        
        # 创建数据框
        df = pd.DataFrame(self.results)
        
        # 过滤掉错误结果
        df = df[~df['total_time'].isin([float('inf')])]
        
        if df.empty:
            print("没有有效结果！")
            return
            
        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_experiments': len(df),
            'algorithms_tested': list(df['algorithm'].unique()),
            'summary_statistics': self._calculate_summary_stats(df),
            'efficiency_gains': self._calculate_efficiency_gains(df),
            'scalability_analysis': self._analyze_scalability(df),
            'robustness_analysis': self._analyze_robustness(df)
        }
        
        # 保存结果
        df.to_csv(os.path.join(self.results_dir, 'experiment_results.csv'), index=False)
        with open(os.path.join(self.results_dir, 'experiment_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # 生成可视化
        try:
            plot_efficiency_comparison(df)
            plot_robustness_analysis(df)
            
            # 如果有解决方案，生成3D可视化
            if hasattr(self, 'latest_solution'):
                plot_3d_solution(self.latest_solution)
                
            generate_comprehensive_report(report)
            
        except Exception as e:
            print(f"可视化错误: {e}")
            
        # 打印总结
        print("\n" + "="*60)
        print("实验完成总结")
        print("="*60)
        print(f"总实验次数: {len(df)}")
        print(f"测试算法: {', '.join(df['algorithm'].unique())}")
        
        if 'proposed' in df['algorithm'].values:
            proposed_mean = df[df['algorithm'] == 'proposed']['total_time'].mean()
            print(f"\n提出算法平均完成时间: {proposed_mean:.2f}小时")
            
            for algo in df['algorithm'].unique():
                if algo != 'proposed':
                    algo_mean = df[df['algorithm'] == algo]['total_time'].mean()
                    improvement = (algo_mean - proposed_mean) / algo_mean * 100
                    print(f"相比{algo}提升: {improvement:.1f}%")
                    
    def _calculate_summary_stats(self, df):
        """计算汇总统计"""
        stats = {}
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            stats[algo] = {
                'mean_time': algo_df['total_time'].mean(),
                'std_time': algo_df['total_time'].std(),
                'min_time': algo_df['total_time'].min(),
                'max_time': algo_df['total_time'].max(),
                'mean_computation': algo_df['computation_time'].mean()
            }
        return stats
        
    def _calculate_efficiency_gains(self, df):
        """计算效率提升"""
        gains = {}
        
        if 'proposed' in df['algorithm'].values:
            proposed_df = df[df['algorithm'] == 'proposed']
            proposed_mean = proposed_df['total_time'].mean()
            
            for algo in df['algorithm'].unique():
                if algo != 'proposed':
                    algo_mean = df[df['algorithm'] == algo]['total_time'].mean()
                    gains[f'vs_{algo}'] = (algo_mean - proposed_mean) / algo_mean * 100
                    
        return gains
        
    def _analyze_scalability(self, df):
        """分析可扩展性"""
        scalability = {}
        
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            if 'num_points' in algo_df.columns and len(algo_df) > 2:
                sizes = algo_df['num_points'].values
                times = algo_df['computation_time'].values
                
                # 过滤有效数据
                valid_idx = (sizes > 0) & (times > 0)
                if np.any(valid_idx):
                    sizes = sizes[valid_idx]
                    times = times[valid_idx]
                    
                    if len(sizes) > 1:
                        # 对数拟合
                        coeffs = np.polyfit(np.log(sizes), np.log(times), 1)
                        scalability[algo] = {
                            'complexity_order': coeffs[0],
                            'num_samples': len(sizes)
                        }
                        
        return scalability
        
    def _analyze_robustness(self, df):
        """分析鲁棒性"""
        robustness = {}
        
        if 'scenario' in df.columns:
            for scenario in df['scenario'].dropna().unique():
                scenario_df = df[df['scenario'] == scenario]
                if len(scenario_df) > 1:
                    robustness[scenario] = {
                        'mean_performance': scenario_df['total_time'].mean(),
                        'std_performance': scenario_df['total_time'].std(),
                        'cv': scenario_df['total_time'].std() / scenario_df['total_time'].mean()
                    }
                    
        return robustness

def main():
    """主函数"""
    print("="*70)
    print("智能无人运输车与无人机协同优化研究 - 优化版")
    print("="*70)
    
    # 创建实验对象
    experiment = OptimizedExperiment()
    
    # 运行快速测试
    experiment.run_quick_test()
    
    # 询问是否运行完整实验
    response = input("\n是否运行完整实验? (y/n): ")
    if response.lower() == 'y':
        experiment.run_full_experiment()
    else:
        print("实验已取消")
        
    print("\n实验结束！")

if __name__ == "__main__":
    main()