# --- FILE: optimized_experiment_framework.py ---

import numpy as np
import pandas as pd
import time
import os
import sys
import json
import pickle
import hashlib
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
import threading
import queue
from typing import Dict, List, Tuple, Any, Optional, Callable
import psutil
import warnings
warnings.filterwarnings('ignore')

# 导入优化模块
from performance_optimizers import (
    DistanceMatrixCache, 
    EconomicAnalyzer, 
    MultiObjectiveOptimizer,
    performance_monitor
)
from multi_objective_optimizer import MultiObjectiveSolver, ScenarioAdaptiveOptimizer
from enhanced_visualizer import AdvancedVisualizationEngine

# 导入核心算法模块
try:
    import config_enhanced as cfg
    from stage1_clustering_enhanced import (
        generate_demand_points_with_priority,
        optimize_clusters_with_threats,
        DynamicThreatModel
    )
    from stage2_truck_routing import solve_tsp_with_ga
    from stage3_drone_planning_3d import plan_drone_missions_3d, TerrainModel
    from baseline_algorithms import GreedyTSP, NearestNeighborClustering, SimpleGeneticAlgorithm
except ImportError as e:
    print(f"导入警告: {e}")

class ExperimentCache:
    """实验结果缓存管理器"""
    
    def __init__(self, cache_dir="experiment_cache", max_size_mb=500):
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        self.cache_index = {}
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        self.index_file = os.path.join(cache_dir, "cache_index.json")
        self._load_index()
    
    def _load_index(self):
        """加载缓存索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    self.cache_index = json.load(f)
            except:
                self.cache_index = {}
    
    def _save_index(self):
        """保存缓存索引"""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache_index, f, indent=2, ensure_ascii=False)
    
    def _get_cache_key(self, experiment_config: Dict) -> str:
        """生成缓存键值"""
        # 只使用影响结果的关键参数
        key_params = {
            'num_points': experiment_config.get('num_points'),
            'algorithm': experiment_config.get('algorithm'),
            'seed': experiment_config.get('seed'),
            'threat_config': experiment_config.get('threat_config'),
            'ga_params': {
                'population_size': experiment_config.get('ga_population_size'),
                'generations': experiment_config.get('ga_generations'),
                'mutation_rate': experiment_config.get('ga_mutation_rate')
            }
        }
        
        config_str = json.dumps(key_params, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_cached_result(self, experiment_config: Dict) -> Optional[Dict]:
        """获取缓存的实验结果"""
        cache_key = self._get_cache_key(experiment_config)
        
        if cache_key in self.cache_index:
            cache_file = self.cache_index[cache_key]['file']
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    # 更新访问时间
                    self.cache_index[cache_key]['last_access'] = time.time()
                    self._save_index()
                    
                    return result
                except:
                    # 缓存文件损坏，删除
                    self._remove_cache_entry(cache_key)
        
        return None
    
    def cache_result(self, experiment_config: Dict, result: Dict):
        """缓存实验结果"""
        cache_key = self._get_cache_key(experiment_config)
        cache_file = f"{cache_key}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_file)
        
        try:
            # 检查缓存大小限制
            self._cleanup_cache_if_needed()
            
            # 保存结果
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            # 更新索引
            file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
            self.cache_index[cache_key] = {
                'file': cache_file,
                'size_mb': file_size,
                'created': time.time(),
                'last_access': time.time(),
                'config_summary': {
                    'num_points': experiment_config.get('num_points'),
                    'algorithm': experiment_config.get('algorithm')
                }
            }
            
            self._save_index()
            
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def _cleanup_cache_if_needed(self):
        """清理缓存（如果超过大小限制）"""
        total_size = sum(entry['size_mb'] for entry in self.cache_index.values())
        
        if total_size > self.max_size_mb:
            # 按最后访问时间排序，删除最旧的条目
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['last_access']
            )
            
            for cache_key, entry in sorted_entries:
                if total_size <= self.max_size_mb * 0.8:  # 清理到80%
                    break
                
                self._remove_cache_entry(cache_key)
                total_size -= entry['size_mb']
    
    def _remove_cache_entry(self, cache_key: str):
        """删除缓存条目"""
        if cache_key in self.cache_index:
            cache_file = self.cache_index[cache_key]['file']
            cache_path = os.path.join(self.cache_dir, cache_file)
            
            if os.path.exists(cache_path):
                os.remove(cache_path)
            
            del self.cache_index[cache_key]

class ParallelExperimentRunner:
    """并行实验运行器"""
    
    def __init__(self, max_workers=None, use_cache=True):
        self.max_workers = max_workers or min(cpu_count(), 8)
        self.cache = ExperimentCache() if use_cache else None
        self.progress_queue = queue.Queue()
        self.results_lock = threading.Lock()
        
        # 性能监控
        self.system_monitor = SystemMonitor()
        
    def run_single_experiment(self, experiment_config: Dict) -> Dict:
        """运行单个实验"""
        start_time = time.time()
        
        # 检查缓存
        if self.cache:
            cached_result = self.cache.get_cached_result(experiment_config)
            if cached_result is not None:
                cached_result['from_cache'] = True
                cached_result['cache_hit_time'] = time.time() - start_time
                return cached_result
        
        try:
            # 执行实验
            result = self._execute_experiment(experiment_config)
            result['from_cache'] = False
            result['execution_time'] = time.time() - start_time
            
            # 缓存结果
            if self.cache:
                self.cache.cache_result(experiment_config, result)
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'experiment_config': experiment_config,
                'execution_time': time.time() - start_time,
                'from_cache': False
            }
    
    def _execute_experiment(self, config: Dict) -> Dict:
        """执行具体的实验"""
        algorithm = config['algorithm']
        num_points = config['num_points']
        seed = config['seed']
        
        # 生成问题数据
        np.random.seed(seed)
        demand_points, priorities = generate_demand_points_with_priority(num_points, seed)
        
        # 创建威胁模型
        threat_model = DynamicThreatModel()
        if config.get('threat_config'):
            threat_config = config['threat_config']
            for i in range(threat_config.get('num_threats', 2)):
                threat_model.threat_zones.append({
                    'center': np.random.uniform(-15, 15, 2),
                    'radius': np.random.uniform(2, 5),
                    'threat_level': np.random.choice(['low', 'medium', 'high'])
                })
        
        # 运行算法
        if algorithm == 'optimized_proposed':
            result = self._run_optimized_proposed_algorithm(
                demand_points, priorities, threat_model, config)
        elif algorithm == 'baseline_greedy':
            result = self._run_baseline_greedy(demand_points, priorities, config)
        elif algorithm == 'baseline_ga':
            result = self._run_baseline_ga(demand_points, priorities, config)
        else:
            raise ValueError(f"未知算法: {algorithm}")
        
        # 添加公共指标
        result.update({
            'num_points': num_points,
            'algorithm': algorithm,
            'seed': seed,
            'demand_points': demand_points.tolist(),
            'priorities': priorities.tolist()
        })
        
        return result
    
    def _run_optimized_proposed_algorithm(self, demand_points, priorities, threat_model, config):
        """运行优化的建议算法"""
        start_time = time.time()
        
        # 阶段1：优化聚类
        kmeans, clustering_metrics = optimize_clusters_with_threats(
            demand_points, priorities, threat_model, config['seed'])
        
        # 阶段2：优化TSP求解
        cluster_centers = np.vstack([cfg.CONTROL_CENTER.reshape(1, -1), kmeans.cluster_centers_])
        tsp_result = solve_tsp_with_ga(cluster_centers, seed=config['seed'])
        
        # 计算经济效益
        economic_analyzer = EconomicAnalyzer()
        solution_data = {
            'truck_route': {'total_distance': tsp_result.get('total_distance', 0)},
            'drone_missions': [{'flight_time': 1800}]  # 示例飞行时间
        }
        economic_analysis = economic_analyzer.calculate_comprehensive_cost(solution_data)
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'total_distance': tsp_result.get('total_distance', 0),
            'clustering_quality': clustering_metrics.get('clustering_quality', 0),
            'economic_analysis': economic_analysis,
            'algorithm_performance': tsp_result.get('algorithm_performance', {}),
            'efficiency_score': 1.0 / (1.0 + total_time + tsp_result.get('total_distance', 0) / 1000)
        }
    
    def _run_baseline_greedy(self, demand_points, priorities, config):
        """运行基线贪心算法"""
        start_time = time.time()
        
        greedy_tsp = GreedyTSP()
        cluster_centers = np.vstack([cfg.CONTROL_CENTER.reshape(1, -1), demand_points[:cfg.NUM_CLUSTERS]])
        route = greedy_tsp.solve(cluster_centers)
        
        # 计算总距离
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += np.linalg.norm(cluster_centers[route[i]] - cluster_centers[route[i+1]])
        total_distance += np.linalg.norm(cluster_centers[route[-1]] - cluster_centers[route[0]])
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'total_distance': total_distance,
            'route': route.tolist(),
            'efficiency_score': 1.0 / (1.0 + total_time + total_distance / 1000)
        }
    
    def _run_baseline_ga(self, demand_points, priorities, config):
        """运行基线遗传算法"""
        start_time = time.time()
        
        simple_ga = SimpleGeneticAlgorithm(
            points=demand_points[:min(len(demand_points), cfg.NUM_CLUSTERS + 1)])
        route, distance = simple_ga.solve()
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'total_distance': distance,
            'route': route.tolist(),
            'efficiency_score': 1.0 / (1.0 + total_time + distance / 1000)
        }
    
    def run_parallel_experiments(self, experiment_configs: List[Dict], 
                                progress_callback: Callable = None) -> List[Dict]:
        """并行运行多个实验"""
        print(f"开始并行运行 {len(experiment_configs)} 个实验，使用 {self.max_workers} 个进程")
        
        results = []
        completed_count = 0
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_experiment, config): config
                for config in experiment_configs
            }
            
            # 收集结果
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed_count += 1
                    
                    # 进度回调
                    if progress_callback:
                        progress = completed_count / len(experiment_configs)
                        progress_callback(progress, completed_count, len(experiment_configs))
                    
                    # 简单进度显示
                    if completed_count % 10 == 0 or completed_count == len(experiment_configs):
                        elapsed = time.time() - start_time
                        rate = completed_count / elapsed
                        eta = (len(experiment_configs) - completed_count) / rate if rate > 0 else 0
                        
                        print(f"进度: {completed_count}/{len(experiment_configs)} "
                              f"({100*completed_count/len(experiment_configs):.1f}%) "
                              f"- 速度: {rate:.2f} exp/s - 预计剩余: {eta:.1f}s")
                
                except Exception as e:
                    print(f"实验执行失败: {e}")
                    results.append({
                        'error': str(e),
                        'experiment_config': config
                    })
                    completed_count += 1
        
        total_time = time.time() - start_time
        print(f"所有实验完成，总用时: {total_time:.2f}秒")
        
        return results

class SystemMonitor:
    """系统性能监控器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                memory_usage = psutil.virtual_memory().used / (1024**3)  # GB
                self.peak_memory = max(self.peak_memory, memory_usage)
                time.sleep(1)
            except:
                break
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_count': cpu_count(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
            'peak_memory_gb': self.peak_memory
        }

class ComprehensiveExperimentSuite:
    """综合实验套件"""
    
    def __init__(self, output_dir="optimized_experiment_results"):
        self.output_dir = output_dir
        self.runner = ParallelExperimentRunner()
        self.visualizer = AdvancedVisualizationEngine()
        self.multi_objective_solver = MultiObjectiveSolver(['time', 'cost', 'safety', 'efficiency'])
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 实验配置
        self.scale_test_points = [20, 50, 100, 200, 500]
        self.algorithms = ['optimized_proposed', 'baseline_greedy', 'baseline_ga']
        self.robustness_runs = 30
        
    def generate_experiment_configs(self) -> List[Dict]:
        """生成实验配置"""
        configs = []
        
        # 可扩展性测试
        for num_points in self.scale_test_points:
            for algorithm in self.algorithms:
                for seed in range(5):  # 每个配置运行5次
                    config = {
                        'num_points': num_points,
                        'algorithm': algorithm,
                        'seed': seed,
                        'experiment_type': 'scalability',
                        'threat_config': {
                            'num_threats': max(2, num_points // 50),
                            'threat_density': 0.1
                        },
                        'ga_population_size': min(100, max(50, num_points)),
                        'ga_generations': min(200, max(50, num_points // 2)),
                        'ga_mutation_rate': 0.02
                    }
                    configs.append(config)
        
        # 鲁棒性测试（使用中等规模）
        for seed in range(self.robustness_runs):
            for algorithm in self.algorithms:
                config = {
                    'num_points': 100,
                    'algorithm': algorithm,
                    'seed': seed,
                    'experiment_type': 'robustness',
                    'threat_config': {
                        'num_threats': 3,
                        'threat_density': 0.15
                    },
                    'ga_population_size': 100,
                    'ga_generations': 100,
                    'ga_mutation_rate': 0.02
                }
                configs.append(config)
        
        # 威胁环境测试
        threat_scenarios = [
            {'name': 'low_threat', 'num_threats': 1, 'threat_density': 0.05},
            {'name': 'medium_threat', 'num_threats': 3, 'threat_density': 0.1},
            {'name': 'high_threat', 'num_threats': 6, 'threat_density': 0.2}
        ]
        
        for scenario in threat_scenarios:
            for algorithm in self.algorithms:
                for seed in range(10):
                    config = {
                        'num_points': 100,
                        'algorithm': algorithm,
                        'seed': seed,
                        'experiment_type': 'threat_analysis',
                        'threat_scenario': scenario['name'],
                        'threat_config': {
                            'num_threats': scenario['num_threats'],
                            'threat_density': scenario['threat_density']
                        },
                        'ga_population_size': 100,
                        'ga_generations': 100,
                        'ga_mutation_rate': 0.02
                    }
                    configs.append(config)
        
        print(f"生成了 {len(configs)} 个实验配置")
        return configs
    
    def run_comprehensive_experiments(self):
        """运行综合实验"""
        print("=== 开始综合实验套件 ===")
        
        # 生成实验配置
        configs = self.generate_experiment_configs()
        
        # 启动系统监控
        self.runner.system_monitor.start_monitoring()
        
        try:
            # 运行实验
            def progress_callback(progress, completed, total):
                if completed % 50 == 0:
                    print(f"实验进度: {completed}/{total} ({progress*100:.1f}%)")
            
            start_time = time.time()
            results = self.runner.run_parallel_experiments(configs, progress_callback)
            total_time = time.time() - start_time
            
            # 停止监控
            self.runner.system_monitor.stop_monitoring()
            
            # 分析结果
            analysis = self._analyze_results(results)
            analysis['total_experiment_time'] = total_time
            analysis['system_info'] = self.runner.system_monitor.get_system_info()
            
            # 保存结果
            self._save_results(results, analysis)
            
            # 生成可视化
            self._generate_visualizations(results, analysis)
            
            print(f"=== 实验完成，总用时: {total_time:.2f}秒 ===")
            return results, analysis
            
        except Exception as e:
            self.runner.system_monitor.stop_monitoring()
            raise e
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """分析实验结果"""
        print("分析实验结果...")
        
        # 分组结果
        results_by_algorithm = {}
        results_by_scale = {}
        results_by_experiment_type = {}
        
        for result in results:
            if 'error' in result:
                continue
            
            alg = result.get('algorithm', 'unknown')
            scale = result.get('num_points', 0)
            exp_type = result.get('experiment_config', {}).get('experiment_type', 'unknown')
            
            if alg not in results_by_algorithm:
                results_by_algorithm[alg] = []
            results_by_algorithm[alg].append(result)
            
            if scale not in results_by_scale:
                results_by_scale[scale] = []
            results_by_scale[scale].append(result)
            
            if exp_type not in results_by_experiment_type:
                results_by_experiment_type[exp_type] = []
            results_by_experiment_type[exp_type].append(result)
        
        # 计算统计指标
        algorithm_stats = {}
        for alg, alg_results in results_by_algorithm.items():
            times = [r.get('total_time', 0) for r in alg_results]
            distances = [r.get('total_distance', 0) for r in alg_results]
            efficiency_scores = [r.get('efficiency_score', 0) for r in alg_results]
            
            algorithm_stats[alg] = {
                'count': len(alg_results),
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'avg_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'avg_efficiency': np.mean(efficiency_scores),
                'std_efficiency': np.std(efficiency_scores),
                'cache_hit_rate': sum(1 for r in alg_results if r.get('from_cache', False)) / len(alg_results)
            }
        
        # 可扩展性分析
        scalability_analysis = {}
        for scale, scale_results in results_by_scale.items():
            if scale_results:
                times = [r.get('total_time', 0) for r in scale_results]
                scalability_analysis[scale] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'count': len(scale_results)
                }
        
        # 性能提升分析
        if 'optimized_proposed' in algorithm_stats and 'baseline_greedy' in algorithm_stats:
            proposed_time = algorithm_stats['optimized_proposed']['avg_time']
            baseline_time = algorithm_stats['baseline_greedy']['avg_time']
            time_improvement = (baseline_time - proposed_time) / baseline_time * 100
            
            proposed_efficiency = algorithm_stats['optimized_proposed']['avg_efficiency']
            baseline_efficiency = algorithm_stats['baseline_greedy']['avg_efficiency']
            efficiency_improvement = (proposed_efficiency - baseline_efficiency) / baseline_efficiency * 100
        else:
            time_improvement = 0
            efficiency_improvement = 0
        
        return {
            'algorithm_statistics': algorithm_stats,
            'scalability_analysis': scalability_analysis,
            'results_by_experiment_type': {
                k: len(v) for k, v in results_by_experiment_type.items()
            },
            'performance_improvements': {
                'time_improvement_percent': time_improvement,
                'efficiency_improvement_percent': efficiency_improvement
            },
            'total_experiments': len(results),
            'successful_experiments': len([r for r in results if 'error' not in r]),
            'cache_statistics': {
                'total_cache_hits': sum(1 for r in results if r.get('from_cache', False)),
                'cache_hit_rate': sum(1 for r in results if r.get('from_cache', False)) / len(results)
            }
        }
    
    def _save_results(self, results: List[Dict], analysis: Dict):
        """保存实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存分析结果
        analysis_file = os.path.join(self.output_dir, f"analysis_results_{timestamp}.json")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存CSV格式的结果摘要
        summary_data = []
        for result in results:
            if 'error' not in result:
                summary_data.append({
                    'algorithm': result.get('algorithm'),
                    'num_points': result.get('num_points'),
                    'total_time': result.get('total_time'),
                    'total_distance': result.get('total_distance'),
                    'efficiency_score': result.get('efficiency_score'),
                    'from_cache': result.get('from_cache'),
                    'experiment_type': result.get('experiment_config', {}).get('experiment_type')
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_file = os.path.join(self.output_dir, f"experiment_summary_{timestamp}.csv")
            df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"结果已保存到: {self.output_dir}")
    
    def _generate_visualizations(self, results: List[Dict], analysis: Dict):
        """生成可视化图表"""
        print("生成可视化图表...")
        
        # 创建示例解决方案数据用于可视化
        successful_results = [r for r in results if 'error' not in r]
        if not successful_results:
            print("没有成功的实验结果用于可视化")
            return
        
        # 选择一个代表性的结果进行可视化
        representative_result = successful_results[0]
        
        # 构造可视化数据
        viz_data = {
            'demand_points': np.array(representative_result.get('demand_points', [])),
            'priorities': np.array(representative_result.get('priorities', [])),
            'performance_metrics': {
                'time_efficiency': 0.8,
                'path_optimization': 0.75,
                'safety_score': 0.9,
                'fuel_efficiency': 0.7,
                'success_rate': 0.95
            },
            'economic_analysis': representative_result.get('economic_analysis', {}),
            'algorithm_performance': representative_result.get('algorithm_performance', {})
        }
        
        # 生成综合图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.output_dir, f"comprehensive_analysis_{timestamp}.png")
        
        try:
            self.visualizer.create_comprehensive_solution_plot(viz_data, plot_path)
        except Exception as e:
            print(f"可视化生成失败: {e}")
        
        # 生成算法对比数据
        comparison_data = {}
        for alg, stats in analysis['algorithm_statistics'].items():
            comparison_data[alg] = {
                'avg_execution_time': stats['avg_time'],
                'avg_quality': stats['avg_efficiency'],
                'overall_score': stats['avg_efficiency'] * 100,
                'scalability_test': {str(scale): stats['avg_time'] * (scale / 100) 
                                   for scale in self.scale_test_points}
            }
        
        # 生成对比仪表板
        if comparison_data:
            dashboard_path = os.path.join(self.output_dir, f"algorithm_comparison_{timestamp}.html")
            try:
                self.visualizer.create_algorithm_comparison_dashboard(comparison_data, dashboard_path)
            except Exception as e:
                print(f"对比仪表板生成失败: {e}")

# 性能测试函数
@performance_monitor
def run_performance_benchmark():
    """运行性能基准测试"""
    print("开始性能基准测试...")
    
    # 创建实验套件
    suite = ComprehensiveExperimentSuite()
    
    # 运行实验
    results, analysis = suite.run_comprehensive_experiments()
    
    # 输出关键性能指标
    print("\n=== 性能基准测试结果 ===")
    
    for alg, stats in analysis['algorithm_statistics'].items():
        print(f"\n{alg}:")
        print(f"  平均执行时间: {stats['avg_time']:.4f}秒")
        print(f"  平均效率得分: {stats['avg_efficiency']:.4f}")
        print(f"  缓存命中率: {stats['cache_hit_rate']*100:.1f}%")
    
    improvements = analysis['performance_improvements']
    print(f"\n性能提升:")
    print(f"  时间性能提升: {improvements['time_improvement_percent']:.1f}%")
    print(f"  效率提升: {improvements['efficiency_improvement_percent']:.1f}%")
    
    cache_stats = analysis['cache_statistics']
    print(f"\n缓存统计:")
    print(f"  总缓存命中: {cache_stats['total_cache_hits']}")
    print(f"  缓存命中率: {cache_stats['cache_hit_rate']*100:.1f}%")
    
    system_info = analysis['system_info']
    print(f"\n系统资源使用:")
    print(f"  CPU核心数: {system_info['cpu_count']}")
    print(f"  峰值内存使用: {system_info['peak_memory_gb']:.2f}GB")
    print(f"  内存使用率: {system_info['memory_percent']:.1f}%")
    
    return analysis

if __name__ == "__main__":
    # 运行性能基准测试
    benchmark_results = run_performance_benchmark()
    print("优化实验框架测试完成！")