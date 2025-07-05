# --- FILE: multi_objective_optimizer.py ---

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Callable, Optional, Any
import json
import time
from performance_optimizers import performance_monitor
import warnings
warnings.filterwarnings('ignore')

class ParetoFrontAnalyzer:
    """帕累托前沿分析器"""
    
    def __init__(self):
        self.pareto_solutions = []
        self.dominated_solutions = []
        
    def dominates(self, solution1: Dict, solution2: Dict, objectives: List[str], 
                  minimization_objectives: List[str] = None) -> bool:
        """判断solution1是否支配solution2"""
        if minimization_objectives is None:
            minimization_objectives = ['time', 'cost', 'distance']
        
        at_least_one_better = False
        
        for obj in objectives:
            if obj not in solution1 or obj not in solution2:
                continue
                
            val1, val2 = solution1[obj], solution2[obj]
            
            if obj in minimization_objectives:
                # 最小化目标：越小越好
                if val1 > val2:
                    return False
                elif val1 < val2:
                    at_least_one_better = True
            else:
                # 最大化目标：越大越好
                if val1 < val2:
                    return False
                elif val1 > val2:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def find_pareto_front(self, solutions: List[Dict], objectives: List[str]) -> List[Dict]:
        """找到帕累托前沿"""
        pareto_front = []
        
        for i, solution in enumerate(solutions):
            is_dominated = False
            
            for j, other_solution in enumerate(solutions):
                if i != j and self.dominates(other_solution, solution, objectives):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(solution)
        
        self.pareto_solutions = pareto_front
        return pareto_front
    
    def calculate_hypervolume(self, pareto_front: List[Dict], objectives: List[str], 
                            reference_point: Dict = None) -> float:
        """计算超体积指标"""
        if not pareto_front:
            return 0.0
        
        # 如果没有提供参考点，使用最差值
        if reference_point is None:
            reference_point = {}
            for obj in objectives:
                values = [sol[obj] for sol in pareto_front if obj in sol]
                if values:
                    reference_point[obj] = max(values) * 1.1  # 比最差值稍大
        
        # 简化的超体积计算（二维情况）
        if len(objectives) == 2:
            obj1, obj2 = objectives
            points = [(sol[obj1], sol[obj2]) for sol in pareto_front 
                     if obj1 in sol and obj2 in sol]
            
            if not points:
                return 0.0
            
            # 排序点
            points.sort()
            
            hypervolume = 0.0
            prev_x = reference_point[obj1]
            
            for x, y in points:
                width = prev_x - x
                height = reference_point[obj2] - y
                hypervolume += width * height
                prev_x = x
            
            return max(0, hypervolume)
        
        return 0.0  # 高维情况暂时返回0

class AdaptiveWeightOptimizer:
    """自适应权重优化器"""
    
    def __init__(self, objectives: List[str], initial_weights: Dict[str, float] = None):
        self.objectives = objectives
        self.weights = initial_weights or {obj: 1.0 for obj in objectives}
        self.weight_history = []
        self.performance_history = []
        self.learning_rate = 0.1
        
    def normalize_weights(self):
        """归一化权重"""
        total = sum(self.weights.values())
        if total > 0:
            for obj in self.weights:
                self.weights[obj] /= total
    
    def update_weights_based_on_feedback(self, solution_performance: Dict, 
                                       user_preferences: Dict = None):
        """基于反馈更新权重"""
        self.performance_history.append(solution_performance.copy())
        self.weight_history.append(self.weights.copy())
        
        # 基于用户偏好调整
        if user_preferences:
            for obj, preference in user_preferences.items():
                if obj in self.weights:
                    # preference: -1(降低重要性) 到 1(提高重要性)
                    adjustment = preference * self.learning_rate
                    self.weights[obj] *= (1 + adjustment)
        
        # 基于性能趋势调整
        if len(self.performance_history) >= 2:
            current = self.performance_history[-1]
            previous = self.performance_history[-2]
            
            for obj in self.objectives:
                if obj in current and obj in previous:
                    improvement = (current[obj] - previous[obj]) / max(abs(previous[obj]), 1e-6)
                    
                    # 如果某个目标改善不明显，增加其权重
                    if abs(improvement) < 0.01:
                        self.weights[obj] *= 1.05
        
        self.normalize_weights()
    
    def get_optimal_weights_ga(self, solutions: List[Dict]) -> Dict[str, float]:
        """使用遗传算法寻找最优权重组合"""
        if len(solutions) < 2:
            return self.weights
        
        def evaluate_weights(weights_array):
            """评估权重组合的效果"""
            weights_dict = {obj: w for obj, w in zip(self.objectives, weights_array)}
            
            # 计算加权得分
            scores = []
            for solution in solutions:
                score = sum(weights_dict.get(obj, 0) * solution.get(obj, 0) 
                          for obj in self.objectives)
                scores.append(score)
            
            # 返回得分的标准差（越小越好，表示权重分配更均衡）
            return np.std(scores)
        
        # 权重约束：和为1，每个权重在[0.01, 1]之间
        bounds = [(0.01, 1.0) for _ in self.objectives]
        constraints = {'type': 'eq', 'fun': lambda x: sum(x) - 1.0}
        
        # 初始值
        x0 = np.array([self.weights[obj] for obj in self.objectives])
        x0 = x0 / np.sum(x0)  # 归一化
        
        try:
            result = minimize(evaluate_weights, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimal_weights = {obj: w for obj, w in zip(self.objectives, result.x)}
                return optimal_weights
        except:
            pass
        
        return self.weights

class MultiObjectiveSolver:
    """多目标优化求解器"""
    
    def __init__(self, objectives: List[str], constraints: List[Callable] = None):
        self.objectives = objectives
        self.constraints = constraints or []
        self.pareto_analyzer = ParetoFrontAnalyzer()
        self.weight_optimizer = AdaptiveWeightOptimizer(objectives)
        self.solution_history = []
        
    def add_constraint(self, constraint_func: Callable, constraint_type: str = 'ineq'):
        """添加约束条件"""
        self.constraints.append({'type': constraint_type, 'fun': constraint_func})
    
    def evaluate_solution(self, solution_data: Dict, weights: Dict[str, float] = None) -> Dict:
        """评估解决方案"""
        if weights is None:
            weights = self.weight_optimizer.weights
        
        # 计算各目标值
        objectives_values = {}
        
        # 时间目标
        total_time = solution_data.get('truck_time', 0) + \
                    max([mission.get('flight_time', 0) 
                         for mission in solution_data.get('drone_missions', [])], default=0)
        objectives_values['time'] = total_time
        
        # 距离目标
        objectives_values['distance'] = solution_data.get('total_distance', 0)
        
        # 成本目标
        objectives_values['cost'] = solution_data.get('total_cost', 0)
        
        # 安全性目标（威胁暴露度的倒数）
        threat_exposure = solution_data.get('threat_exposure', 1)
        objectives_values['safety'] = 1.0 / (1.0 + threat_exposure)
        
        # 成功率目标
        objectives_values['success_rate'] = solution_data.get('success_probability', 0.9)
        
        # 计算加权得分
        weighted_score = sum(weights.get(obj, 0) * val 
                           for obj, val in objectives_values.items())
        
        # 应用约束惩罚
        constraint_penalty = 0
        for constraint in self.constraints:
            violation = constraint['fun'](solution_data)
            if constraint['type'] == 'ineq' and violation < 0:
                constraint_penalty += abs(violation) * 1000
            elif constraint['type'] == 'eq' and abs(violation) > 1e-6:
                constraint_penalty += abs(violation) * 1000
        
        final_score = weighted_score - constraint_penalty
        
        result = {
            'objectives': objectives_values,
            'weighted_score': weighted_score,
            'constraint_penalty': constraint_penalty,
            'final_score': final_score,
            'weights_used': weights.copy()
        }
        
        return result
    
    def optimize_multi_objective(self, problem_data: Dict, algorithm_variants: List[Callable],
                                population_size: int = 50, generations: int = 100) -> Dict:
        """多目标优化主函数"""
        print("开始多目标优化...")
        start_time = time.time()
        
        all_solutions = []
        
        # 运行不同算法变体
        for i, algorithm in enumerate(algorithm_variants):
            print(f"运行算法变体 {i+1}/{len(algorithm_variants)}")
            
            # 生成多个解决方案
            for seed in range(population_size // len(algorithm_variants)):
                try:
                    solution = algorithm(problem_data, seed=seed)
                    evaluation = self.evaluate_solution(solution)
                    
                    # 合并解决方案和评估结果
                    combined_solution = {**solution, **evaluation}
                    all_solutions.append(combined_solution)
                    
                except Exception as e:
                    print(f"算法 {i+1} 种子 {seed} 执行失败: {e}")
                    continue
        
        if not all_solutions:
            raise ValueError("所有算法变体都执行失败")
        
        # 找到帕累托前沿
        pareto_front = self.pareto_analyzer.find_pareto_front(all_solutions, self.objectives)
        
        # 更新权重
        if len(all_solutions) > 1:
            optimal_weights = self.weight_optimizer.get_optimal_weights_ga(all_solutions)
            self.weight_optimizer.weights = optimal_weights
        
        # 重新评估帕累托前沿解决方案
        for solution in pareto_front:
            new_eval = self.evaluate_solution(solution, self.weight_optimizer.weights)
            solution.update(new_eval)
        
        # 选择最佳妥协解
        if pareto_front:
            best_compromise = max(pareto_front, key=lambda x: x.get('final_score', 0))
        else:
            best_compromise = max(all_solutions, key=lambda x: x.get('final_score', 0))
        
        # 计算超体积
        hypervolume = self.pareto_analyzer.calculate_hypervolume(
            pareto_front, self.objectives[:2])  # 只使用前两个目标
        
        optimization_time = time.time() - start_time
        
        result = {
            'best_solution': best_compromise,
            'pareto_front': pareto_front,
            'all_solutions': all_solutions,
            'pareto_front_size': len(pareto_front),
            'hypervolume': hypervolume,
            'optimal_weights': self.weight_optimizer.weights,
            'optimization_time': optimization_time,
            'convergence_metrics': {
                'total_evaluations': len(all_solutions),
                'pareto_ratio': len(pareto_front) / len(all_solutions),
                'weight_stability': self._calculate_weight_stability()
            }
        }
        
        self.solution_history.append(result)
        return result
    
    def _calculate_weight_stability(self) -> float:
        """计算权重稳定性"""
        if len(self.weight_optimizer.weight_history) < 2:
            return 1.0
        
        recent_weights = self.weight_optimizer.weight_history[-5:]  # 最近5次
        
        stabilities = []
        for obj in self.objectives:
            values = [w.get(obj, 0) for w in recent_weights]
            if len(values) > 1:
                stability = 1.0 - (np.std(values) / (np.mean(values) + 1e-6))
                stabilities.append(max(0, stability))
        
        return np.mean(stabilities) if stabilities else 1.0
    
    def interactive_weight_adjustment(self, current_solution: Dict) -> Dict[str, float]:
        """交互式权重调整"""
        print("\n=== 交互式权重调整 ===")
        print("当前目标值:")
        
        objectives_values = current_solution.get('objectives', {})
        for obj, value in objectives_values.items():
            print(f"  {obj}: {value:.4f}")
        
        print(f"\n当前权重:")
        for obj, weight in self.weight_optimizer.weights.items():
            print(f"  {obj}: {weight:.4f}")
        
        print("\n请输入权重调整 (-1到1, 0表示不变):")
        
        adjustments = {}
        for obj in self.objectives:
            try:
                adj = float(input(f"{obj} 调整: ") or 0)
                adjustments[obj] = np.clip(adj, -1, 1)
            except ValueError:
                adjustments[obj] = 0
        
        # 应用调整
        self.weight_optimizer.update_weights_based_on_feedback(
            objectives_values, adjustments)
        
        return self.weight_optimizer.weights

class ScenarioAdaptiveOptimizer:
    """场景自适应优化器"""
    
    def __init__(self):
        self.scenario_weights = {
            'normal': {'time': 0.3, 'cost': 0.3, 'safety': 0.2, 'success_rate': 0.2},
            'high_threat': {'time': 0.2, 'cost': 0.2, 'safety': 0.4, 'success_rate': 0.2},
            'emergency': {'time': 0.5, 'cost': 0.1, 'safety': 0.2, 'success_rate': 0.2},
            'cost_sensitive': {'time': 0.2, 'cost': 0.5, 'safety': 0.15, 'success_rate': 0.15}
        }
        
    def detect_scenario(self, problem_data: Dict) -> str:
        """检测当前场景类型"""
        threat_zones = problem_data.get('threat_zones', [])
        priorities = problem_data.get('priorities', [])
        
        # 威胁评估
        high_threat_count = sum(1 for zone in threat_zones 
                               if zone.get('threat_level') in ['high', 'critical'])
        threat_ratio = high_threat_count / max(len(threat_zones), 1)
        
        # 紧急任务评估
        high_priority_ratio = sum(1 for p in priorities if p >= 4) / max(len(priorities), 1)
        
        # 场景判断
        if threat_ratio > 0.3:
            return 'high_threat'
        elif high_priority_ratio > 0.4:
            return 'emergency'
        elif problem_data.get('cost_constraint', False):
            return 'cost_sensitive'
        else:
            return 'normal'
    
    def get_scenario_weights(self, scenario: str) -> Dict[str, float]:
        """获取场景对应的权重"""
        return self.scenario_weights.get(scenario, self.scenario_weights['normal'])

# 测试和使用示例
@performance_monitor
def test_multi_objective_optimization():
    """测试多目标优化系统"""
    print("测试多目标优化系统...")
    
    # 创建测试数据
    test_solutions = []
    for i in range(20):
        solution = {
            'time': np.random.uniform(100, 500),
            'cost': np.random.uniform(1000, 5000),
            'distance': np.random.uniform(50, 200),
            'safety': np.random.uniform(0.6, 1.0),
            'success_rate': np.random.uniform(0.8, 1.0),
            'truck_time': np.random.uniform(50, 200),
            'drone_missions': [{'flight_time': np.random.uniform(30, 120)}],
            'total_distance': np.random.uniform(50, 200),
            'total_cost': np.random.uniform(1000, 5000),
            'threat_exposure': np.random.uniform(0, 5),
            'success_probability': np.random.uniform(0.8, 1.0)
        }
        test_solutions.append(solution)
    
    # 创建多目标求解器
    objectives = ['time', 'cost', 'safety', 'success_rate']
    solver = MultiObjectiveSolver(objectives)
    
    # 测试帕累托前沿分析
    pareto_front = solver.pareto_analyzer.find_pareto_front(test_solutions, objectives)
    print(f"帕累托前沿包含 {len(pareto_front)} 个解决方案")
    
    # 测试超体积计算
    hypervolume = solver.pareto_analyzer.calculate_hypervolume(
        pareto_front, ['time', 'cost'])
    print(f"超体积指标: {hypervolume:.4f}")
    
    # 测试自适应权重
    solver.weight_optimizer.update_weights_based_on_feedback(
        test_solutions[0], {'time': -0.2, 'safety': 0.3})
    print("自适应权重:", solver.weight_optimizer.weights)
    
    # 测试场景自适应
    scenario_optimizer = ScenarioAdaptiveOptimizer()
    test_problem = {
        'threat_zones': [
            {'threat_level': 'high'}, {'threat_level': 'critical'}
        ],
        'priorities': [5, 4, 3, 5, 4]
    }
    
    scenario = scenario_optimizer.detect_scenario(test_problem)
    scenario_weights = scenario_optimizer.get_scenario_weights(scenario)
    print(f"检测到场景: {scenario}")
    print(f"场景权重: {scenario_weights}")
    
    return {
        'pareto_front_size': len(pareto_front),
        'hypervolume': hypervolume,
        'detected_scenario': scenario,
        'adaptive_weights': solver.weight_optimizer.weights
    }

if __name__ == "__main__":
    results = test_multi_objective_optimization()
    print("\n多目标优化测试完成!")
    print(f"测试结果: {json.dumps(results, indent=2, ensure_ascii=False)}")