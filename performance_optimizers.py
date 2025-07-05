# --- FILE: performance_optimizers.py ---

import numpy as np
from scipy.spatial import cKDTree, distance_matrix
from scipy.spatial.distance import cdist
from functools import lru_cache
import numba
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import pickle
from typing import List, Tuple, Dict, Optional, Union
import time

class DistanceMatrixCache:
    """高性能距离矩阵缓存系统"""
    
    def __init__(self, max_cache_size=100):
        self.cache = {}
        self.access_count = {}
        self.max_size = max_cache_size
    
    def _get_points_hash(self, points):
        """生成点集的唯一哈希值"""
        points_bytes = points.tobytes()
        return hashlib.md5(points_bytes).hexdigest()
    
    def get_distance_matrix(self, points):
        """获取或计算距离矩阵"""
        points_hash = self._get_points_hash(points)
        
        if points_hash in self.cache:
            self.access_count[points_hash] += 1
            return self.cache[points_hash]
        
        # 缓存已满时清理最少使用的项
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        
        # 计算距离矩阵
        dist_matrix = cdist(points, points, metric='euclidean')
        self.cache[points_hash] = dist_matrix
        self.access_count[points_hash] = 1
        
        return dist_matrix
    
    def _cleanup_cache(self):
        """清理缓存中最少使用的项"""
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
        to_remove = sorted_items[:self.max_size // 3]  # 移除1/3最少使用的项
        
        for points_hash, _ in to_remove:
            del self.cache[points_hash]
            del self.access_count[points_hash]

class OptimizedSpatialIndex:
    """优化的空间索引系统"""
    
    def __init__(self, points, leaf_size=30):
        self.points = np.array(points)
        self.tree = cKDTree(self.points, leafsize=leaf_size)
        self.n_points = len(points)
    
    def query_radius_batch(self, centers, radii):
        """批量半径查询，复杂度从O(n²)降至O(n log n)"""
        results = []
        for center, radius in zip(centers, radii):
            indices = self.tree.query_ball_point(center, radius)
            results.append(indices)
        return results
    
    def nearest_neighbors_batch(self, query_points, k=1):
        """批量最近邻查询"""
        distances, indices = self.tree.query(query_points, k=k)
        return distances, indices
    
    def range_query_optimized(self, min_bounds, max_bounds):
        """优化的范围查询"""
        indices = []
        for i, point in enumerate(self.points):
            if np.all(point >= min_bounds) and np.all(point <= max_bounds):
                indices.append(i)
        return indices

@numba.jit(nopython=True, parallel=True)
def vectorized_distance_calculation(points1, points2):
    """使用Numba加速的向量化距离计算"""
    n1, n2 = len(points1), len(points2)
    distances = np.zeros((n1, n2))
    
    for i in numba.prange(n1):
        for j in range(n2):
            dist = 0.0
            for k in range(points1.shape[1]):
                diff = points1[i, k] - points2[j, k]
                dist += diff * diff
            distances[i, j] = np.sqrt(dist)
    
    return distances

@numba.jit(nopython=True, parallel=True)
def threat_zone_check_vectorized(points, threat_centers, threat_radii):
    """向量化的威胁区域检查"""
    n_points = len(points)
    n_threats = len(threat_centers)
    threat_matrix = np.zeros((n_points, n_threats), dtype=numba.boolean)
    
    for i in numba.prange(n_points):
        for j in range(n_threats):
            dist_sq = 0.0
            for k in range(points.shape[1]):
                diff = points[i, k] - threat_centers[j, k]
                dist_sq += diff * diff
            threat_matrix[i, j] = dist_sq <= threat_radii[j] * threat_radii[j]
    
    return threat_matrix

class ParallelGeneticOptimizer:
    """并行遗传算法优化器"""
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.distance_cache = DistanceMatrixCache()
    
    def parallel_fitness_evaluation(self, population, distance_matrix):
        """并行适应度评估"""
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(self._calculate_individual_fitness, individual, distance_matrix)
                for individual in population
            ]
            fitness_values = [future.result() for future in futures]
        return np.array(fitness_values)
    
    def _calculate_individual_fitness(self, individual, distance_matrix):
        """计算单个个体的适应度"""
        total_distance = 0.0
        for i in range(len(individual) - 1):
            total_distance += distance_matrix[individual[i], individual[i + 1]]
        # 添加返回起点的距离
        total_distance += distance_matrix[individual[-1], individual[0]]
        return 1.0 / (1.0 + total_distance)  # 适应度值

class OptimizedAStarPlanner:
    """优化的A*路径规划器"""
    
    def __init__(self, grid_resolution=0.1):
        self.resolution = grid_resolution
        self.movement_cache = self._precompute_movements()
        self.heuristic_cache = {}
    
    def _precompute_movements(self):
        """预计算移动方向，避免重复计算"""
        movements = []
        # 26个3D移动方向（6个面 + 12个边 + 8个角）
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    movements.append((dx, dy, dz))
        return movements
    
    @lru_cache(maxsize=10000)
    def heuristic_distance(self, pos1, pos2):
        """缓存的启发式距离计算"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def optimized_astar_search(self, start, goal, obstacle_checker, terrain_model):
        """优化的A*搜索算法"""
        start_time = time.time()
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic_distance(start, goal)}
        closed_set = set()
        
        nodes_explored = 0
        
        while open_set:
            current_f, current = min(open_set)
            open_set.remove((current_f, current))
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            nodes_explored += 1
            
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                search_time = time.time() - start_time
                return {
                    'path': path,
                    'nodes_explored': nodes_explored,
                    'search_time': search_time,
                    'path_length': len(path)
                }
            
            for dx, dy, dz in self.movement_cache:
                neighbor = (
                    current[0] + dx * self.resolution,
                    current[1] + dy * self.resolution,
                    current[2] + dz * self.resolution
                )
                
                if neighbor in closed_set:
                    continue
                
                if obstacle_checker(neighbor) or not terrain_model.is_valid_position(neighbor):
                    continue
                
                tentative_g = g_score[current] + self.heuristic_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic_distance(neighbor, goal)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        return {'path': [], 'nodes_explored': nodes_explored, 'search_time': time.time() - start_time}
    
    def _reconstruct_path(self, came_from, current):
        """重构路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

class EconomicAnalyzer:
    """经济效益分析器"""
    
    def __init__(self):
        # 成本参数 (单位：元)
        self.fuel_cost_per_liter = 7.5
        self.electricity_cost_per_kwh = 0.8
        self.truck_maintenance_per_km = 1.2
        self.drone_maintenance_per_flight_hour = 50.0
        self.operator_cost_per_hour = 100.0
        self.insurance_cost_per_day = 200.0
        
        # 效率参数
        self.truck_fuel_consumption = 0.3  # L/km
        self.drone_power_consumption = 2.5  # kWh/hour
        self.average_truck_speed = 60.0  # km/h
        self.average_drone_speed = 50.0  # km/h
    
    def calculate_comprehensive_cost(self, solution_data):
        """计算综合成本分析"""
        truck_cost = self._calculate_truck_costs(solution_data['truck_route'])
        drone_costs = [self._calculate_drone_costs(mission) for mission in solution_data['drone_missions']]
        
        total_fuel_cost = truck_cost['fuel_cost']
        total_electricity_cost = sum(dc['electricity_cost'] for dc in drone_costs)
        total_maintenance_cost = truck_cost['maintenance_cost'] + sum(dc['maintenance_cost'] for dc in drone_costs)
        total_operator_cost = truck_cost['operator_cost'] + sum(dc['operator_cost'] for dc in drone_costs)
        total_insurance_cost = self.insurance_cost_per_day
        
        return {
            'total_cost': total_fuel_cost + total_electricity_cost + total_maintenance_cost + total_operator_cost + total_insurance_cost,
            'fuel_cost': total_fuel_cost,
            'electricity_cost': total_electricity_cost,
            'maintenance_cost': total_maintenance_cost,
            'operator_cost': total_operator_cost,
            'insurance_cost': total_insurance_cost,
            'cost_breakdown': {
                'truck': truck_cost,
                'drones': drone_costs
            }
        }
    
    def _calculate_truck_costs(self, route_data):
        """计算卡车成本"""
        distance_km = route_data['total_distance']
        time_hours = distance_km / self.average_truck_speed
        
        fuel_cost = distance_km * self.truck_fuel_consumption * self.fuel_cost_per_liter
        maintenance_cost = distance_km * self.truck_maintenance_per_km
        operator_cost = time_hours * self.operator_cost_per_hour
        
        return {
            'fuel_cost': fuel_cost,
            'maintenance_cost': maintenance_cost,
            'operator_cost': operator_cost,
            'total_distance': distance_km,
            'total_time': time_hours
        }
    
    def _calculate_drone_costs(self, mission_data):
        """计算无人机成本"""
        flight_time_hours = mission_data['flight_time'] / 3600.0  # 转换为小时
        
        electricity_cost = flight_time_hours * self.drone_power_consumption * self.electricity_cost_per_kwh
        maintenance_cost = flight_time_hours * self.drone_maintenance_per_flight_hour
        operator_cost = flight_time_hours * self.operator_cost_per_hour * 0.5  # 无人机操作员成本较低
        
        return {
            'electricity_cost': electricity_cost,
            'maintenance_cost': maintenance_cost,
            'operator_cost': operator_cost,
            'flight_time': flight_time_hours
        }

class MultiObjectiveOptimizer:
    """多目标优化权衡机制"""
    
    def __init__(self, objectives=['time', 'cost', 'safety', 'success_rate']):
        self.objectives = objectives
        self.weights = {obj: 1.0 for obj in objectives}
        self.normalization_factors = {obj: 1.0 for obj in objectives}
    
    def set_weights(self, weight_dict):
        """设置目标权重"""
        for obj, weight in weight_dict.items():
            if obj in self.objectives:
                self.weights[obj] = weight
    
    def calculate_weighted_score(self, solution_metrics):
        """计算多目标加权得分"""
        normalized_scores = {}
        
        for obj in self.objectives:
            if obj in solution_metrics:
                raw_score = solution_metrics[obj]
                normalized_score = raw_score / self.normalization_factors[obj]
                normalized_scores[obj] = normalized_score
        
        # 计算加权得分
        weighted_sum = sum(
            self.weights[obj] * normalized_scores.get(obj, 0)
            for obj in self.objectives
        )
        
        total_weight = sum(self.weights.values())
        final_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        return {
            'final_score': final_score,
            'normalized_scores': normalized_scores,
            'weights_used': self.weights.copy()
        }
    
    def pareto_front_analysis(self, solutions_list):
        """帕累托前沿分析"""
        pareto_solutions = []
        
        for i, solution in enumerate(solutions_list):
            is_dominated = False
            
            for j, other_solution in enumerate(solutions_list):
                if i != j and self._dominates(other_solution, solution):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(solution)
        
        return pareto_solutions
    
    def _dominates(self, solution1, solution2):
        """判断solution1是否支配solution2"""
        better_in_at_least_one = False
        
        for obj in self.objectives:
            if obj in solution1 and obj in solution2:
                val1, val2 = solution1[obj], solution2[obj]
                
                # 对于成本和时间，越小越好；对于成功率和安全性，越大越好
                if obj in ['cost', 'time']:
                    if val1 > val2:
                        return False
                    elif val1 < val2:
                        better_in_at_least_one = True
                else:  # safety, success_rate等
                    if val1 < val2:
                        return False
                    elif val1 > val2:
                        better_in_at_least_one = True
        
        return better_in_at_least_one

# 性能监控装饰器
def performance_monitor(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0  # 简化版本，可使用psutil获取实际内存使用
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        if hasattr(result, '__dict__'):
            result.performance_metrics = {
                'execution_time': execution_time,
                'function_name': func.__name__
            }
        
        print(f"[性能监控] {func.__name__}: {execution_time:.4f}秒")
        return result
    
    return wrapper