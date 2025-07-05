# --- FILE: stage2_truck_routing.py ---

import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ThreadPoolExecutor
import config_enhanced as cfg
from performance_optimizers import (
    DistanceMatrixCache, 
    ParallelGeneticOptimizer,
    performance_monitor
)

# 全局缓存实例
_distance_cache = DistanceMatrixCache(max_cache_size=50)

@performance_monitor
def calculate_distance_matrix(points, restricted_areas=None):
    """计算考虑禁区的距离矩阵 - 优化版本
    
    参数:
        points: numpy数组，路径点坐标
        restricted_areas: 禁区列表
        
    返回:
        距离矩阵
    """
    # 使用缓存获取基础距离矩阵
    base_matrix = _distance_cache.get_distance_matrix(points)
    
    # 如果没有限制区域，直接返回缓存的矩阵
    if not restricted_areas:
        return base_matrix.copy()
    
    # 创建修改后的距离矩阵
    dist_matrix = base_matrix.copy()
    n = len(points)
    
    # 向量化威胁区域处理
    threat_multipliers = {'low': 1.5, 'medium': 2.0, 'high': 3.0, 'critical': 4.0}
    
    for area in restricted_areas:
        threat_level = area.get('threat_level', 'medium')
        multiplier = threat_multipliers.get(threat_level, 2.0)
        
        # 批量检查所有点对
        for i in range(n):
            for j in range(i+1, n):
                if line_intersects_circle(points[i], points[j], area['center'], area['radius']):
                    dist_matrix[i, j] *= multiplier
                    dist_matrix[j, i] = dist_matrix[i, j]
    
    return dist_matrix

def line_intersects_circle(p1, p2, circle_center, radius):
    """检查线段是否与圆相交
    
    参数:
        p1, p2: 线段的两个端点
        circle_center: 圆心坐标
        radius: 圆的半径
        
    返回:
        如果线段与圆相交，返回True；否则返回False
    """
    # 将问题转化为参数方程
    v = p2 - p1
    w = p1 - circle_center
    
    a = np.dot(v, v)
    b = 2 * np.dot(v, w)
    c = np.dot(w, w) - radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return False
    
    t1 = (-b - np.sqrt(discriminant)) / (2*a)
    t2 = (-b + np.sqrt(discriminant)) / (2*a)
    
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)

class OptimizedGeneticAlgorithm:
    """优化的遗传算法解决TSP问题"""
    
    def __init__(self, dist_matrix, pop_size=100, generations=100, mutation_rate=0.02, elite_size=20):
        self.dist_matrix = dist_matrix
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.n_cities = len(dist_matrix)
        
        # 性能优化组件
        self.parallel_optimizer = ParallelGeneticOptimizer()
        self.fitness_cache = {}
        self.convergence_history = []
        self.best_fitness_unchanged = 0
        self.early_stop_patience = 50
        
    def create_initial_population(self):
        """创建初始种群"""
        population = []
        for _ in range(self.pop_size):
            # 随机排列路径（保持第一个点不变，因为这是控制中心）
            route = np.arange(1, self.n_cities)
            np.random.shuffle(route)
            route = np.insert(route, 0, 0)  # 插入控制中心作为起点
            population.append(route)
        return population
        
    def calculate_fitness(self, route):
        """计算适应度（距离的倒数） - 优化版本"""
        # 使用缓存避免重复计算
        route_key = tuple(route)
        if route_key in self.fitness_cache:
            return self.fitness_cache[route_key]
        
        # 向量化距离计算
        route_indices = np.append(route, route[0])  # 添加返回起点
        distances = self.dist_matrix[route_indices[:-1], route_indices[1:]]
        total_distance = np.sum(distances)
        
        fitness = 1.0 / (1.0 + total_distance)  # 加1避免除零
        self.fitness_cache[route_key] = fitness
        return fitness
        
    def select_parents(self, population, fitnesses):
        """选择父代"""
        # 轮盘赌选择
        selection_probs = fitnesses / np.sum(fitnesses)
        selected_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
        return [population[i] for i in selected_indices]
        
    def crossover(self, parent1, parent2):
        """交叉操作"""
        # 有序交叉(OX)
        start, end = sorted(np.random.choice(len(parent1) - 1, 2) + 1)
        
        child = np.zeros(len(parent1), dtype=int)
        # 保持起点为控制中心
        child[0] = 0
        
        # 复制parent1的一部分
        child[start:end] = parent1[start:end]
        
        # 填充来自parent2的剩余部分
        idx = end
        for i in range(1, len(parent2)):
            if idx >= len(child):
                idx = 1
            if parent2[i] not in child:
                child[idx] = parent2[i]
                idx += 1
                
        return child
        
    def mutate(self, route):
        """变异操作"""
        for i in range(1, len(route)):
            if np.random.random() < self.mutation_rate:
                j = np.random.randint(1, len(route))
                route[i], route[j] = route[j], route[i]
        return route
        
    def next_generation(self, population):
        """生成下一代 - 并行优化版本"""
        # 并行计算适应度
        fitnesses = self.parallel_optimizer.parallel_fitness_evaluation(population, self.dist_matrix)
        
        # 记录收敛历史
        best_fitness = np.max(fitnesses)
        self.convergence_history.append(best_fitness)
        
        # 早停检查
        if len(self.convergence_history) > 1:
            if abs(self.convergence_history[-1] - self.convergence_history[-2]) < 1e-6:
                self.best_fitness_unchanged += 1
            else:
                self.best_fitness_unchanged = 0
        
        # 精英保留策略
        elite_indices = np.argsort(fitnesses)[-self.elite_size:]
        elites = [population[i].copy() for i in elite_indices]
        
        # 使用轮盘赌选择
        selection_probs = fitnesses / np.sum(fitnesses)
        selected_indices = np.random.choice(len(population), size=self.pop_size-self.elite_size, p=selection_probs)
        selected_parents = [population[i] for i in selected_indices]
        
        # 交叉和变异
        next_gen = elites.copy()
        for i in range(0, len(selected_parents), 2):
            if i+1 < len(selected_parents):
                child1 = self.crossover(selected_parents[i], selected_parents[i+1])
                child2 = self.crossover(selected_parents[i+1], selected_parents[i])
                next_gen.append(self.mutate(child1))
                if len(next_gen) < self.pop_size:
                    next_gen.append(self.mutate(child2))
        
        # 确保种群大小正确
        next_gen = next_gen[:self.pop_size]
        
        return next_gen
        
    def solve(self):
        """解决TSP问题 - 优化版本"""
        import time
        start_time = time.time()
        
        population = self.create_initial_population()
        
        # 记录初始最佳适应度
        initial_fitnesses = [self.calculate_fitness(route) for route in population]
        initial_best = np.max(initial_fitnesses)
        
        for generation in range(self.generations):
            population = self.next_generation(population)
            
            # 早停检查
            if self.best_fitness_unchanged >= self.early_stop_patience:
                print(f"算法在第{generation}代收敛，启动早停")
                break
        
        # 获取最佳路径
        final_fitnesses = [self.calculate_fitness(route) for route in population]
        best_idx = np.argmax(final_fitnesses)
        best_route = population[best_idx]
        best_fitness = final_fitnesses[best_idx]
        
        # 计算实际距离
        route_indices = np.append(best_route, best_route[0])
        distances = self.dist_matrix[route_indices[:-1], route_indices[1:]]
        best_distance = np.sum(distances)
        
        execution_time = time.time() - start_time
        
        # 返回详细结果
        return {
            'route': best_route,
            'distance': best_distance,
            'fitness': best_fitness,
            'execution_time': execution_time,
            'generations_run': len(self.convergence_history),
            'convergence_history': self.convergence_history,
            'improvement_ratio': (best_fitness - initial_best) / initial_best if initial_best > 0 else 0
        }

@performance_monitor
def solve_tsp_with_ga(points, restricted_areas=None, seed=cfg.RANDOM_SEED):
    """使用优化的遗传算法解决TSP问题
    
    参数:
        points: 路径点坐标，第一个点为控制中心
        restricted_areas: 禁区列表
        seed: 随机种子
        
    返回:
        最佳路径，路径总成本
    """
    np.random.seed(seed)
    
    # 计算距离矩阵
    dist_matrix = calculate_distance_matrix(points, restricted_areas)
    
    # 使用优化的遗传算法求解
    ga = OptimizedGeneticAlgorithm(
        dist_matrix=dist_matrix,
        pop_size=cfg.GA_POPULATION_SIZE,
        generations=cfg.GA_GENERATIONS,
        mutation_rate=cfg.GA_MUTATION_RATE,
        elite_size=cfg.GA_ELITE_SIZE
    )
    
    result = ga.solve()
    
    # 返回详细结果
    return {
        'route': result['route'],
        'total_distance': result['distance'],
        'execution_time': result['execution_time'],
        'algorithm_performance': {
            'generations_run': result['generations_run'],
            'improvement_ratio': result['improvement_ratio'],
            'convergence_history': result['convergence_history']
        }
    }

# 测试代码
if __name__ == "__main__":
    # 简单测试
    np.random.seed(42)
    test_points = np.random.rand(10, 2) * 20
    test_areas = [
        {'center': np.array([5, 5]), 'radius': 2, 'threat_level': 'medium'},
        {'center': np.array([15, 15]), 'radius': 3, 'threat_level': 'high'}
    ]
    
    route, distance = solve_tsp_with_ga(test_points, test_areas)
    print(f"最佳路径: {route}")
    print(f"总距离: {distance}") 