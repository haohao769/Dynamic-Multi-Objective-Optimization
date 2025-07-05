# --- FILE: baseline_algorithms.py ---

import numpy as np
from scipy.spatial.distance import pdist, squareform
import config_enhanced as cfg

class GreedyTSP:
    """贪心算法求解TSP问题"""
    
    def solve(self, points):
        """贪心方法解决TSP问题
        
        参数:
            points: numpy数组，路径点坐标
            
        返回:
            路径顺序
        """
        n = len(points)
        # 控制中心为起点
        start_node = 0
        current_node = start_node
        unvisited = list(range(1, n))  # 排除起点
        route = [start_node]
        
        # 每次选择距离最近的下一个点
        while unvisited:
            min_dist = float('inf')
            next_node = None
            
            for node in unvisited:
                dist = np.linalg.norm(points[current_node] - points[node])
                if dist < min_dist:
                    min_dist = dist
                    next_node = node
                    
            route.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
            
        return np.array(route)
        
class NearestNeighborClustering:
    """最近邻聚类算法"""
    
    def __init__(self, n_clusters=cfg.NUM_CLUSTERS):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, X):
        """最近邻聚类
        
        参数:
            X: numpy数组，待聚类的点
            
        返回:
            self
        """
        n = len(X)
        
        # 随机选择初始中心点
        np.random.seed(cfg.RANDOM_SEED)
        indices = np.random.choice(n, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices].copy()
        
        # 分配聚类标签
        self.labels_ = np.zeros(n, dtype=int)
        for i in range(n):
            dists = [np.linalg.norm(X[i] - center) for center in self.cluster_centers_]
            self.labels_[i] = np.argmin(dists)
            
        return self
        
    def predict(self, X):
        """预测新数据的聚类标签
        
        参数:
            X: numpy数组，待预测的点
            
        返回:
            聚类标签
        """
        n = len(X)
        labels = np.zeros(n, dtype=int)
        
        for i in range(n):
            dists = [np.linalg.norm(X[i] - center) for center in self.cluster_centers_]
            labels[i] = np.argmin(dists)
            
        return labels
        
class SimpleGeneticAlgorithm:
    """简单遗传算法类"""
    
    def __init__(self, pop_size=50, generations=50, mutation_rate=0.1):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
    def optimize_tsp(self, points):
        """使用简单遗传算法优化TSP问题
        
        参数:
            points: numpy数组，路径点坐标
            
        返回:
            最优路径，总距离
        """
        n_points = len(points)
        
        # 计算距离矩阵
        dist_matrix = squareform(pdist(points))
        
        # 初始化种群（排除起点）
        population = []
        for _ in range(self.pop_size):
            # 随机路径，保持第一个点（控制中心）固定
            route = np.arange(1, n_points)
            np.random.shuffle(route)
            route = np.insert(route, 0, 0)  # 插入起点
            population.append(route)
            
        # 进化
        best_distance = float('inf')
        best_route = None
        
        for _ in range(self.generations):
            # 计算适应度（路径长度的倒数）
            fitness = []
            for route in population:
                distance = self._calculate_total_distance(route, dist_matrix)
                fitness.append(1.0 / distance)
                
                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()
                    
            # 创建下一代
            next_gen = []
            
            # 精英保留
            elite_size = max(2, int(self.pop_size * 0.1))
            elite_indices = np.argsort(fitness)[-elite_size:]
            for idx in elite_indices:
                next_gen.append(population[idx])
                
            # 填充剩余
            while len(next_gen) < self.pop_size:
                # 轮盘赌选择
                fitness_sum = sum(fitness)
                select_probs = [f / fitness_sum for f in fitness]
                
                parent1_idx = np.random.choice(len(population), p=select_probs)
                parent2_idx = np.random.choice(len(population), p=select_probs)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # 单点交叉（保持第一个点）
                crossover_point = np.random.randint(2, n_points)
                child = np.ones(n_points, dtype=int) * -1
                child[0] = 0  # 固定起点
                
                # 复制交叉点前的序列
                child[1:crossover_point] = parent1[1:crossover_point]
                
                # 填充剩余元素
                parent2_remaining = [x for x in parent2 if x not in child]
                for i in range(crossover_point, n_points):
                    child[i] = parent2_remaining.pop(0)
                    
                # 变异
                if np.random.rand() < self.mutation_rate:
                    # 随机交换两个位置（不包括起点）
                    i, j = np.random.choice(range(1, n_points), size=2, replace=False)
                    child[i], child[j] = child[j], child[i]
                    
                next_gen.append(child)
                
            population = next_gen
            
        return best_route, best_distance
        
    def _calculate_total_distance(self, route, dist_matrix):
        """计算路径总距离"""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += dist_matrix[route[i], route[i+1]]
            
        # 添加返回起点的距离
        total_distance += dist_matrix[route[-1], route[0]]
        
        return total_distance

class BasicClusteringAndRouting:
    """基本聚类和路由算法"""
    
    def __init__(self):
        self.clustering = NearestNeighborClustering()
        self.routing = GreedyTSP()
        
    def solve(self, points, control_center=cfg.CONTROL_CENTER):
        """解决问题
        
        参数:
            points: numpy数组，需求点坐标
            control_center: 控制中心坐标
            
        返回:
            聚类标签，聚类中心，卡车路径
        """
        # 聚类
        self.clustering.fit(points)
        labels = self.clustering.labels_
        centers = self.clustering.cluster_centers_
        
        # 求解卡车路径（从控制中心出发，经过各聚类中心）
        truck_points = np.vstack([control_center.reshape(1, -1), centers])
        truck_route = self.routing.solve(truck_points)
        
        return {
            'labels': labels,
            'centers': centers,
            'truck_route': truck_route,
            'truck_points': truck_points
        } 