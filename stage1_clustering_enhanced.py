# --- FILE: stage1_clustering_enhanced.py ---

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import config_enhanced as cfg
from performance_optimizers import (
    OptimizedSpatialIndex, 
    threat_zone_check_vectorized,
    performance_monitor
)

class DynamicThreatModel:
    """动态威胁模型，模拟战场环境变化"""
    def __init__(self, threat_zones=None, update_interval=300):
        self.threat_zones = threat_zones if threat_zones else []
        self.update_interval = update_interval  # 威胁更新间隔（秒）
        self.time_elapsed = 0
        self.threat_levels = ['low', 'medium', 'high', 'critical']
        
    def update_threats(self, time_delta, intelligence_updates=None):
        """基于时间和情报更新威胁区域"""
        self.time_elapsed += time_delta
        
        if intelligence_updates:
            # 基于实时情报更新
            self._process_intelligence(intelligence_updates)
            
        # 周期性随机更新
        if self.time_elapsed >= self.update_interval:
            self._random_threat_evolution()
            self.time_elapsed = 0
            
    def _process_intelligence(self, updates):
        """处理情报更新"""
        for update in updates:
            if update['type'] == 'new_threat':
                self.threat_zones.append({
                    'center': update['location'],
                    'radius': update['radius'],
                    'threat_level': update['level'],
                    'mobility': update.get('mobility', 'static')
                })
            elif update['type'] == 'threat_removed':
                # 移除威胁
                self.threat_zones = [z for z in self.threat_zones 
                                   if np.linalg.norm(z['center'] - update['location']) > 0.1]
                                   
    def _random_threat_evolution(self):
        """威胁随机演化"""
        for zone in self.threat_zones:
            if zone.get('mobility') == 'mobile':
                # 移动威胁区域
                offset = np.random.normal(0, 1.5, 2)
                zone['center'] += offset
                
            # 威胁等级变化
            if np.random.rand() < 0.3:
                current_level_idx = self.threat_levels.index(zone['threat_level'])
                change = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
                new_idx = np.clip(current_level_idx + change, 0, len(self.threat_levels)-1)
                zone['threat_level'] = self.threat_levels[new_idx]

@performance_monitor
def generate_demand_points_with_priority(num_points, seed=cfg.RANDOM_SEED):
    """生成带优先级的需求点 - 优化版本，使用空间索引"""
    np.random.seed(seed)
    points = []
    priorities = []  # 1-5, 5为最高优先级
    
    # 使用更高效的点生成策略
    max_attempts = num_points * 10  # 防止无限循环
    attempts = 0
    
    while len(points) < num_points and attempts < max_attempts:
        attempts += 1
        angle = 2 * np.pi * np.random.rand()
        dist = np.random.uniform(cfg.MIN_DIST_CENTER, cfg.MAX_DIST_CENTER)
        new_point = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        # 使用空间索引优化距离检查
        if len(points) == 0:
            is_valid = True
        else:
            # 使用向量化距离计算
            existing_points = np.array(points)
            distances = np.linalg.norm(existing_points - new_point, axis=1)
            is_valid = np.all(distances >= cfg.MIN_DIST_BETWEEN)
        
        if is_valid:
            points.append(new_point)
            # 分配优先级（紧急医疗、弹药等优先级高）
            priority = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
            priorities.append(priority)
    
    if len(points) < num_points:
        print(f"警告：只生成了{len(points)}个点，目标是{num_points}个点")
            
    return np.array(points), np.array(priorities)

@performance_monitor
def optimize_clusters_with_threats(demand_points, priorities, threat_model, seed=cfg.RANDOM_SEED):
    """考虑威胁和优先级的聚类优化 - 向量化版本"""
    kmeans = KMeans(n_clusters=cfg.NUM_CLUSTERS, random_state=seed, n_init='auto')
    
    # 使用向量化威胁区域检查
    if threat_model.threat_zones:
        threat_centers = np.array([threat['center'] for threat in threat_model.threat_zones])
        threat_radii = np.array([threat['radius'] * 2 for threat in threat_model.threat_zones])
        threat_levels = [threat['threat_level'] for threat in threat_model.threat_zones]
        
        # 向量化威胁检查
        threat_matrix = threat_zone_check_vectorized(demand_points, threat_centers, threat_radii)
        
        # 创建权重矩阵
        weights = np.ones(len(demand_points))
        threat_multipliers = {'low': 1.2, 'medium': 1.5, 'high': 2.0, 'critical': 3.0}
        
        for i, level in enumerate(threat_levels):
            multiplier = threat_multipliers[level]
            affected_points = threat_matrix[:, i]
            weights[affected_points] *= multiplier
    else:
        weights = np.ones(len(demand_points))
    
    # 结合优先级权重（向量化）
    priority_weights = 6 - priorities  # 优先级越高，权重越小
    weights *= priority_weights
    
    # 使用加权K-means
    sample_weights = 1.0 / weights
    kmeans.fit(demand_points, sample_weight=sample_weights)
    
    # 添加性能指标
    performance_metrics = {
        'num_points': len(demand_points),
        'num_threats': len(threat_model.threat_zones),
        'clustering_quality': kmeans.inertia_
    }
    
    return kmeans, performance_metrics

def generate_dynamic_restricted_airspace(kmeans, demand_points, threat_model):
    """生成动态禁飞区"""
    restricted_areas = []
    
    # 基于聚类生成基础禁飞区
    for i in range(kmeans.n_clusters):
        cluster_points = demand_points[kmeans.labels_ == i]
        if len(cluster_points) == 0: 
            continue
            
        center = kmeans.cluster_centers_[i]
        
        # 基于威胁模型生成禁飞区
        for threat in threat_model.threat_zones:
            if np.linalg.norm(center - threat['center']) < cfg.MAX_DIST_CENTER:
                area = {
                    'center': threat['center'],
                    'radius': threat['radius'],
                    'type': 'dynamic',
                    'threat_level': threat['threat_level']
                }
                restricted_areas.append(area)
                
    return restricted_areas