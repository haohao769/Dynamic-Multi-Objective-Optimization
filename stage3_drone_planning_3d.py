# --- FILE: stage3_drone_planning_3d.py ---

import heapq
import numpy as np
from utilities import is_line_segment_intersecting_cylinder_3d
import config_enhanced as cfg
import time

class TerrainModel:
    """地形模型，提供高程数据"""
    def __init__(self, area_size=cfg.AREA_LIMIT*2, resolution=0.1):
        self.area_size = area_size
        self.resolution = resolution
        # 生成模拟地形（实际应用中应使用DEM数据）
        self._generate_terrain()
        
    def _generate_terrain(self):
        """生成模拟地形高程图"""
        x = np.arange(-self.area_size/2, self.area_size/2, self.resolution)
        y = np.arange(-self.area_size/2, self.area_size/2, self.resolution)
        X, Y = np.meshgrid(x, y)
        
        # 组合多个高斯函数创建山地地形
        Z = np.zeros_like(X)
        
        # 添加几座"山"
        mountains = [
            {'center': [5, 5], 'height': 0.5, 'spread': 3},
            {'center': [-8, 3], 'height': 0.3, 'spread': 2},
            {'center': [2, -7], 'height': 0.4, 'spread': 2.5}
        ]
        
        for mountain in mountains:
            cx, cy = mountain['center']
            h = mountain['height']
            s = mountain['spread']
            Z += h * np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*s**2))
            
        self.elevation_map = Z
        self.x_coords = x
        self.y_coords = y
        
    def get_elevation(self, x, y):
        """获取指定位置的高程"""
        # 找到最近的网格点
        xi = np.argmin(np.abs(self.x_coords - x))
        yi = np.argmin(np.abs(self.y_coords - y))
        
        if 0 <= xi < len(self.x_coords) and 0 <= yi < len(self.y_coords):
            return self.elevation_map[yi, xi]
        return 0.0
        
    def get_safe_altitude(self, x, y, safety_margin=0.1):
        """获取安全飞行高度"""
        terrain_height = self.get_elevation(x, y)
        return terrain_height + safety_margin

class DronePathPlanner3D:
    """3D无人机路径规划器"""
    
    def __init__(self, terrain_model):
        self.terrain = terrain_model
        self.min_altitude = 0.05  # 最小离地高度 50m
        self.max_altitude = 0.5   # 最大飞行高度 500m
        self.vertical_resolution = 0.02  # 垂直分辨率 20m
        self.horizontal_resolution = 0.1  # 水平分辨率 100m
        
    def a_star_3d(self, start, goal, restricted_areas_3d, max_climb_rate=5.0, max_iterations=10000, timeout_s=3.0):
        """3D A*算法 (优化版)
        
        Parameters:
        -----------
        start, goal : tuple (x, y, z)
            起点和终点的3D坐标
        restricted_areas_3d : list
            3D禁飞区列表，每个包含center(x,y,z), radius, height
        max_climb_rate : float
            最大爬升率 (m/s)
        max_iterations : int
            最大迭代次数，防止无限循环
        timeout_s : float
            算法超时时间(秒)
        """
        start_time = time.time()
        start = tuple(start)
        goal = tuple(goal)
        
        # 直接计算起点到终点的距离
        direct_distance = np.linalg.norm(np.array(goal) - np.array(start))
        
        # 如果距离小于阈值，直接使用直线路径
        if direct_distance < 1.0:  # 1公里阈值
            # 检查直线是否穿过禁飞区
            if not self._is_in_restricted_area_3d(start, goal, restricted_areas_3d):
                return [start, goal]
        
        # 优化：计算搜索边界，限制搜索空间
        bounds_min = [
            min(start[0], goal[0]) - 2.0,  # 水平搜索范围扩展2公里
            min(start[1], goal[1]) - 2.0,
            min(start[2], goal[2]) - 0.1   # 垂直搜索范围扩展100米
        ]
        bounds_max = [
            max(start[0], goal[0]) + 2.0,
            max(start[1], goal[1]) + 2.0,
            max(start[2], goal[2]) + 0.1
        ]
        
        # 优先队列：(f_score, count, node)
        count = 0
        frontier = [(0, count, start)]
        heapq.heapify(frontier)
        
        came_from = {start: None}
        g_score = {start: 0}
        iterations = 0
        
        # 优化：减少搜索方向以提高效率
        # 使用6个主方向加12个对角线方向（共18个方向）
        directions = []
        # 6个主方向
        for d in [-1, 0, 1]:
            if d == 0:
                continue
            directions.append((d, 0, 0))
            directions.append((0, d, 0))
            directions.append((0, 0, d))
        
        # 12个对角线方向（每个平面的4个对角线）
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                directions.append((dx, dy, 0))  # xy平面
        for dx in [-1, 1]:
            for dz in [-1, 1]:
                directions.append((dx, 0, dz))  # xz平面
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                directions.append((0, dy, dz))  # yz平面
                
        while frontier and iterations < max_iterations:
            # 检查超时
            if time.time() - start_time > timeout_s:
                print(f"A*算法超时，返回简化路径 (iterations={iterations})")
                # 提供备选方案：生成一个简单的中转路径
                midpoint = (
                    (start[0] + goal[0]) / 2,
                    (start[1] + goal[1]) / 2,
                    max(start[2], goal[2]) + 0.1  # 稍高一些以避开障碍
                )
                return [start, midpoint, goal]
            
            iterations += 1
            _, _, current = heapq.heappop(frontier)
            
            # 到达目标
            if np.linalg.norm(np.array(current) - np.array(goal)) < self.horizontal_resolution:
                path = self._reconstruct_path_3d(came_from, current, goal)
                return path
                
            # 探索邻居
            for dx, dy, dz in directions:
                neighbor = (
                    current[0] + dx * self.horizontal_resolution,
                    current[1] + dy * self.horizontal_resolution,
                    current[2] + dz * self.vertical_resolution
                )
                
                # 检查边界
                if (neighbor[0] < bounds_min[0] or neighbor[0] > bounds_max[0] or
                    neighbor[1] < bounds_min[1] or neighbor[1] > bounds_max[1] or
                    neighbor[2] < bounds_min[2] or neighbor[2] > bounds_max[2]):
                    continue
                
                # 检查高度限制
                terrain_height = self.terrain.get_safe_altitude(neighbor[0], neighbor[1])
                if neighbor[2] < terrain_height or neighbor[2] > self.max_altitude:
                    continue
                    
                # 检查爬升率限制
                horizontal_dist = np.sqrt(dx**2 + dy**2) * self.horizontal_resolution
                if horizontal_dist > 0:
                    climb_angle = np.arctan(dz * self.vertical_resolution / horizontal_dist)
                    if abs(climb_angle) > np.arctan(max_climb_rate / (cfg.DRONE_BASE_SPEED_KMH / 3.6)):
                        continue
                        
                # 检查3D禁飞区
                if self._is_in_restricted_area_3d(current, neighbor, restricted_areas_3d):
                    continue
                    
                # 计算代价
                step_cost = np.linalg.norm(np.array(neighbor) - np.array(current))
                
                # 额外的高度变化代价（鼓励平稳飞行）
                altitude_change_cost = abs(dz) * self.vertical_resolution * 2
                
                new_g = g_score[current] + step_cost + altitude_change_cost
                
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    h = np.linalg.norm(np.array(goal) - np.array(neighbor))
                    f = new_g + h
                    
                    count += 1
                    heapq.heappush(frontier, (f, count, neighbor))
                    came_from[neighbor] = current
        
        # 未找到路径或达到最大迭代次数，创建备选路径
        print(f"A*算法无法找到路径或达到迭代限制 (iterations={iterations})")
        # 创建一个简单的3点路径
        midpoint = (
            (start[0] + goal[0]) / 2,
            (start[1] + goal[1]) / 2,
            max(start[2], goal[2]) + 0.1  # 稍高一些以避开障碍
        )
        return [start, midpoint, goal]
        
    def _is_in_restricted_area_3d(self, p1, p2, restricted_areas):
        """检查路径段是否穿过3D禁飞区"""
        for area in restricted_areas:
            center = area['center']
            radius = area['radius']
            bottom = area.get('bottom', 0)
            top = area.get('top', self.max_altitude)
            
            # 检查高度范围
            if p1[2] > top and p2[2] > top:
                continue
            if p1[2] < bottom and p2[2] < bottom:
                continue
                
            # 检查水平距离（投影到XY平面）
            cylinder_axis_start = np.array([center[0], center[1], bottom])
            cylinder_axis_end = np.array([center[0], center[1], top])
            if is_line_segment_intersecting_cylinder_3d(p1, p2, cylinder_axis_start, cylinder_axis_end, radius):
                return True
                
        return False
        
    def _reconstruct_path_3d(self, came_from, current, goal):
        """重建3D路径"""
        path = [goal]
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        path.reverse()
        
        # 路径平滑（可选）
        smoothed_path = self._smooth_path_3d(path)
        return smoothed_path
        
    def _smooth_path_3d(self, path, iterations=3):
        """3D路径平滑"""
        if len(path) <= 2:
            return path
            
        smoothed = list(path)
        
        for _ in range(iterations):
            new_path = [smoothed[0]]
            
            for i in range(1, len(smoothed) - 1):
                # 加权平均
                prev_point = np.array(smoothed[i-1])
                curr_point = np.array(smoothed[i])
                next_point = np.array(smoothed[i+1])
                
                # 确保不低于地形
                new_point = 0.25 * prev_point + 0.5 * curr_point + 0.25 * next_point
                min_altitude = self.terrain.get_safe_altitude(new_point[0], new_point[1])
                new_point[2] = max(new_point[2], min_altitude)
                
                new_path.append(tuple(new_point))
                
            new_path.append(smoothed[-1])
            smoothed = new_path
            
        return smoothed

def plan_drone_missions_3d(cluster_center, cluster_points, restricted_areas_3d, 
                          terrain_model, payload_kg=2.5):
    """3D无人机任务规划 (优化版)"""
    planner = DronePathPlanner3D(terrain_model)
    missions = []
    current_time_offset = 0
    max_points_to_process = min(len(cluster_points), 10)  # 最多处理10个点
    
    # 优化: 如果点太多，选择优先级高的点
    if len(cluster_points) > max_points_to_process:
        print(f"点数过多 ({len(cluster_points)}), 仅处理前{max_points_to_process}个")
        # 这里简单地选择前N个点，实际应用中应根据优先级选择
        selected_points = cluster_points[:max_points_to_process]
    else:
        selected_points = cluster_points
    
    # 设置起点高度
    start_altitude = terrain_model.get_safe_altitude(cluster_center[0], cluster_center[1])
    start_3d = (cluster_center[0], cluster_center[1], start_altitude + 0.05)
    
    for i, point in enumerate(selected_points):
        # 设置终点高度
        end_altitude = terrain_model.get_safe_altitude(point[0], point[1])
        end_3d = (point[0], point[1], end_altitude + 0.05)
        
        # 规划3D路径 (使用优化后的A*算法，包含超时参数)
        path_3d = planner.a_star_3d(start_3d, end_3d, restricted_areas_3d, 
                                   max_climb_rate=5.0, timeout_s=3.0)
        
        # 计算路径长度
        path_length = 0
        for j in range(len(path_3d) - 1):
            path_length += np.linalg.norm(
                np.array(path_3d[j]) - np.array(path_3d[j+1])
            )
            
        # 往返路径
        round_trip_length = path_length * 2
        
        # 考虑载荷的速度调整
        effective_speed = cfg.DRONE_BASE_SPEED_KMH * (1 - payload_kg / 10.0 * 0.2)
        effective_speed = max(effective_speed, 30.0)  # 确保最低速度
        
        mission_time_s = (round_trip_length / effective_speed) * 3600
        
        missions.append({
            'point_index': i,
            'path_3d': path_3d,
            'path_length_km': round_trip_length,
            'mission_time_s': mission_time_s,
            'departure_delay_s': current_time_offset,
            'altitude_profile': [p[2] for p in path_3d]
        })
        
        current_time_offset += cfg.DRONE_DECONFLICTION_DELAY_S
        
    total_time_h = max(m['departure_delay_s'] + m['mission_time_s'] 
                      for m in missions) / 3600 if missions else 0
                      
    return missions, total_time_h