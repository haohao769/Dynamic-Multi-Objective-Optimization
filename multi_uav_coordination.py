# --- FILE: multi_uav_coordination.py ---

import numpy as np
from collections import defaultdict
import heapq
import config_enhanced as cfg

class MultiUAVCoordinator:
    """多无人机协调与防碰撞系统"""
    
    def __init__(self, safety_distance=0.05, time_resolution=1.0):
        """
        Parameters:
        -----------
        safety_distance : float
            最小安全距离（km）
        time_resolution : float
            时间分辨率（秒）
        """
        self.safety_distance = safety_distance
        self.time_resolution = time_resolution
        self.space_time_reservations = defaultdict(set)  # {time: {(x,y,z)}}
        self.uav_trajectories = {}  # {uav_id: trajectory}
        
    def register_trajectory(self, uav_id, trajectory, start_time=0):
        """注册无人机轨迹
        
        Parameters:
        -----------
        uav_id : str
            无人机ID
        trajectory : list
            轨迹点列表 [(x,y,z,t), ...]
        start_time : float
            起始时间
        """
        self.uav_trajectories[uav_id] = {
            'trajectory': trajectory,
            'start_time': start_time,
            'current_position': trajectory[0][:3] if trajectory else None
        }
        
        # 在时空图中预留空间
        for point in trajectory:
            x, y, z, t = point
            time_slot = int((start_time + t) / self.time_resolution)
            # 预留一个球形空间
            self._reserve_space(time_slot, (x, y, z), uav_id)
            
    def _reserve_space(self, time_slot, position, uav_id):
        """在时空图中预留空间"""
        # 离散化位置
        discrete_pos = self._discretize_position(position)
        
        # 预留周围的空间点
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    reserved_pos = (
                        discrete_pos[0] + dx,
                        discrete_pos[1] + dy,
                        discrete_pos[2] + dz
                    )
                    self.space_time_reservations[time_slot].add((reserved_pos, uav_id))
                    
    def _discretize_position(self, position):
        """将连续位置离散化到网格"""
        grid_size = self.safety_distance
        return tuple(int(p / grid_size) for p in position)
        
    def check_collision(self, uav_id, proposed_trajectory, start_time=0):
        """检查提议的轨迹是否会发生碰撞
        
        Returns:
        --------
        list : 碰撞点列表 [(time, position, conflicting_uav_id), ...]
        """
        collisions = []
        
        for point in proposed_trajectory:
            x, y, z, t = point
            time_slot = int((start_time + t) / self.time_resolution)
            discrete_pos = self._discretize_position((x, y, z))
            
            # 检查该时空点是否已被占用
            for reserved_pos, other_uav_id in self.space_time_reservations.get(time_slot, set()):
                if other_uav_id != uav_id and reserved_pos == discrete_pos:
                    collisions.append((start_time + t, (x, y, z), other_uav_id))
                    
        return collisions
        
    def resolve_conflicts(self, uav_missions):
        """解决多无人机任务冲突
        
        Parameters:
        -----------
        uav_missions : list
            无人机任务列表 [{'uav_id': str, 'path': list, 'priority': int}, ...]
            
        Returns:
        --------
        dict : 调整后的任务 {uav_id: adjusted_mission}
        """
        # 按优先级排序
        sorted_missions = sorted(uav_missions, key=lambda x: x['priority'], reverse=True)
        adjusted_missions = {}
        
        for mission in sorted_missions:
            uav_id = mission['uav_id']
            original_path = mission['path']
            priority = mission['priority']
            
            # 尝试不同的起飞时间
            best_start_time = self._find_conflict_free_slot(
                uav_id, original_path, priority
            )
            
            if best_start_time is not None:
                # 转换路径为时空轨迹
                trajectory = self._path_to_trajectory(original_path, mission.get('speed', 120))
                self.register_trajectory(uav_id, trajectory, best_start_time)
                
                adjusted_missions[uav_id] = {
                    'original_mission': mission,
                    'start_time': best_start_time,
                    'trajectory': trajectory,
                    'conflicts_resolved': True
                }
            else:
                # 无法找到无冲突时间槽，需要重新规划路径
                alternative_path = self._plan_alternative_path(
                    uav_id, original_path, self.space_time_reservations
                )
                
                if alternative_path:
                    trajectory = self._path_to_trajectory(alternative_path, mission.get('speed', 120))
                    self.register_trajectory(uav_id, trajectory, 0)
                    
                    adjusted_missions[uav_id] = {
                        'original_mission': mission,
                        'start_time': 0,
                        'trajectory': trajectory,
                        'alternative_path': True,
                        'conflicts_resolved': True
                    }
                else:
                    # 任务失败
                    adjusted_missions[uav_id] = {
                        'original_mission': mission,
                        'conflicts_resolved': False,
                        'reason': 'No conflict-free path found'
                    }
                    
        return adjusted_missions
        
    def _find_conflict_free_slot(self, uav_id, path, priority, max_delay=600):
        """寻找无冲突的时间槽"""
        for delay in range(0, max_delay, int(cfg.DRONE_DECONFLICTION_DELAY_S)):
            trajectory = self._path_to_trajectory(path, 120)  # 默认速度
            collisions = self.check_collision(uav_id, trajectory, delay)
            
            if not collisions:
                return delay
                
            # 检查是否所有冲突都是与低优先级无人机
            all_lower_priority = True
            for _, _, other_id in collisions:
                other_mission = next((m for m in self.uav_trajectories.values() 
                                    if m.get('uav_id') == other_id), None)
                if other_mission and other_mission.get('priority', 0) >= priority:
                    all_lower_priority = False
                    break
                    
            if all_lower_priority:
                # 可以抢占低优先级无人机的时空槽
                return delay
                
        return None
        
    def _path_to_trajectory(self, path, speed_kmh):
        """将路径转换为时空轨迹"""
        trajectory = []
        current_time = 0
        
        for i in range(len(path)):
            if len(path[i]) == 3:
                x, y, z = path[i]
            else:
                x, y = path[i]
                z = 0.1  # 默认高度
                
            trajectory.append((x, y, z, current_time))
            
            if i < len(path) - 1:
                # 计算到下一点的时间
                if len(path[i+1]) == 3:
                    next_point = path[i+1]
                else:
                    next_point = (*path[i+1], 0.1)
                    
                distance = np.linalg.norm(np.array(next_point) - np.array((x, y, z)))
                time_to_next = (distance / speed_kmh) * 3600  # 转换为秒
                current_time += time_to_next
                
        return trajectory
        
    def _plan_alternative_path(self, uav_id, original_path, reservations):
        """规划替代路径（简化版本）"""
        # 这里应该实现更复杂的路径规划算法
        # 现在只是简单地增加高度偏移
        if len(original_path[0]) == 3:
            # 3D路径
            alternative = []
            height_offset = 0.05  # 50米高度偏移
            
            for point in original_path:
                x, y, z = point
                alternative.append((x, y, z + height_offset))
                
            return alternative
        else:
            # 2D路径，转换为3D
            alternative = []
            base_height = 0.15  # 150米
            
            for point in original_path:
                x, y = point
                alternative.append((x, y, base_height))
                
            return alternative

class DistributedConflictResolution:
    """分布式冲突解决算法（用于实时调整）"""
    
    def __init__(self, communication_range=2.0):
        self.communication_range = communication_range
        self.uav_states = {}
        
    def update_uav_state(self, uav_id, position, velocity, intention):
        """更新无人机状态"""
        self.uav_states[uav_id] = {
            'position': np.array(position),
            'velocity': np.array(velocity),
            'intention': intention,  # 预期路径
            'last_update': time.time() if 'time' in globals() else 0
        }
        
    def detect_potential_conflicts(self, uav_id, time_horizon=30):
        """检测潜在冲突"""
        if uav_id not in self.uav_states:
            return []
            
        my_state = self.uav_states[uav_id]
        conflicts = []
        
        for other_id, other_state in self.uav_states.items():
            if other_id == uav_id:
                continue
                
            # 检查是否在通信范围内
            distance = np.linalg.norm(my_state['position'] - other_state['position'])
            if distance > self.communication_range:
                continue
                
            # 预测未来位置
            for t in np.arange(0, time_horizon, 1):
                my_future_pos = my_state['position'] + my_state['velocity'] * t
                other_future_pos = other_state['position'] + other_state['velocity'] * t
                
                future_distance = np.linalg.norm(my_future_pos - other_future_pos)
                
                if future_distance < 0.05:  # 50米安全距离
                    conflicts.append({
                        'time': t,
                        'other_uav': other_id,
                        'distance': future_distance,
                        'my_position': my_future_pos,
                        'other_position': other_future_pos
                    })
                    
        return conflicts
        
    def negotiate_resolution(self, uav_id, conflicts):
        """协商解决方案"""
        if not conflicts:
            return None
            
        # 简单的右侧避让规则
        resolution = {
            'type': 'velocity_adjustment',
            'adjustments': []
        }
        
        my_state = self.uav_states[uav_id]
        
        for conflict in conflicts:
            other_id = conflict['other_uav']
            other_state = self.uav_states[other_id]
            
            # 计算相对位置
            relative_pos = other_state['position'] - my_state['position']
            
            # 右侧避让
            avoid_direction = np.array([-relative_pos[1], relative_pos[0], 0])
            avoid_direction = avoid_direction / np.linalg.norm(avoid_direction)
            
            # 速度调整
            speed_adjustment = avoid_direction * 5  # 5 m/s 横向速度
            
            resolution['adjustments'].append({
                'time': conflict['time'],
                'velocity_change': speed_adjustment,
                'reason': f'Avoid {other_id}'
            })
            
        return resolution