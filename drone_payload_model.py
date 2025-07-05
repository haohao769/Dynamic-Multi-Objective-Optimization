# --- FILE: drone_payload_model.py ---

import numpy as np
import config_enhanced as cfg

class DronePayloadEnduranceModel:
    """增强的无人机载荷-续航模型"""
    
    def __init__(self):
        self.empty_weight = cfg.DRONE_EMPTY_WEIGHT_KG
        self.max_payload = cfg.DRONE_MAX_PAYLOAD_KG
        self.battery_capacity = cfg.DRONE_BATTERY_ENERGY_WH
        self.base_power = cfg.DRONE_AVG_POWER_EMPTY_WATT
        
    def calculate_flight_parameters(self, payload_kg, supply_type='general'):
        """计算给定载荷下的飞行参数
        
        Parameters:
        -----------
        payload_kg : float
            载荷重量（kg）
        supply_type : str
            补给类型：'medical'(医疗), 'ammunition'(弹药), 'food'(食物), 'general'(一般)
            
        Returns:
        --------
        dict : 包含速度、续航时间、最大航程等参数
        """
        if payload_kg < 0 or payload_kg > self.max_payload:
            raise ValueError(f"载荷必须在0-{self.max_payload}kg之间")
            
        # 总重量
        total_weight = self.empty_weight + payload_kg
        
        # 功率消耗模型（考虑载荷类型的空气动力学影响）
        aero_factor = {
            'medical': 1.05,      # 医疗物资通常包装紧凑
            'ammunition': 1.15,   # 弹药密度大但体积小
            'food': 1.10,        # 食物包装可能较大
            'general': 1.08
        }.get(supply_type, 1.08)
        
        # 功率 = 基础功率 × (总重/空重)^1.5 × 空气动力学系数
        power_consumption = self.base_power * (total_weight / self.empty_weight) ** 1.5 * aero_factor
        
        # 飞行时间（小时）
        flight_time_h = self.battery_capacity / power_consumption
        
        # 速度调整（载荷越大，速度越慢）
        # 考虑不同物资类型对飞行稳定性的影响
        stability_factor = {
            'medical': 0.95,      # 医疗物资需要更稳定的飞行
            'ammunition': 0.90,   # 弹药需要最稳定的飞行
            'food': 0.98,
            'general': 1.0
        }.get(supply_type, 1.0)
        
        speed_reduction = 1 - (payload_kg / self.max_payload) * 0.3
        adjusted_speed = cfg.DRONE_BASE_SPEED_KMH * speed_reduction * stability_factor
        
        # 最大航程
        max_range_km = adjusted_speed * flight_time_h
        
        # 有效航程（考虑安全余量20%）
        effective_range_km = max_range_km * 0.8
        
        # 爬升性能（载荷影响爬升率）
        climb_rate_ms = 5.0 * (1 - payload_kg / self.max_payload * 0.5)
        
        return {
            'payload_kg': payload_kg,
            'supply_type': supply_type,
            'total_weight_kg': total_weight,
            'power_consumption_w': power_consumption,
            'flight_time_h': flight_time_h,
            'cruise_speed_kmh': adjusted_speed,
            'max_range_km': max_range_km,
            'effective_range_km': effective_range_km,
            'climb_rate_ms': climb_rate_ms,
            'energy_efficiency': max_range_km / (self.battery_capacity / 1000)  # km/kWh
        }
        
    def optimize_payload_distribution(self, demand_points, supply_requirements):
        """优化载荷分配策略
        
        Parameters:
        -----------
        demand_points : array
            需求点坐标
        supply_requirements : list
            每个需求点的补给需求 [{'weight': kg, 'type': str, 'priority': int}, ...]
            
        Returns:
        --------
        list : 优化后的无人机任务分配
        """
        missions = []
        
        # 按优先级排序
        sorted_requirements = sorted(enumerate(supply_requirements), 
                                   key=lambda x: x[1]['priority'], 
                                   reverse=True)
        
        for idx, req in sorted_requirements:
            mission = {
                'demand_point_idx': idx,
                'position': demand_points[idx],
                'payload': req['weight'],
                'supply_type': req['type'],
                'priority': req['priority']
            }
            
            # 计算该任务的飞行参数
            flight_params = self.calculate_flight_parameters(req['weight'], req['type'])
            mission.update(flight_params)
            
            missions.append(mission)
            
        return missions
        
    def calculate_mission_feasibility(self, start_pos, end_pos, payload_kg, 
                                    supply_type='general', wind_speed_ms=0, 
                                    wind_direction_deg=0):
        """评估任务可行性（考虑风速等环境因素）"""
        distance_km = np.linalg.norm(end_pos - start_pos)
        
        # 获取基础飞行参数
        params = self.calculate_flight_parameters(payload_kg, supply_type)
        
        # 风速影响（简化模型）
        # 计算有效地速
        wind_angle_rad = np.deg2rad(wind_direction_deg)
        flight_direction = (end_pos - start_pos) / distance_km
        
        # 风速在飞行方向上的分量
        wind_component = wind_speed_ms * 3.6 * np.cos(wind_angle_rad)  # 转换为km/h
        
        # 往返需要考虑顺风和逆风
        speed_to = params['cruise_speed_kmh'] + wind_component
        speed_from = params['cruise_speed_kmh'] - wind_component
        
        # 往返时间
        time_to = distance_km / speed_to if speed_to > 0 else float('inf')
        time_from = distance_km / speed_from if speed_from > 0 else float('inf')
        total_time_h = time_to + time_from
        
        # 任务可行性
        is_feasible = total_time_h <= params['flight_time_h'] * 0.9  # 留10%安全余量
        
        return {
            'is_feasible': is_feasible,
            'distance_km': distance_km,
            'round_trip_time_h': total_time_h,
            'available_flight_time_h': params['flight_time_h'],
            'safety_margin': (params['flight_time_h'] - total_time_h) / params['flight_time_h'] 
                           if is_feasible else -1
        }