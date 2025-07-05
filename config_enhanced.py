# --- FILE: config_enhanced.py ---

import numpy as np

# ========== 实验控制参数 ==========
# 可扩展性测试
SCALE_TEST_POINTS = [20, 50, 100, 200, 500]  # 扩展测试规模
ROBUSTNESS_TEST_RUNS = 30  # 增加鲁棒性测试次数
COMPARISON_ALGORITHMS = ['proposed', 'greedy', 'nearest_neighbor', 'simple_ga', 'pso']

# 动态环境参数
DYNAMIC_AIRSPACE_ENABLED = True
THREAT_UPDATE_INTERVAL = 300  # 威胁更新间隔（秒）
INTELLIGENCE_UPDATE_PROBABILITY = 0.3  # 情报更新概率

# ========== 仿真全局参数 ==========
RANDOM_SEED = 42
AREA_LIMIT = 20
AREA_HEIGHT_LIMIT = 0.5  # km (500m)
CONTROL_CENTER = np.array([0.0, 0.0])

# ========== 阶段一：聚类参数 ==========
NUM_CLUSTERS = 5
MIN_DIST_CENTER = 2.0
MAX_DIST_CENTER = 15.0
MIN_DIST_BETWEEN = 0.5

# 威胁模型参数
THREAT_LEVELS = ['low', 'medium', 'high', 'critical']
THREAT_LEVEL_MULTIPLIERS = {
    'low': 1.2,
    'medium': 1.5,
    'high': 2.0,
    'critical': 3.0
}

# ========== 阶段二：卡车路径规划参数 ==========
TRUCK_SPEED_KMH = 60.0
TRUCK_FUEL_CONSUMPTION_L_PER_KM = 0.3  # 油耗
TRUCK_MAX_LOAD_KG = 1000  # 最大载重

# 增强的遗传算法参数
GA_POPULATION_SIZE = 100
GA_GENERATIONS = 200
GA_MUTATION_RATE = 0.02
GA_CROSSOVER_RATE = 0.85
GA_TOURNAMENT_SIZE = 7
GA_ELITE_SIZE = 20
TSP_PENALTY = 1e9

# ========== 阶段三：无人机参数 ==========
# 基础性能参数
DRONE_BASE_SPEED_KMH = 80.0
DRONE_EMPTY_WEIGHT_KG = 5.0
DRONE_MAX_PAYLOAD_KG = 5.0

# 电池和能耗参数
DRONE_BATTERY_ENERGY_WH = 200
DRONE_AVG_POWER_EMPTY_WATT = 250
DRONE_POWER_PAYLOAD_FACTOR = 1.5  # 载荷对功率的影响系数

# 飞行限制
DRONE_MAX_ALTITUDE_M = 500
DRONE_MIN_ALTITUDE_M = 50
DRONE_MAX_CLIMB_RATE_MS = 5.0
DRONE_MAX_DESCENT_RATE_MS = 3.0

# 任务调度参数
DRONE_DECONFLICTION_DELAY_S = 30.0
DRONE_MIN_SEPARATION_M = 50  # 最小安全间隔
DRONE_COMMUNICATION_RANGE_KM = 2.0

# 无人机参数
DRONE_MAX_ENDURANCE_MIN = 30.0  # 最大续航时间(分钟)
DRONE_SAFETY_ALTITUDE_M = 50.0  # 安全高度(m)

# ========== 多目标优化权重 ==========
WEIGHT_TIME_EFFICIENCY = 0.4
WEIGHT_ENERGY_EFFICIENCY = 0.2
WEIGHT_SAFETY = 0.3
WEIGHT_MISSION_SUCCESS = 0.1

# ========== 经济成本参数 ==========
COST_TRUCK_PER_KM = 2.0  # 元/公里
COST_DRONE_PER_FLIGHT_HOUR = 50.0  # 元/飞行小时
COST_FUEL_PER_LITER = 7.5  # 元/升
COST_ELECTRICITY_PER_KWH = 0.6  # 元/千瓦时
COST_MAINTENANCE_FACTOR = 0.1  # 维护成本系数

# ========== 军事任务参数 ==========
SUPPLY_TYPES = {
    'medical': {
        'priority_weight': 5,
        'time_critical': True,
        'fragility': 'high',
        'typical_weight_kg': 2.5
    },
    'ammunition': {
        'priority_weight': 4,
        'time_critical': True,
        'fragility': 'low',
        'typical_weight_kg': 4.0
    },
    'food': {
        'priority_weight': 2,
        'time_critical': False,
        'fragility': 'medium',
        'typical_weight_kg': 3.0
    },
    'general': {
        'priority_weight': 1,
        'time_critical': False,
        'fragility': 'low',
        'typical_weight_kg': 2.0
    }
}

# ========== 环境因素 ==========
WEATHER_CONDITIONS = {
    'clear': {'visibility': 10, 'wind_speed': 5, 'precipitation': 0},
    'cloudy': {'visibility': 5, 'wind_speed': 10, 'precipitation': 0},
    'rain': {'visibility': 2, 'wind_speed': 15, 'precipitation': 5},
    'storm': {'visibility': 1, 'wind_speed': 25, 'precipitation': 20}
}

# ========== 3D地形参数 ==========
TERRAIN_RESOLUTION_M = 100  # 地形分辨率
TERRAIN_MAX_HEIGHT_M = 500  # 最大地形高度
TERRAIN_SAFETY_MARGIN_M = 50  # 地形安全余量

# ========== 通信和传感器参数 ==========
COMM_LATENCY_MS = 50  # 通信延迟
COMM_PACKET_LOSS_RATE = 0.01  # 丢包率
SENSOR_RANGE_M = 200  # 传感器范围
SENSOR_UPDATE_RATE_HZ = 10  # 传感器更新频率

# ========== 输出和日志参数 ==========
OUTPUT_DIR = "./results/"
LOG_LEVEL = "INFO"
SAVE_INTERMEDIATE_RESULTS = True
VISUALIZATION_DPI = 300
REPORT_FORMAT = "pdf"  # pdf, html, markdown

# 无人机型号参数
DRONE_MODELS = {
    'small': {
        'max_payload_kg': 2.0,
        'energy_capacity_wh': 150,
        'base_weight_kg': 1.5,
        'max_speed_kmh': 60,
        'power_empty_w': 100,
        'power_per_kg_w': 40
    },
    'medium': {
        'max_payload_kg': 5.0,
        'energy_capacity_wh': 350,
        'base_weight_kg': 3.0,
        'max_speed_kmh': 80,
        'power_empty_w': 200,
        'power_per_kg_w': 30
    },
    'large': {
        'max_payload_kg': 10.0,
        'energy_capacity_wh': 700,
        'base_weight_kg': 6.0,
        'max_speed_kmh': 70,
        'power_empty_w': 400,
        'power_per_kg_w': 25
    }
}

# 无人机载荷类型参数
PAYLOAD_TYPES = {
    'medical': {
        'priority_factor': 1.5,  # 优先级因子
        'time_sensitivity': 1.8,  # 时间敏感度
        'weight_distribution': (0.5, 3.0),  # 重量分布范围(kg)
        'default_model': 'small'
    },
    'ammunition': {
        'priority_factor': 1.3,
        'time_sensitivity': 1.0,
        'weight_distribution': (2.0, 8.0),
        'default_model': 'medium'
    },
    'food': {
        'priority_factor': 0.8,
        'time_sensitivity': 0.5,
        'weight_distribution': (1.0, 5.0),
        'default_model': 'medium'
    },
    'general': {
        'priority_factor': 1.0,
        'time_sensitivity': 0.7,
        'weight_distribution': (0.5, 5.0),
        'default_model': 'medium'
    }
}

# 供应链参数
SUPPLY_CHAIN_PARAMS = {
    'medical': {
        'max_delay_h': 2.0,  # 最大可接受延迟(小时)
        'expiry_factor': 0.8  # 过期因子(时间敏感物品)
    },
    'ammunition': {
        'max_delay_h': 6.0,
        'expiry_factor': 0.0
    },
    'food': {
        'max_delay_h': 12.0,
        'expiry_factor': 0.3
    },
    'general': {
        'max_delay_h': 24.0,
        'expiry_factor': 0.1
    }
}

# 实验参数
EXPERIMENT_RUNS = 30  # 实验运行次数
TEST_SIZES = [20, 50, 100, 200, 500]  # 测试规模
TEST_SCENARIOS = [
    {'name': 'normal', 'threat_density': 0.1, 'threat_mobility': 0.2},
    {'name': 'high_threat', 'threat_density': 0.3, 'threat_mobility': 0.3},
    {'name': 'dynamic', 'threat_density': 0.2, 'threat_mobility': 0.8}
]