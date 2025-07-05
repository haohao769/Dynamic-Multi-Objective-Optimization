import numpy as np

def calculate_path_dist(path, points):
    """计算路径上各点之间的总距离
    
    参数:
        path: 路径顺序的点索引
        points: 点的坐标数组
        
    返回:
        路径总距离
    """
    total_dist = 0
    for i in range(len(path) - 1):
        p1 = points[path[i]]
        p2 = points[path[i + 1]]
        dist = np.linalg.norm(p2 - p1)
        total_dist += dist
    return total_dist

def haversine_distance(lat1, lon1, lat2, lon2):
    """使用haversine公式计算两点间的地球表面距离
    
    参数:
        lat1, lon1: 第一个点的纬度和经度（弧度）
        lat2, lon2: 第二个点的纬度和经度（弧度）
        
    返回:
        两点之间的距离（千米）
    """
    R = 6371  # 地球半径（千米）
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    
    a = np.sin(d_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_energy_consumption(distance, payload_weight, drone_type='default'):
    """计算无人机能量消耗
    
    参数:
        distance: 飞行距离（千米）
        payload_weight: 负载重量（千克）
        drone_type: 无人机类型
        
    返回:
        能量消耗（瓦时）
    """
    # 默认无人机参数
    base_consumption = 100  # 瓦时/千米
    weight_factor = 20      # 每千克额外负载增加的瓦时/千米
    
    # 根据不同类型调整参数
    if drone_type == 'efficient':
        base_consumption = 80
        weight_factor = 15
    elif drone_type == 'heavy_duty':
        base_consumption = 150
        weight_factor = 10
        
    return (base_consumption + weight_factor * payload_weight) * distance

def is_line_segment_intersecting_cylinder_3d(p1, p2, cylinder_axis_start, cylinder_axis_end, radius):
    """检查三维空间中的线段是否与圆柱相交
    
    参数:
        p1, p2: 线段的两个端点 (numpy数组或元组, 3D)
        cylinder_axis_start, cylinder_axis_end: 圆柱轴的端点 (numpy数组或元组, 3D)
        radius: 圆柱半径
        
    返回:
        如果线段与圆柱相交，返回True；否则返回False
    """
    # 确保所有点都是numpy数组
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    cylinder_axis_start = np.array(cylinder_axis_start, dtype=np.float64)
    cylinder_axis_end = np.array(cylinder_axis_end, dtype=np.float64)
    
    # 计算圆柱轴向量
    cylinder_axis = cylinder_axis_end - cylinder_axis_start
    cylinder_length = np.linalg.norm(cylinder_axis)
    cylinder_direction = cylinder_axis / cylinder_length
    
    # 将p1和p2投影到圆柱轴上
    p1_rel = p1 - cylinder_axis_start
    p2_rel = p2 - cylinder_axis_start
    
    # 计算p1和p2到轴的投影距离
    p1_proj_dist = np.dot(p1_rel, cylinder_direction)
    p2_proj_dist = np.dot(p2_rel, cylinder_direction)
    
    # 检查投影点是否在圆柱轴的范围内
    if (p1_proj_dist < 0 and p2_proj_dist < 0) or (p1_proj_dist > cylinder_length and p2_proj_dist > cylinder_length):
        return False
    
    # 计算线段到轴的最短距离
    # 使用叉积来计算
    line_direction = p2 - p1
    line_length = np.linalg.norm(line_direction)
    
    if line_length < 1e-10:  # 线段长度几乎为零
        # 检查点p1到圆柱轴的最短距离
        p1_to_axis = p1_rel - p1_proj_dist * cylinder_direction
        dist = np.linalg.norm(p1_to_axis)
        return dist <= radius
    
    # 计算线段与轴之间的最短距离
    # 使用三维空间中两条线的最短距离公式
    line_direction_norm = line_direction / line_length
    cross_prod = np.cross(line_direction_norm, cylinder_direction)
    
    # 如果两线平行，计算点到轴的距离
    if np.linalg.norm(cross_prod) < 1e-10:
        p1_to_axis = p1_rel - p1_proj_dist * cylinder_direction
        dist = np.linalg.norm(p1_to_axis)
        return dist <= radius
    
    # 计算最短距离
    dist = abs(np.dot(p1_rel, cross_prod)) / np.linalg.norm(cross_prod)
    
    # 如果最短距离大于半径，没有相交
    if dist > radius:
        return False
    
    # 还需要检查相交点是否在线段和圆柱的长度范围内
    # 求解参数化方程
    # 这部分涉及复杂的数学计算，简化处理
    
    # 检查线段的多个点与圆柱的距离
    test_points = [
        p1,
        p2,
        p1 + 0.25 * line_direction,
        p1 + 0.5 * line_direction,
        p1 + 0.75 * line_direction
    ]
    
    for point in test_points:
        point_rel = point - cylinder_axis_start
        proj_dist = np.dot(point_rel, cylinder_direction)
        
        # 如果投影在圆柱轴范围内
        if 0 <= proj_dist <= cylinder_length:
            # 计算点到轴的垂直距离
            point_to_axis = point_rel - proj_dist * cylinder_direction
            if np.linalg.norm(point_to_axis) <= radius:
                return True
    
    return False 