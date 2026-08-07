"""
Основной скрипт пайплайна.
Выполняет синхронизацию видео (DTW), трансформацию координат (Прокруст),
расчет метрик и рендеринг итогового видео с наложением скелетов.
"""

import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.math_utils import pad, unpad, get_affine_transform
from src.metrics import cosine_distance, weight_distance
from myproject_data.video_loader import extract_all_keypoints

def draw_skeleton(frame, kps, limbs, color):
    """
    Отрисовка линий (костей) и точек (суставов) на кадре.
    color: кортеж (B, G, R).
    """
    for i, j in limbs:
        pt1 = (int(kps[i][0]), int(kps[i][1]))
        pt2 = (int(kps[j][0]), int(kps[j][1]))
        cv2.line(frame, pt1, pt2, color, 3)
    
    for pt in kps:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)


def evaluate_squat_dtw_swapped(ref_video, my_video, output_video, model, device, limbs):
    """
    Основной алгоритм: 
    1. Извлекает точки из двух видео.
    2. Синхронизирует движения через DTW.
    3. Трансформирует эталон к пользователю (Прокрустов анализ).
    4. Считает метрики и рендерит результат.
    """
    print("Извлечение точек из эталонного видео...")
    ref_kps, ref_conf = extract_all_keypoints(ref_video, model, device)
    
    print("Извлечение точек из пользовательского видео...")
    my_kps, my_conf = extract_all_keypoints(my_video, model, device)
    
    print("Синхронизация временных рядов (DTW)...")
    ref_flat = ref_kps.reshape(ref_kps.shape[0], -1)
    my_flat = my_kps.reshape(my_kps.shape[0], -1)
    
    distance, path = fastdtw(my_flat, ref_flat, dist=euclidean)
    
    frame_mapping = {}
    for my_idx, ref_idx in path:
        if my_idx not in frame_mapping:
            frame_mapping[my_idx] = ref_idx

    cap_my = cv2.VideoCapture(my_video)
    fps = cap_my.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap_my.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_my.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # --- ИСПРАВЛЕНИЕ МАСШТАБА ---
    # Применяем тот же масштаб (720p), который использовался при извлечении точек нейросетью
    if orig_height > 720:
        scale = 720 / orig_height
        render_width = int(orig_width * scale)
        render_height = int(orig_height * scale)
    else:
        render_width = orig_width
        render_height = orig_height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (render_width, render_height))
    
    print("Рендеринг результата...")
    frame_idx = 0
    
    while True:
        ret, frame = cap_my.read()
        if not ret:
            break
            
        # Масштабируем кадр перед отрисовкой до 720p (чтобы совпало с координатами)
        if orig_height > 720:
            frame = cv2.resize(frame, (render_width, render_height))
            
        if frame_idx in frame_mapping and frame_idx < len(my_kps):
            ref_idx = frame_mapping[frame_idx]
            
            my_pose = my_kps[frame_idx]
            my_c = my_conf[frame_idx]
            ref_pose = ref_kps[ref_idx]
            
            A = get_affine_transform(ref_pose, my_pose)
            ref_pose_transformed = unpad(pad(ref_pose) @ A)
            
            cos_sim = cosine_distance(my_pose, ref_pose_transformed)
            w_dist = weight_distance(my_pose, ref_pose_transformed, my_c)
            
            draw_skeleton(frame, ref_pose_transformed, limbs, (0, 255, 0)) # Зеленый - эталон
            draw_skeleton(frame, my_pose, limbs, (0, 0, 255)) # Красный - пользователь
            
            cv2.putText(frame, f"Cos Sim: {cos_sim:.3f}", (40, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, f"Cos Sim: {cos_sim:.3f}", (40, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.putText(frame, f"W-Dist: {w_dist:.1f} px", (40, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(frame, f"W-Dist: {w_dist:.1f} px", (40, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
        out.write(frame)
        frame_idx += 1
        
    cap_my.release()
    out.release()
    print(f"Готово! Результат сохранен в {output_video}")
