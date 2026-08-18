"""
Основной скрипт пайплайна.
Выполняет синхронизацию видео (DTW), трансформацию координат (Прокруст),
расчет метрик и рендеринг итогового видео с наложением скелетов.
Запускается через командную строку с аргументами.
"""

import argparse
import logging
import cv2
import torch
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm

from src.math_utils import pad, unpad, get_affine_transform
from src.metrics import cosine_distance, weight_distance
from myproject_data.video_loader import extract_all_keypoints

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_skeleton(frame, kps, limbs, color):
    """Отрисовка линий (костей) и точек (суставов) на кадре."""
    for i, j in limbs:
        pt1 = (int(kps[i][0]), int(kps[i][1]))
        pt2 = (int(kps[j][0]), int(kps[j][1]))
        cv2.line(frame, pt1, pt2, color, 3)
    
    for pt in kps:
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)


def evaluate_squat_dtw_swapped(ref_video, my_video, output_video, model, device, limbs):
    """Основной алгоритм синхронизации и расчета метрик."""
    
    ref_kps, ref_conf = extract_all_keypoints(ref_video, model, device)
    my_kps, my_conf = extract_all_keypoints(my_video, model, device)
    
    logging.info("Синхронизация временных рядов (DTW)...")
    ref_flat = ref_kps.reshape(ref_kps.shape[0], -1)
    my_flat = my_kps.reshape(my_kps.shape[0], -1)
    
    distance, path = fastdtw(my_flat, ref_flat, dist=euclidean)
    
    frame_mapping = {}
    for my_idx, ref_idx in path:
        if my_idx not in frame_mapping:
            frame_mapping[my_idx] = ref_idx

    cap_my = cv2.VideoCapture(my_video)
    total_frames = int(cap_my.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_my.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap_my.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap_my.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Применяем масштаб (720p), чтобы координаты совпали с кадром
    if orig_height > 720:
        scale = 720 / orig_height
        render_width = int(orig_width * scale)
        render_height = int(orig_height * scale)
    else:
        render_width = orig_width
        render_height = orig_height
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (render_width, render_height))
    
    logging.info("Начало рендеринга итогового видео...")
    frame_idx = 0
    
    with tqdm(total=total_frames, desc="Рендеринг", unit="кадр") as pbar:
        while True:
            ret, frame = cap_my.read()
            if not ret:
                break
                
            if orig_height > 720:
                frame = cv2.resize(frame, (render_width, render_height))
                
            if frame_idx in frame_mapping and frame_idx < len(my_kps):
                ref_idx = frame_mapping[frame_idx]
                
                my_pose = my_kps[frame_idx]
                my_c = my_conf[frame_idx]
                ref_pose = ref_kps[ref_idx]
                
                # Прокрустов анализ
                A = get_affine_transform(ref_pose, my_pose)
                ref_pose_transformed = unpad(pad(ref_pose) @ A)
                
                cos_sim = cosine_distance(my_pose, ref_pose_transformed)
                w_dist = weight_distance(my_pose, ref_pose_transformed, my_c)
                
                draw_skeleton(frame, ref_pose_transformed, limbs, (0, 255, 0)) # Зеленый - эталон
                draw_skeleton(frame, my_pose, limbs, (0, 0, 255)) # Красный - пользователь
                
                cv2.putText(frame, f"Cos Sim: {cos_sim:.3f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                cv2.putText(frame, f"Cos Sim: {cos_sim:.3f}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"W-Dist: {w_dist:.1f} px", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
                cv2.putText(frame, f"W-Dist: {w_dist:.1f} px", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
            out.write(frame)
            frame_idx += 1
            pbar.update(1)
            
    cap_my.release()
    out.release()
    logging.info(f"Готово! Результат сохранен в {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Тренер: Сравнение техники приседаний.")
    parser.add_argument("--ref", type=str, required=True, help="Путь к эталонному видео (референсу)")
    parser.add_argument("--target", type=str, required=True, help="Путь к пользовательскому видео")
    parser.add_argument("--output", type=str, default="results/output.mp4", help="Путь для сохранения результата")
    
    args = parser.parse_args()

    # Инициализация устройства и модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Используемое устройство: {device}")
    
    logging.info("Загрузка модели Keypoint R-CNN...")
    from myproject_models.pose_model import load_pose_model, LIMBS
    
    model = load_pose_model(device)
    logging.info("Модель успешно загружена!")
    
    # Запуск пайплайна
    evaluate_squat_dtw_swapped(args.ref, args.target, args.output, model, device, LIMBS)
