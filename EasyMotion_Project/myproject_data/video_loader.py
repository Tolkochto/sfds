"""
Модуль для работы с видеоданными.
Содержит логику для парсинга видео, масштабирования кадров 
и извлечения временных рядов (координат ключевых точек).
"""

import cv2
import numpy as np
import logging
from tqdm import tqdm

from myproject_models.pose_model import get_keypoints_from_frame

def extract_all_keypoints(video_path, model, device, conf_threshold=0.8):
    """
    Проходит по всем кадрам видео, масштабирует их (при необходимости)
    и извлекает координаты ключевых точек скелета с помощью модели.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видео по пути {video_path}")
        raise ValueError(f"Ошибка: Не удалось открыть видео по пути {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kps_list = []
    conf_list = []
    
    logging.info(f"Начало обработки видео: {video_path} (Кадров: {total_frames})")
    
    # Оборачиваем цикл в прогресс-бар tqdm
    with tqdm(total=total_frames, desc=f"Извлечение точек", unit="кадр") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Сжимаем слишком большие кадры для экономии памяти
            height, width = frame.shape[:2]
            if height > 720:
                scale = 720 / height
                frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
            kps, conf = get_keypoints_from_frame(frame, model, device, conf_threshold)
            
            # Fallback-логика: Защита от потери человека в кадре
            if kps is None:
                logging.warning(f"Человек не распознан на кадре {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}. Используем fallback-логику.")
                if len(kps_list) > 0:
                    # Берем координаты с предыдущего кадра
                    kps, conf = kps_list[-1], conf_list[-1]
                else:
                    # Если это первый кадр, заполняем нулями
                    kps, conf = np.zeros((17, 2)), np.zeros(17)
                
            kps_list.append(kps)
            conf_list.append(conf)
            pbar.update(1)
            
    cap.release()
    return np.array(kps_list), np.array(conf_list)
