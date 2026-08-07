"""
Модуль для работы с видеоданными.
Содержит логику для парсинга видео, масштабирования кадров
и извлечения временных рядов (координат ключевых точек).
"""

import cv2
import numpy as np

# Импортируем функцию инференса нейросети из соседнего модуля
from myproject_models.pose_model import get_keypoints_from_frame

def extract_all_keypoints(video_path, model, device, conf_threshold=0.8):
    """
    Проходит по всем кадрам видео, масштабирует их (при необходимости)
    и извлекает координаты ключевых точек скелета с помощью модели.

    Args:
        video_path (str): Путь к видеофайлу (.mp4).
        model (torch.nn.Module): Предобученная модель Keypoint R-CNN.
        device (torch.device): Вычислительное устройство (cpu или cuda).
        conf_threshold (float): Порог уверенности для отсечения шума.

    Returns:
        tuple:
            - np.ndarray: Массив координат формы (N, 17, 2).
            - np.ndarray: Массив уверенности модели формы (N, 17).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Ошибка: Не удалось открыть видео по пути {video_path}")

    kps_list = []
    conf_list = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ОПТИМИЗАЦИЯ: Сжимаем слишком большие кадры (свыше 720p)
        # Это критически важно для экономии оперативной памяти (RAM)
        height, width = frame.shape[:2]
        if height > 720:
            scale = 720 / height
            frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

        # Извлекаем точки (используем функцию из pose_model.py)
        kps, conf = get_keypoints_from_frame(frame, model, device, conf_threshold)

        # Защита от "пустых" кадров, где человек не распознан
        if kps is None and len(kps_list) > 0:
            kps, conf = kps_list[-1], conf_list[-1]
        elif kps is None:
            kps, conf = np.zeros((17, 2)), np.zeros(17)

        kps_list.append(kps)
        conf_list.append(conf)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Извлечено {frame_idx} кадров...")

    cap.release()
    return np.array(kps_list), np.array(conf_list)