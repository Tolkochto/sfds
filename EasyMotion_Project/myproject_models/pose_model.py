"""
Модуль инициализации и инференса нейросети.
Содержит загрузку предобученной модели Keypoint R-CNN (PyTorch)
и логику предсказания ключевых точек на отдельном кадре.
"""

import gc
import cv2
import torch
import torchvision
import torchvision.transforms as T

# Список 17 ключевых точек формата COCO
KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Коннекции суставов для отрисовки "костей" скелета
LIMBS = [
    [1, 0], [2, 0], [3, 1], [4, 2], [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 13], [13, 15], [12, 14], [14, 16], [5, 6], [11, 12], [5, 11], [6, 12]
]

def load_pose_model(device):
    """
    Загружает предобученную модель Keypoint R-CNN ResNet-50 FPN.

    Args:
        device (torch.device): Вычислительное устройство (cpu или cuda).

    Returns:
        torch.nn.Module: Модель в режиме оценки (eval).
    """
    # Загружаем модель с весами по умолчанию (COCO_V1)
    weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)

    # Переводим в режим инференса и отправляем на GPU/CPU
    model.to(device).eval()
    return model


def get_keypoints_from_frame(frame, model, device, conf_threshold=0.8):
    """
    Прогоняет один кадр через нейросеть, извлекает скелет с наибольшей
    уверенностью и очищает за собой видеопамять (VRAM/RAM).

    Args:
        frame (np.ndarray): Изображение в формате BGR (из OpenCV).
        model (torch.nn.Module): Модель Keypoint R-CNN.
        device (torch.device): Устройство (cpu/cuda).
        conf_threshold (float): Порог уверенности.

    Returns:
        tuple: (Координаты точек, Уверенность) для лучшего человека в кадре.
               Возвращает (None, None), если человек не найден.
    """
    # Конвертация и подготовка тензора
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img_rgb).to(device)

    # Инференс (строго без вычисления градиентов для экономии памяти)
    with torch.no_grad():
        output = model([img_tensor])[0]

    # Если нейросеть никого не нашла
    if len(output['keypoints']) == 0:
        del img_tensor, output
        return None, None

    # Нас интересует только первый человек с максимальным score
    # Отвязываем тензоры от GPU и переносим в numpy
    best_person_kps = output['keypoints'][0, :, :2].detach().cpu().numpy()
    best_person_conf = output['keypoints_scores'][0, :].detach().cpu().numpy()

    # Жесткая очистка мусора из памяти
    del img_tensor, output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return best_person_kps, best_person_conf