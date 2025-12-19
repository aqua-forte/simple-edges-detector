"""
Вспомогательные функции для обработки изображений
"""

import cv2
import numpy as np


def resize_image(image, max_width=1920, max_height=1080):
    """
    Изменяет размер изображения с сохранением пропорций

    Параметры:
    - image: входное изображение
    - max_width: максимальная ширина
    - max_height: максимальная высота

    Возвращает:
    - resized_image: изображение с измененным размером
    """
    h, w = image.shape[:2]

    # Вычисляем коэффициент масштабирования
    scale = min(max_width / w, max_height / h, 1.0)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def convert_to_grayscale(image):
    """
    Преобразует изображение в оттенки серого

    Параметры:
    - image: входное изображение (RGB или BGR)

    Возвращает:
    - gray: изображение в оттенках серого
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def apply_morphology(image, operation='close', kernel_size=5, iterations=1):
    """
    Применяет морфологические операции к изображению

    Параметры:
    - image: бинарное изображение
    - operation: тип операции ('dilate', 'erode', 'open', 'close')
    - kernel_size: размер ядра
    - iterations: количество итераций

    Возвращает:
    - result: обработанное изображение
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    operations = {
        'dilate': cv2.dilate,
        'erode': cv2.erode,
        'open': lambda img, k, i: cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=i),
        'close': lambda img, k, i: cv2.morphologyEx(img, cv2.MORPH_CLOSE, k, iterations=i)
    }

    if operation in operations:
        return operations[operation](image, kernel, iterations)

    return image


def find_largest_contour(contours):
    """
    Находит самый большой контур по площади

    Параметры:
    - contours: список контуров

    Возвращает:
    - largest_contour: самый большой контур или None
    """
    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def create_mask_from_contours(image_shape, contours, fill_value=255):
    """
    Создает маску из контуров

    Параметры:
    - image_shape: размер изображения (height, width)
    - contours: список контуров
    - fill_value: значение для заполнения (0-255)

    Возвращает:
    - mask: бинарная маска
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, fill_value, -1)
    return mask


def enhance_edges(edges, kernel_size=3):
    """
    Улучшает качество обнаруженных границ

    Параметры:
    - edges: бинарное изображение границ
    - kernel_size: размер ядра для морфологии

    Возвращает:
    - enhanced_edges: улучшенные границы
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Сначала расширяем
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Затем сужаем для восстановления формы
    enhanced = cv2.erode(dilated, kernel, iterations=1)

    return enhanced


def filter_small_contours(contours, min_area=100):
    """
    Фильтрует мелкие контуры по площади

    Параметры:
    - contours: список контуров
    - min_area: минимальная площадь контура

    Возвращает:
    - filtered_contours: отфильтрованный список контуров
    """
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]


def calculate_contour_properties(contour):
    """
    Вычисляет свойства контура

    Параметры:
    - contour: контур

    Возвращает:
    - properties: словарь со свойствами (area, perimeter, centroid)
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    # Вычисляем центроид
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    return {
        'area': area,
        'perimeter': perimeter,
        'centroid': (cx, cy)
    }


def smooth_contour(contour, epsilon_factor=0.01):
    """
    Сглаживает контур используя алгоритм Douglas-Peucker

    Параметры:
    - contour: входной контур
    - epsilon_factor: коэффициент аппроксимации (0.01 = 1% от периметра)

    Возвращает:
    - smoothed_contour: сглаженный контур
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def create_binary_mask(image, threshold=127):
    """
    Создает бинарную маску из изображения в оттенках серого

    Параметры:
    - image: изображение в оттенках серого
    - threshold: порог бинаризации (0-255)

    Возвращает:
    - binary_mask: бинарная маска
    """
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Накладывает цветную маску на изображение

    Параметры:
    - image: исходное изображение (RGB)
    - mask: бинарная маска
    - color: цвет маски в формате RGB
    - alpha: прозрачность (0.0 - 1.0)

    Возвращает:
    - result: изображение с наложенной маской
    """
    result = image.copy()

    # Создаем цветную маску
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Смешиваем
    result = cv2.addWeighted(result, 1.0, colored_mask, alpha, 0)

    return result


def get_bounding_rect(contour):
    """
    Получает ограничивающий прямоугольник для контура

    Параметры:
    - contour: контур

    Возвращает:
    - rect: кортеж (x, y, width, height)
    """
    return cv2.boundingRect(contour)


def extract_roi_from_mask(image, mask):
    """
    Извлекает область интереса из изображения по маске

    Параметры:
    - image: исходное изображение
    - mask: бинарная маска

    Возвращает:
    - roi: область интереса
    - bbox: ограничивающий прямоугольник (x, y, w, h)
    """
    # Находим контуры маски
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    # Находим ограничивающий прямоугольник
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Извлекаем ROI
    roi = image[y:y + h, x:x + w]

    return roi, (x, y, w, h)