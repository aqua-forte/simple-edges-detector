import cv2
import numpy as np


class CannyEdgeDetector:
    def __init__(self, threshold1=50, threshold2=150, blur_size=5):
        """
        Инициализация детектора границ Canny

        Параметры:
        - threshold1: нижний порог для гистерезиса
        - threshold2: верхний порог для гистерезиса
        - blur_size: размер ядра для Gaussian blur
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1

    def detect_edges(self, image, keep_points=None, offset_x=0, offset_y=0):
        """
        Обнаружение границ на изображении

        Параметры:
        - image: входное изображение (RGB)
        - keep_points: список точек линий которые должны быть на границах
        - offset_x, offset_y: смещение относительно исходного изображения
        """
        # 1. Преобразование в оттенки серого
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 2. Применение Gaussian blur для уменьшения шума
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # 3. Применение алгоритма Canny
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2)

        # 4. Если есть точки keep (линии), усиливаем границы вдоль них
        if keep_points and len(keep_points) > 0:
            # Создаем маску для усиления
            enhance_mask = np.zeros_like(edges)

            # Рисуем линии по всем точкам keep_points
            points = []
            for point in keep_points:
                x, y = int(point[0] - offset_x), int(point[1] - offset_y)
                if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                    points.append((x, y))

            # Рисуем линии между последовательными точками
            for i in range(len(points) - 1):
                cv2.line(enhance_mask, points[i], points[i + 1], 255, 5)

            # Применяем дилатацию для расширения усиленной области
            kernel_enhance = np.ones((7, 7), np.uint8)
            enhance_mask = cv2.dilate(enhance_mask, kernel_enhance, iterations=1)

            # Объединяем с исходными границами
            edges = cv2.bitwise_or(edges, enhance_mask)

        # 5. Морфологические операции для замыкания контуров
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Дополнительное закрытие для лучшего замыкания контуров
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 6. Находим контуры
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. Создаем маску - заполняем контуры
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if contours:
            # Если есть keep_points, находим контуры которые пересекают нарисованные линии
            if keep_points and len(keep_points) > 0:
                selected_contours = []

                for contour in contours:
                    # Фильтруем слишком маленькие контуры
                    if cv2.contourArea(contour) < 100:
                        continue

                    # Проверяем, пересекает ли контур какую-либо точку линии
                    for point in keep_points:
                        x, y = int(point[0] - offset_x), int(point[1] - offset_y)
                        if 0 <= x < edges.shape[1] and 0 <= y < edges.shape[0]:
                            dist = cv2.pointPolygonTest(contour, (x, y), True)
                            if dist >= -10:  # Точка внутри или близко к контуру
                                selected_contours.append(contour)
                                break

                if selected_contours:
                    cv2.drawContours(mask, selected_contours, -1, 255, -1)
                else:
                    # Если не нашли подходящие контуры, берем самый большой
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            else:
                # Если нет keep_points, берем самый большой контур
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 8. Создание изображения с границами (для визуализации)
        result = image.copy()
        result[edges > 0] = [0, 255, 0]  # Зеленые границы

        return result, mask

    def find_contours(self, edges):
        """
        Поиск контуров на изображении границ

        Параметры:
        - edges: бинарное изображение границ

        Возвращает:
        - contours: список контуров
        """
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours