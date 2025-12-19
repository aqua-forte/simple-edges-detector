from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import numpy as np


class ImageCanvas(QLabel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #2b2b2b; border: 2px solid #444;")
        self.setMinimumSize(800, 600)

        self.image = None
        self.pixmap = None
        self.mode = "view"
        self.mask = None

        # Для рисования прямоугольника
        self.drawing = False
        self.start_point = None
        self.end_point = None

        # Для рисования линий границ (вместо точек)
        self.keep_lines = []  # Список линий [(x1,y1), (x2,y2), ...]
        self.current_line = []  # Текущая рисуемая линия
        self.drawing_line = False

        # Параметры масштабирования
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.setMouseTracking(True)

    def set_image(self, image):
        """Устанавливает изображение (numpy array RGB)"""
        self.image = image
        self.update_display()

    def set_mode(self, mode):
        """Устанавливает режим работы"""
        self.mode = mode
        if mode == "view":
            self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def set_mask(self, mask):
        """Устанавливает маску"""
        self.mask = mask

    def clear_annotations(self):
        """Очищает аннотации"""
        self.start_point = None
        self.end_point = None
        self.keep_lines = []
        self.current_line = []
        self.parent.rect = None
        self.parent.keep_points = []
        self.update_display()

    def update_display(self):
        """Обновляет отображение"""
        if self.image is None:
            return

        h, w = self.image.shape[:2]
        bytes_per_line = 3 * w
        q_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.pixmap = QPixmap.fromImage(q_image)

        # Вычисляем параметры масштабирования
        widget_width = self.width()
        widget_height = self.height()
        pixmap_width = self.pixmap.width()
        pixmap_height = self.pixmap.height()

        # Масштабируем с сохранением пропорций
        scale = min(widget_width / pixmap_width, widget_height / pixmap_height)
        scaled_width = int(pixmap_width * scale)
        scaled_height = int(pixmap_height * scale)

        # Вычисляем смещение для центрирования
        self.offset_x = (widget_width - scaled_width) // 2
        self.offset_y = (widget_height - scaled_height) // 2

        # Коэффициенты масштабирования (от виджета к изображению)
        self.scale_x = pixmap_width / scaled_width
        self.scale_y = pixmap_height / scaled_height

        # Масштабируем pixmap
        scaled_pixmap = self.pixmap.scaled(
            scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Создаем новый pixmap с аннотациями
        result_pixmap = QPixmap(widget_width, widget_height)
        result_pixmap.fill(QColor(43, 43, 43))  # Фон как в стиле

        painter = QPainter(result_pixmap)

        # Рисуем масштабированное изображение по центру
        painter.drawPixmap(self.offset_x, self.offset_y, scaled_pixmap)

        # Рисуем прямоугольник (координаты уже в пространстве виджета)
        if self.start_point and self.end_point:
            pen = QPen(QColor(0, 128, 255), 3)
            painter.setPen(pen)

            # Преобразуем координаты изображения в координаты виджета
            widget_start = self.image_to_widget(self.start_point)
            widget_end = self.image_to_widget(self.end_point)

            rect = QRect(widget_start, widget_end)
            painter.drawRect(rect.normalized())

        # Рисуем завершенные линии границ
        pen = QPen(QColor(255, 0, 0), 3)
        painter.setPen(pen)

        for line in self.keep_lines:
            if len(line) > 1:
                for i in range(len(line) - 1):
                    p1 = self.image_to_widget(QPoint(int(line[i][0]), int(line[i][1])))
                    p2 = self.image_to_widget(QPoint(int(line[i + 1][0]), int(line[i + 1][1])))
                    painter.drawLine(p1, p2)

        # Рисуем текущую рисуемую линию
        if self.current_line and len(self.current_line) > 1:
            pen = QPen(QColor(255, 100, 100), 3)
            painter.setPen(pen)
            for i in range(len(self.current_line) - 1):
                p1 = self.image_to_widget(QPoint(int(self.current_line[i][0]), int(self.current_line[i][1])))
                p2 = self.image_to_widget(QPoint(int(self.current_line[i + 1][0]), int(self.current_line[i + 1][1])))
                painter.drawLine(p1, p2)

        painter.end()

        self.setPixmap(result_pixmap)

    def widget_to_image(self, pos):
        """Преобразует координаты виджета в координаты изображения"""
        if self.pixmap is None or self.image is None:
            return QPoint(0, 0)

        # Вычитаем смещение
        x = pos.x() - self.offset_x
        y = pos.y() - self.offset_y

        # Применяем масштабирование
        x = int(x * self.scale_x)
        y = int(y * self.scale_y)

        # Ограничиваем координаты
        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))

        return QPoint(x, y)

    def image_to_widget(self, pos):
        """Преобразует координаты изображения в координаты виджета"""
        if self.pixmap is None:
            return pos

        # Применяем обратное масштабирование
        x = int(pos.x() / self.scale_x)
        y = int(pos.y() / self.scale_y)

        # Добавляем смещение
        x += self.offset_x
        y += self.offset_y

        return QPoint(x, y)

    def mousePressEvent(self, event):
        if self.image is None:
            return

        # Конвертируем координаты в координаты изображения
        pos = self.widget_to_image(event.pos())

        if self.mode == "rect":
            self.drawing = True
            self.start_point = pos
            self.end_point = pos
        elif self.mode == "keep":
            # Начинаем рисовать новую линию
            self.drawing_line = True
            self.current_line = [(pos.x(), pos.y())]

    def mouseMoveEvent(self, event):
        pos = self.widget_to_image(event.pos())

        if self.drawing and self.mode == "rect":
            self.end_point = pos
            self.update_display()
        elif self.drawing_line and self.mode == "keep":
            # Добавляем точку к текущей линии
            self.current_line.append((pos.x(), pos.y()))

            # Добавляем точку в keep_points родителя для алгоритма
            self.parent.keep_points.append((pos.x(), pos.y()))

            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.mode == "rect":
            self.drawing = False
            self.end_point = self.widget_to_image(event.pos())

            # Сохраняем прямоугольник в родителе
            rect = QRect(self.start_point, self.end_point).normalized()
            self.parent.rect = (rect.x(), rect.y(), rect.width(), rect.height())

            self.update_display()
        elif self.drawing_line and self.mode == "keep":
            # Завершаем рисование линии
            self.drawing_line = False

            if len(self.current_line) > 1:
                # Сохраняем завершенную линию
                self.keep_lines.append(self.current_line.copy())

            self.current_line = []
            self.update_display()

    def resizeEvent(self, event):
        """Обработка изменения размера виджета"""
        super().resizeEvent(event)
        if self.image is not None:
            self.update_display()