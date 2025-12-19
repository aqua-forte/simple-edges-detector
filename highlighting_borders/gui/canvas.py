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

        # Точки которые нужно сохранить
        self.keep_points = []

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
        self.keep_points = []
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

        # Рисуем аннотации
        painter = QPainter(self.pixmap)

        # Рисуем прямоугольник
        if self.start_point and self.end_point:
            pen = QPen(QColor(0, 128, 255), 3)
            painter.setPen(pen)
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect.normalized())

        # Рисуем точки keep
        for point in self.keep_points:
            pen = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(point, 5, 5)

        painter.end()

        # Масштабируем под размер виджета
        scaled_pixmap = self.pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)

    def mousePressEvent(self, event):
        if self.image is None:
            return

        # Конвертируем координаты в координаты изображения
        pos = self.map_to_image(event.pos())

        if self.mode == "rect":
            self.drawing = True
            self.start_point = pos
            self.end_point = pos
        elif self.mode == "keep":
            self.keep_points.append(pos)
            self.parent.keep_points.append((pos.x(), pos.y()))
            self.update_display()

    def mouseMoveEvent(self, event):
        if self.drawing and self.mode == "rect":
            self.end_point = self.map_to_image(event.pos())
            self.update_display()

    def mouseReleaseEvent(self, event):
        if self.drawing and self.mode == "rect":
            self.drawing = False
            self.end_point = self.map_to_image(event.pos())

            # Сохраняем прямоугольник в родителе
            rect = QRect(self.start_point, self.end_point).normalized()
            self.parent.rect = (rect.x(), rect.y(), rect.width(), rect.height())

            self.update_display()

    def map_to_image(self, pos):
        """Преобразует координаты виджета в координаты изображения"""
        if self.pixmap is None:
            return pos

        # Размеры отображаемого изображения
        pixmap_rect = self.pixmap.rect()
        widget_rect = self.rect()

        # Вычисляем смещение
        x_offset = (widget_rect.width() - pixmap_rect.width()) // 2
        y_offset = (widget_rect.height() - pixmap_rect.height()) // 2

        # Преобразуем координаты
        x = int((pos.x() - x_offset) * self.image.shape[1] / pixmap_rect.width())
        y = int((pos.y() - y_offset) * self.image.shape[0] / pixmap_rect.height())

        # Ограничиваем координаты
        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))

        return QPoint(x, y)