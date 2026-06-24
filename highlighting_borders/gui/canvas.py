from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPolygon, QBrush
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
        self.region_mode = "include"  # include или exclude

        # Для рисования прямоугольника
        self.drawing = False
        self.start_point = None
        self.end_point = None

        # Для произвольных областей
        self.freeform_polygons = []  # Завершенные полигоны
        self.current_polygon = []  # Текущий рисуемый полигон

        # Для рисования линий границ
        self.keep_lines = []
        self.current_line = []
        self.drawing_line = False

        # Параметры масштабирования
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.current_zoom = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint(0, 0)

        # Selection state
        self.selected_polygon_idx = None
        self.selected_line_idx = None

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
        self.freeform_polygons = []
        self.current_polygon = []
        self.keep_lines = []
        self.current_line = []
        self.parent.model.rect = None
        self.parent.model.freeform_polygons = []
        self.parent.model.keep_points = []
        self.parent.model.keep_lines = []
        self.update_display()

    def update_display(self):
        """Updates the canvas visualization with zoom and pan."""
        if self.image is None:
            return

        h, w = self.image.shape[:2]
        bytes_per_line = 3 * w
        
        # Create QImage from numpy array
        q_image = QImage(self.image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

        self.pixmap = QPixmap.fromImage(q_image)

        widget_width = self.width()
        widget_height = self.height()
        
        # Base scale to fit the image in the widget
        base_scale = min(widget_width / w, widget_height / h)
        effective_scale = base_scale * self.current_zoom

        scaled_width = int(w * effective_scale)
        scaled_height = int(h * effective_scale)

        # Center the image and apply pan offset
        self.offset_x = (widget_width - scaled_width) // 2 + self.pan_offset.x()
        self.offset_y = (widget_height - scaled_height) // 2 + self.pan_offset.y()

        self.scale_x = 1.0 / effective_scale
        self.scale_y = 1.0 / effective_scale

        # Scale the pixmap for drawing
        scaled_pixmap = self.pixmap.scaled(
            scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        result_pixmap = QPixmap(widget_width, widget_height)
        result_pixmap.fill(QColor(43, 43, 43))

        painter = QPainter(result_pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, scaled_pixmap)

        # Определяем цвет в зависимости от режима
        region_color = QColor(0, 128, 255) if self.region_mode == "include" else QColor(255, 128, 0)
        region_fill_color = QColor(0, 128, 255, 25) if self.region_mode == "include" else QColor(255, 128, 0, 25)

        # Рисуем прямоугольник
        if self.start_point and self.end_point:
            pen = QPen(region_color, 3)
            painter.setPen(pen)
            painter.setBrush(QBrush(region_fill_color))

            widget_start = self.image_to_widget(self.start_point)
            widget_end = self.image_to_widget(self.end_point)

            rect = QRect(widget_start, widget_end)
            painter.drawRect(rect.normalized())

        # Рисуем завершенные произвольные области
        pen = QPen(region_color, 3)
        painter.setPen(pen)
        painter.setBrush(QBrush(region_fill_color))

        for idx, polygon_points in enumerate(self.freeform_polygons):
            if len(polygon_points) > 2:
                q_polygon = QPolygon()
                for pt in polygon_points:
                    widget_pt = self.image_to_widget(QPoint(int(pt[0]), int(pt[1])))
                    q_polygon.append(widget_pt)
                
                if idx == self.selected_polygon_idx:
                    painter.setPen(QPen(Qt.yellow, 4))
                    painter.setBrush(QBrush(QColor(255, 255, 0, 40)))
                else:
                    painter.setPen(pen)
                    painter.setBrush(QBrush(region_fill_color))
                
                painter.drawPolygon(q_polygon)

        # Рисуем текущий рисуемый полигон
        if len(self.current_polygon) > 0:
            pen = QPen(region_color.lighter(120), 3)
            painter.setPen(pen)

            for i in range(len(self.current_polygon) - 1):
                p1 = self.image_to_widget(QPoint(int(self.current_polygon[i][0]), int(self.current_polygon[i][1])))
                p2 = self.image_to_widget(
                    QPoint(int(self.current_polygon[i + 1][0]), int(self.current_polygon[i + 1][1])))
                painter.drawLine(p1, p2)

            # Рисуем точки
            for pt in self.current_polygon:
                widget_pt = self.image_to_widget(QPoint(int(pt[0]), int(pt[1])))
                painter.setBrush(QBrush(region_color))
                painter.drawEllipse(widget_pt, 5, 5)

        # Рисуем завершенные линии границ
        pen = QPen(QColor(255, 0, 0), 3)
        painter.setPen(pen)

        for idx, line in enumerate(self.keep_lines):
            if len(line) > 1:
                if idx == self.selected_line_idx:
                    painter.setPen(QPen(Qt.yellow, 4))
                else:
                    painter.setPen(pen)
                
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

    def delete_selected(self):
        """Removes the selected polygon or line from both the canvas and the model."""
        deleted = False
        
        if self.selected_polygon_idx is not None:
            idx = self.selected_polygon_idx
            if 0 <= idx < len(self.freeform_polygons):
                self.freeform_polygons.pop(idx)
                if 0 <= idx < len(self.parent.model.freeform_polygons):
                    self.parent.model.freeform_polygons.pop(idx)
                deleted = True
            self.selected_polygon_idx = None

        if self.selected_line_idx is not None:
            idx = self.selected_line_idx
            if 0 <= idx < len(self.keep_lines):
                self.keep_lines.pop(idx)
                if 0 <= idx < len(self.parent.model.keep_lines):
                    self.parent.model.keep_lines.pop(idx)
                deleted = True
            self.selected_line_idx = None

        if deleted:
            self.update_display()
            
        return deleted

    def wheelEvent(self, event):
        """Handles zooming with the mouse wheel"""
        if self.image is None:
            return

        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.current_zoom *= zoom_in_factor
        else:
            self.current_zoom *= zoom_out_factor

        # Limit zoom levels
        self.current_zoom = max(0.1, min(self.current_zoom, 10.0))
        
        self.update_display()

    def widget_to_image(self, pos):
        """Преобразует координаты виджета в координаты изображения"""
        if self.pixmap is None or self.image is None:
            return QPoint(0, 0)

        x = pos.x() - self.offset_x
        y = pos.y() - self.offset_y

        x = int(x * self.scale_x)
        y = int(y * self.scale_y)

        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))
        return QPoint(x, y)

    def image_to_widget(self, pos):
        """Преобразует координаты изображения в координаты виджета"""
        if self.pixmap is None:
            return pos

        x = int(pos.x() / self.scale_x)
        y = int(pos.y() / self.scale_y)

        x += self.offset_x
        y += self.offset_y

        return QPoint(x, y)

    def mouseMoveEvent(self, event):
        """Handles mouse movement for panning and drawing."""
        if self.image is None:
            return

        # Handle Pan (Middle Mouse Button)
        if self.is_panning:
            current_pos = event.pos()
            delta = current_pos - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = current_pos
            self.update_display()
            return

        # Handle Drawing (Rect/Freeform/Keep)
        if self.drawing and self.mode == "rect":
            self.end_point = self.widget_to_image(event.pos())
            self.update_display()
        elif self.mode == "freeform" and self.current_polygon:
            # Optional: update last point of current polygon for visual feedback
            pass 
        elif self.drawing_line and self.mode == "keep":
            # Add point to current line as mouse moves
            pos = self.widget_to_image(event.pos())
            self.current_line.append((pos.x(), pos.y()))
            self.update_display()

    def mousePressEvent(self, event):
        if self.image is None:
            return

        # Handle Pan (Middle Mouse Button)
        if event.button() == Qt.MiddleButton:
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        pos = self.widget_to_image(event.pos())

        if self.mode == "select":
            self.selected_polygon_idx = None
            self.selected_line_idx = None

            # 1. Check polygons (point-in-polygon)
            for idx, poly in enumerate(self.freeform_polygons):
                if self._is_point_in_polygon(pos, poly):
                    self.selected_polygon_idx = idx
                    break

            # 2. Check lines (distance to segment)
            if self.selected_polygon_idx is None:
                min_dist = 10 # pixels threshold
                for idx, line in enumerate(self.keep_lines):
                    for i in range(len(line) - 1):
                        d = self._dist_point_to_segment(pos, line[i], line[i+1])
                        if d < min_dist:
                            min_dist = d
                            self.selected_line_idx = idx

            self.update_display()
            return

        if self.mode == "rect":
            self.drawing = True
            self.start_point = pos
            self.end_point = pos
        elif self.mode == "freeform":
            # Добавляем точку к текущему полигону
            self.current_polygon.append((pos.x(), pos.y()))
            self.update_display()
        elif self.mode == "keep":
            # Начинаем НОВУЮ линию
            self.drawing_line = True
            self.current_line = [(pos.x(), pos.y())]

    def _is_point_in_polygon(self, point, polygon):
        """Standard Ray-Casting point-in-polygon algorithm"""
        x, y = point.x(), point.y()
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _dist_point_to_segment(self, p, a, b):
        """Calculate shortest distance from point p to line segment ab"""
        px, py = p.x(), p.y()
        ax, ay = a[0], a[1]
        bx, by = b[0], b[1]

        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return np.hypot(px - ax, py - ay)

        t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
        t = max(0, min(1, t))

        nearest_x = ax + t * dx
        nearest_y = ay + t * dy
        return np.hypot(px - nearest_x, py - nearest_y)

    def widget_to_image(self, pos):
        """Преобразует координаты виджета в координаты изображения"""
        if self.pixmap is None or self.image is None:
            return QPoint(0, 0)

        x = pos.x() - self.offset_x
        y = pos.y() - self.offset_y

        x = int(x * self.scale_x)
        y = int(y * self.scale_y)

        x = max(0, min(x, self.image.shape[1] - 1))
        y = max(0, min(y, self.image.shape[0] - 1))
        return QPoint(x, y)

    def mouseReleaseEvent(self, event):
        # Handle Pan
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor if self.mode == "view" else Qt.CrossCursor)
            return

        if self.drawing and self.mode == "rect":
            self.drawing = False
            self.end_point = self.widget_to_image(event.pos())

            rect = QRect(self.start_point, self.end_point).normalized()
            self.parent.model.rect = (rect.x(), rect.y(), rect.width(), rect.height())

            self.update_display()
        elif self.drawing_line and self.mode == "keep":
            self.drawing_line = False

            if len(self.current_line) > 1:
                # Сохраняем завершенную линию и в холсте, и в модели
                line_copy = self.current_line.copy()
                self.keep_lines.append(line_copy)
                self.parent.model.keep_lines.append(line_copy)

            # Очищаем текущую линию для новой
            self.current_line = []
            self.update_display()

    def mouseDoubleClickEvent(self, event):
        """Обработка двойного клика для завершения полигона"""
        if self.mode == "freeform" and len(self.current_polygon) > 2:
            # Завершаем текущий полигон
            self.freeform_polygons.append(self.current_polygon.copy())
            self.parent.model.freeform_polygons.append(self.current_polygon.copy())
            self.current_polygon = []
            self.update_display()

    def resizeEvent(self, event):
        """Обработка изменения размера виджета"""
        super().resizeEvent(event)
        if self.image is not None:
            self.update_display()