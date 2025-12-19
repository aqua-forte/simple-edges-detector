import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog,
                             QComboBox, QGroupBox, QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from gui.canvas import ImageCanvas
from algorithms.canny import CannyEdgeDetector


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Локализация и выделение границ объектов")
        self.setGeometry(100, 100, 1400, 800)

        # Переменные
        self.original_image = None
        self.current_image = None
        self.mask = None
        self.rect = None
        self.keep_points = []
        self.mode = "view"  # view, rect, keep

        # Параметры Canny
        self.threshold1 = 50
        self.threshold2 = 150
        self.blur_size = 5

        # Флаг автообновления
        self.auto_update = False

        self.init_ui()

    def init_ui(self):
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Левая панель управления
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        # Правая панель - холст
        self.canvas = ImageCanvas(self)
        main_layout.addWidget(self.canvas, 3)

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Группа загрузки файла
        file_group = QGroupBox("Файл")
        file_layout = QVBoxLayout()

        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(load_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Группа режимов
        mode_group = QGroupBox("Режим работы")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Просмотр", "Выделить область", "Отметить границы"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Группа параметров Canny
        canny_group = QGroupBox("Параметры Canny Edge Detection")
        canny_layout = QVBoxLayout()

        # Чекбокс автообновления
        self.auto_update_checkbox = QCheckBox("Автообновление в реальном времени")
        self.auto_update_checkbox.setChecked(False)
        self.auto_update_checkbox.stateChanged.connect(self.toggle_auto_update)
        canny_layout.addWidget(self.auto_update_checkbox)

        # Нижний порог
        self.threshold1_label = QLabel(f"Нижний порог: {self.threshold1}")
        canny_layout.addWidget(self.threshold1_label)

        self.threshold1_slider = QSlider(Qt.Horizontal)
        self.threshold1_slider.setMinimum(0)
        self.threshold1_slider.setMaximum(200)
        self.threshold1_slider.setValue(self.threshold1)
        self.threshold1_slider.valueChanged.connect(self.update_threshold1)
        canny_layout.addWidget(self.threshold1_slider)

        # Верхний порог
        self.threshold2_label = QLabel(f"Верхний порог: {self.threshold2}")
        canny_layout.addWidget(self.threshold2_label)

        self.threshold2_slider = QSlider(Qt.Horizontal)
        self.threshold2_slider.setMinimum(0)
        self.threshold2_slider.setMaximum(300)
        self.threshold2_slider.setValue(self.threshold2)
        self.threshold2_slider.valueChanged.connect(self.update_threshold2)
        canny_layout.addWidget(self.threshold2_slider)

        # Размытие
        self.blur_label = QLabel(f"Размытие: {self.blur_size}")
        canny_layout.addWidget(self.blur_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(15)
        self.blur_slider.setValue(self.blur_size)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.valueChanged.connect(self.update_blur)
        canny_layout.addWidget(self.blur_slider)

        # Кнопка применения
        self.apply_btn = QPushButton("Найти границы")
        self.apply_btn.clicked.connect(self.apply_edge_detection)
        canny_layout.addWidget(self.apply_btn)

        canny_group.setLayout(canny_layout)
        layout.addWidget(canny_group)

        # Группа действий
        actions_group = QGroupBox("Действия")
        actions_layout = QVBoxLayout()

        preview_btn = QPushButton("Предпросмотр маски")
        preview_btn.clicked.connect(self.preview_mask)
        actions_layout.addWidget(preview_btn)

        save_no_bg_btn = QPushButton("Сохранить без фона")
        save_no_bg_btn.clicked.connect(self.save_without_background)
        actions_layout.addWidget(save_no_bg_btn)

        save_with_border_btn = QPushButton("Сохранить с границей")
        save_with_border_btn.clicked.connect(self.save_with_border)
        actions_layout.addWidget(save_with_border_btn)

        reset_btn = QPushButton("Сбросить")
        reset_btn.clicked.connect(self.reset)
        actions_layout.addWidget(reset_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()

        return panel

    def toggle_auto_update(self, state):
        """Включает/выключает автообновление"""
        self.auto_update = (state == Qt.Checked)

        if self.auto_update:
            self.apply_btn.setEnabled(False)
            self.apply_btn.setText("Автообновление активно")
            # Если уже было хотя бы одно применение, обновляем сразу
            if self.original_image is not None:
                self.apply_edge_detection()
        else:
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("Найти границы")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Выберите изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.current_image = self.original_image.copy()
            self.canvas.set_image(self.current_image)
            self.rect = None
            self.keep_points = []
            self.mask = None
            # Отключаем автообновление при загрузке нового изображения
            self.auto_update_checkbox.setChecked(False)

    def change_mode(self, mode_text):
        mode_map = {
            "Просмотр": "view",
            "Выделить область": "rect",
            "Отметить границы": "keep"
        }
        self.mode = mode_map[mode_text]
        self.canvas.set_mode(self.mode)

    def update_threshold1(self, value):
        self.threshold1 = value
        self.threshold1_label.setText(f"Нижний порог: {value}")

        # Автообновление при изменении параметра
        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def update_threshold2(self, value):
        self.threshold2 = value
        self.threshold2_label.setText(f"Верхний порог: {value}")

        # Автообновление при изменении параметра
        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def update_blur(self, value):
        if value % 2 == 0:
            value += 1
        self.blur_size = value
        self.blur_label.setText(f"Размытие: {value}")

        # Автообновление при изменении параметра
        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def apply_edge_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите изображение!")
            return

        # Получаем область обработки
        if self.rect:
            x, y, w, h = self.rect
            roi = self.original_image[y:y + h, x:x + w]
        else:
            roi = self.original_image
            x, y = 0, 0

        # Применяем Canny
        detector = CannyEdgeDetector(
            self.threshold1,
            self.threshold2,
            self.blur_size
        )
        edges, mask = detector.detect_edges(roi, self.keep_points, x, y)

        # Создаем результирующее изображение
        result = self.original_image.copy()

        if self.rect:
            result[y:y + h, x:x + w] = edges
        else:
            result = edges

        self.current_image = result
        self.mask = mask
        self.canvas.set_image(self.current_image)
        self.canvas.set_mask(mask)

        # Если это первое применение, включаем чекбокс автообновления
        if not self.auto_update:
            self.auto_update_checkbox.setEnabled(True)

    def preview_mask(self):
        """Показывает предварительный просмотр маски"""
        if self.mask is None:
            QMessageBox.warning(self, "Ошибка", "Сначала найдите границы!")
            return

        # Создаем визуализацию маски
        preview = self.original_image.copy().astype(float)

        # Создаем цветное наложение
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = 255  # Зеленый канал

        # Создаем маску для наложения (0.0 - 1.0)
        alpha = (self.mask / 255.0).astype(float)
        alpha = np.stack([alpha] * 3, axis=-1)

        # Смешиваем: где маска=255 показываем оригинал с зеленым оттенком,
        # где маска=0 затемняем
        preview = np.where(
            self.mask[:, :, np.newaxis] > 0,
            preview * 0.7 + overlay * 0.3,  # Оригинал + зеленый оттенок
            preview * 0.3  # Затемненный фон
        ).astype(np.uint8)

        # Обводим контур красным
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 0, 0), 3)

        self.current_image = preview
        self.canvas.set_image(self.current_image)

        QMessageBox.information(
            self,
            "Предпросмотр",
            "Зеленая область - будет сохранена\nЗатемненная область - будет прозрачной"
        )

    def save_without_background(self):
        if self.mask is None:
            QMessageBox.warning(self, "Ошибка", "Сначала найдите границы!")
            return

        # Проверяем, что маска не пустая
        if not np.any(self.mask):
            QMessageBox.warning(self, "Ошибка", "Маска пуста! Попробуйте изменить параметры.")
            return

        # Создаем изображение с прозрачным фоном
        result = np.zeros((*self.original_image.shape[:2], 4), dtype=np.uint8)

        # Копируем RGB каналы
        result[:, :, :3] = self.original_image

        # Альфа-канал = маска
        result[:, :, 3] = self.mask

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "", "PNG (*.png)"
        )

        if file_path:
            # Конвертируем RGBA в BGRA для OpenCV
            result_bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
            success = cv2.imwrite(file_path, result_bgra)

            if success:
                QMessageBox.information(self, "Успех", f"Изображение сохранено!\nПуть: {file_path}")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось сохранить изображение!")

    def save_with_border(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Ошибка", "Нет изображения для сохранения!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "", "PNG (*.png);;JPEG (*.jpg)"
        )

        if file_path:
            result_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(file_path, result_bgr)
            QMessageBox.information(self, "Успех", "Изображение сохранено!")

    def reset(self):
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.canvas.set_image(self.current_image)
            self.rect = None
            self.keep_points = []
            self.mask = None
            self.canvas.clear_annotations()
            # Отключаем автообновление при сбросе
            self.auto_update_checkbox.setChecked(False)