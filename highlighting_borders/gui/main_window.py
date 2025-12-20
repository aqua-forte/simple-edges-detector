import cv2
import numpy as np
import sys
import os

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog,
                             QComboBox, QGroupBox, QMessageBox, QCheckBox,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt

# Добавляем корневую директорию проекта в путь
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.canvas import ImageCanvas
from algorithms.canny import CannyEdgeDetector


class MainWindow(QMainWindow):
    DEFAULT_THRESHOLD1 = 50
    DEFAULT_THRESHOLD2 = 150
    DEFAULT_BLUR_SIZE = 5

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Локализация и выделение границ объектов")
        self.setGeometry(100, 100, 1400, 800)

        # Переменные
        self.original_image = None
        self.current_image = None
        self.mask = None
        self.rect = None
        self.freeform_polygons = []  # Список произвольных областей
        self.keep_points = []
        self.mode = "view"  # view, rect, freeform, keep
        self.region_mode = "include"  # include (работать только в области), exclude (исключить область)

        # Параметры Canny
        self.threshold1 = self.DEFAULT_THRESHOLD1
        self.threshold2 = self.DEFAULT_THRESHOLD2
        self.blur_size = self.DEFAULT_BLUR_SIZE

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
        self.mode_combo.addItems(["Просмотр", "Прямоугольная область", "Произвольная область", "Отметить границы"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)

        # Режим обработки области
        region_mode_label = QLabel("Режим обработки области:")
        region_mode_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        mode_layout.addWidget(region_mode_label)

        self.region_button_group = QButtonGroup()

        self.include_radio = QRadioButton("Обработка внутри области")
        self.include_radio.setChecked(True)
        self.include_radio.toggled.connect(lambda: self.set_region_mode("include"))
        self.region_button_group.addButton(self.include_radio)
        mode_layout.addWidget(self.include_radio)

        self.exclude_radio = QRadioButton("Обработка вне области")
        self.exclude_radio.toggled.connect(lambda: self.set_region_mode("exclude"))
        self.region_button_group.addButton(self.exclude_radio)
        mode_layout.addWidget(self.exclude_radio)

        # Кнопка очистки аннотаций
        clear_annotations_btn = QPushButton("Очистить аннотации")
        clear_annotations_btn.clicked.connect(self.clear_current_annotations)
        mode_layout.addWidget(clear_annotations_btn)

        # Информационная метка
        self.info_label = QLabel("Выберите режим работы")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "color: #888; font-size: 10px; padding: 5px; background: #f0f0f0; border-radius: 5px; margin-top: 5px;")
        mode_layout.addWidget(self.info_label)

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

    def set_region_mode(self, mode):
        """Устанавливает режим обработки области"""
        self.region_mode = mode
        self.canvas.region_mode = mode
        self.canvas.update_display()

        # Обновляем цвет существующих аннотаций
        if self.auto_update and self.original_image is not None and (self.rect or self.freeform_polygons):
            self.apply_edge_detection()

    def toggle_auto_update(self, state):
        """Включает/выключает автообновление"""
        self.auto_update = (state == Qt.Checked)

        if self.auto_update:
            self.apply_btn.setEnabled(False)
            self.apply_btn.setText("Автообновление активно")
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
            self.freeform_polygons = []
            self.keep_points = []
            self.mask = None
            self.auto_update_checkbox.setChecked(False)

    def change_mode(self, mode_text):
        mode_map = {
            "Просмотр": "view",
            "Прямоугольная область": "rect",
            "Произвольная область": "freeform",
            "Отметить границы": "keep"
        }
        self.mode = mode_map[mode_text]
        self.canvas.set_mode(self.mode)

        # Обновляем информационную метку
        info_texts = {
            "view": "Режим просмотра",
            "rect": "Нарисуйте прямоугольник,\nзажав ЛКМ",
            "freeform": "Кликайте для создания точек.\nДвойной клик - завершить область",
            "keep": "Зажмите ЛКМ и рисуйте\nлинию вдоль границы объекта"
        }
        self.info_label.setText(info_texts.get(self.mode, ""))

    def clear_current_annotations(self):
        """Очищает текущие аннотации"""
        if self.mode == "rect":
            self.rect = None
            self.canvas.start_point = None
            self.canvas.end_point = None
        elif self.mode == "freeform":
            self.freeform_polygons = []
            self.canvas.freeform_polygons = []
            self.canvas.current_polygon = []
        elif self.mode == "keep":
            self.keep_points = []
            self.canvas.keep_lines = []
            self.canvas.current_line = []
            self.canvas.drawing_line = False  # Важно сбросить флаг

        self.canvas.update_display()

    def update_threshold1(self, value):
        self.threshold1 = value
        self.threshold1_label.setText(f"Нижний порог: {value}")

        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def update_threshold2(self, value):
        self.threshold2 = value
        self.threshold2_label.setText(f"Верхний порог: {value}")

        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def update_blur(self, value):
        if value % 2 == 0:
            value += 1
        self.blur_size = value
        self.blur_label.setText(f"Размытие: {value}")

        if self.auto_update and self.original_image is not None:
            self.apply_edge_detection()

    def apply_edge_detection(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите изображение!")
            return

        # Создаем маску области для обработки
        region_mask = None

        if self.rect or self.freeform_polygons:
            region_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)

            # Добавляем прямоугольник
            if self.rect:
                x, y, w, h = self.rect
                cv2.rectangle(region_mask, (x, y), (x + w, y + h), 255, -1)

            # Добавляем произвольные области
            for polygon in self.freeform_polygons:
                if len(polygon) > 2:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(region_mask, [pts], 255)

            # Инвертируем маску если режим "исключить"
            if self.region_mode == "exclude":
                region_mask = cv2.bitwise_not(region_mask)

        # Применяем Canny с ОТДЕЛЬНЫМИ ЛИНИЯМИ
        detector = CannyEdgeDetector(
            self.threshold1,
            self.threshold2,
            self.blur_size
        )

        # Передаем линии как список отдельных линий
        edges, mask = detector.detect_edges(
            self.original_image,
            self.keep_points,  # Это все точки из всех линий
            0,
            0,
            region_mask,
            self.canvas.keep_lines  # Добавляем отдельные линии
        )

        self.current_image = edges
        self.mask = mask
        self.canvas.set_image(self.current_image)
        self.canvas.set_mask(mask)

        if not self.auto_update:
            self.auto_update_checkbox.setEnabled(True)

    def preview_mask(self):
        """Показывает предварительный просмотр маски"""
        if self.mask is None:
            QMessageBox.warning(self, "Ошибка", "Сначала найдите границы!")
            return

        preview = self.original_image.copy().astype(float)
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = 255

        alpha = (self.mask / 255.0).astype(float)
        alpha = np.stack([alpha] * 3, axis=-1)

        preview = np.where(
            self.mask[:, :, np.newaxis] > 0,
            preview * 0.7 + overlay * 0.3,
            preview * 0.3
        ).astype(np.uint8)

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

        if not np.any(self.mask):
            QMessageBox.warning(self, "Ошибка", "Маска пуста! Попробуйте изменить параметры.")
            return

        result = np.zeros((*self.original_image.shape[:2], 4), dtype=np.uint8)
        result[:, :, :3] = self.original_image
        result[:, :, 3] = self.mask

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить изображение", "", "PNG (*.png)"
        )

        if file_path:
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
        """Полный сброс всех параметров и аннотаций"""
        if self.original_image is None:
            return

        reply = QMessageBox.question(
            self,
            'Подтверждение сброса',
            'Вы уверены, что хотите сбросить все параметры?\n\n'
            'Будут сброшены:\n'
            '• Все аннотации (области и линии)\n'
            '• Параметры Canny (пороги и размытие)\n'
            '• Режим работы\n'
            '• Автообновление',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return

        self.current_image = self.original_image.copy()
        self.canvas.set_image(self.current_image)

        self.rect = None
        self.freeform_polygons = []
        self.keep_points = []
        self.mask = None
        self.canvas.clear_annotations()

        self.auto_update_checkbox.setChecked(False)
        self.mode_combo.setCurrentIndex(0)
        self.include_radio.setChecked(True)

        self.threshold1 = self.DEFAULT_THRESHOLD1
        self.threshold2 = self.DEFAULT_THRESHOLD2
        self.blur_size = self.DEFAULT_BLUR_SIZE

        self.threshold1_slider.setValue(self.threshold1)
        self.threshold2_slider.setValue(self.threshold2)
        self.blur_slider.setValue(self.blur_size)

        self.threshold1_label.setText(f"Нижний порог: {self.threshold1}")
        self.threshold2_label.setText(f"Верхний порог: {self.threshold2}")
        self.blur_label.setText(f"Размытие: {self.blur_size}")

        QMessageBox.information(
            self,
            "Сброс выполнен",
            f"✓ Все параметры успешно сброшены!\n\n"
            f"Нижний порог: {self.DEFAULT_THRESHOLD1}\n"
            f"Верхний порог: {self.DEFAULT_THRESHOLD2}\n"
            f"Размытие: {self.DEFAULT_BLUR_SIZE}"
        )