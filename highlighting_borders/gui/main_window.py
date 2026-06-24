import cv2
import numpy as np
import sys
import os

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFileDialog,
                             QComboBox, QGroupBox, QMessageBox, QCheckBox,
                             QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gui.canvas import ImageCanvas
from core.model import EdgeDetectionModel
from core.controller import EdgeDetectionController


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edge Detection and Localization")
        self.setGeometry(100, 100, 1400, 800)

        # MVC Initialization
        self.model = EdgeDetectionModel()
        self.controller = EdgeDetectionController(self.model)
        
        # Connect controller signals to view updates
        self.controller.processing_finished.connect(self.on_processing_finished)

        self.init_ui()

    def init_ui(self):

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)

        self.canvas = ImageCanvas(self)
        main_layout.addWidget(self.canvas, 3)

    @pyqtSlot(np.ndarray, np.ndarray, str)
    def on_processing_finished(self, edges, mask, task_type):
        """Slot to handle results from the worker thread."""
        self.canvas.set_image(edges)
        self.canvas.set_mask(mask)
        
        if task_type == "grabcut":
            QMessageBox.information(self, "GrabCut", "Smart selection completed!")
        
        # Re-enable apply button if it was disabled
        self.apply_btn.setEnabled(True)
        self.apply_btn.setText("Find Edges")


    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # File group
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        file_layout.addWidget(load_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Mode group
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QVBoxLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["View", "Rectangular Area", "Freeform Area", "Mark Boundaries"])
        self.mode_combo.currentTextChanged.connect(self.change_mode)
        mode_layout.addWidget(self.mode_combo)

        # Region processing mode
        region_mode_label = QLabel("Region processing mode:")
        region_mode_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        mode_layout.addWidget(region_mode_label)

        self.region_button_group = QButtonGroup()

        self.include_radio = QRadioButton("Process inside area")
        self.include_radio.setChecked(True)
        self.include_radio.toggled.connect(lambda: self.set_region_mode("include"))
        self.region_button_group.addButton(self.include_radio)
        mode_layout.addWidget(self.include_radio)

        self.exclude_radio = QRadioButton("Process outside area")
        self.exclude_radio.toggled.connect(lambda: self.set_region_mode("exclude"))
        self.region_button_group.addButton(self.exclude_radio)
        mode_layout.addWidget(self.exclude_radio)

        clear_annotations_btn = QPushButton("Clear Annotations")
        clear_annotations_btn.clicked.connect(self.clear_current_annotations)
        mode_layout.addWidget(clear_annotations_btn)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)


        # Canny parameters group
        canny_group = QGroupBox("Canny Edge Detection Parameters")
        canny_layout = QVBoxLayout()

        self.auto_update_checkbox = QCheckBox("Real-time auto-update")
        self.auto_update_checkbox.setChecked(False)
        self.auto_update_checkbox.stateChanged.connect(self.toggle_auto_update)
        canny_layout.addWidget(self.auto_update_checkbox)

        self.auto_threshold_checkbox = QCheckBox("Auto-calculate thresholds")
        self.auto_threshold_checkbox.setChecked(False)
        self.auto_threshold_checkbox.stateChanged.connect(self.toggle_auto_threshold)
        canny_layout.addWidget(self.auto_threshold_checkbox)

        # Lower threshold
        self.threshold1_label = QLabel(f"Lower threshold: {self.model.threshold1}")
        canny_layout.addWidget(self.threshold1_label)

        self.threshold1_slider = QSlider(Qt.Horizontal)
        self.threshold1_slider.setMinimum(0)
        self.threshold1_slider.setMaximum(200)
        self.threshold1_slider.setValue(self.model.threshold1)
        self.threshold1_slider.valueChanged.connect(self.update_threshold1)
        canny_layout.addWidget(self.threshold1_slider)

        # Upper threshold
        self.threshold2_label = QLabel(f"Upper threshold: {self.model.threshold2}")
        canny_layout.addWidget(self.threshold2_label)

        self.threshold2_slider = QSlider(Qt.Horizontal)
        self.threshold2_slider.setMinimum(0)
        self.threshold2_slider.setMaximum(300)
        self.threshold2_slider.setValue(self.model.threshold2)
        self.threshold2_slider.valueChanged.connect(self.update_threshold2)
        canny_layout.addWidget(self.threshold2_slider)

        # Blur
        self.blur_label = QLabel(f"Blur: {self.model.blur_size}")
        canny_layout.addWidget(self.blur_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(15)
        self.blur_slider.setValue(self.model.blur_size)
        self.blur_slider.setSingleStep(2)
        self.blur_slider.valueChanged.connect(self.update_blur)
        canny_layout.addWidget(self.blur_slider)

        # Closing
        self.closing_label = QLabel(f"Closing: {self.model.closing_size}")
        canny_layout.addWidget(self.closing_label)

        self.closing_slider = QSlider(Qt.Horizontal)
        self.closing_slider.setMinimum(0)
        self.closing_slider.setMaximum(15)
        self.closing_slider.setValue(self.model.closing_size)
        self.closing_slider.valueChanged.connect(self.update_closing)
        canny_layout.addWidget(self.closing_slider)

        # Sigma Color
        self.sigma_color_label = QLabel(f"Sigma Color: {self.model.sigma_color}")
        canny_layout.addWidget(self.sigma_color_label)

        self.sigma_color_slider = QSlider(Qt.Horizontal)
        self.sigma_color_slider.setMinimum(1)
        self.sigma_color_slider.setMaximum(200)
        self.sigma_color_slider.setValue(self.model.sigma_color)
        self.sigma_color_slider.valueChanged.connect(self.update_sigma_color)
        canny_layout.addWidget(self.sigma_color_slider)

        # Sigma Space
        self.sigma_space_label = QLabel(f"Sigma Space: {self.model.sigma_space}")
        canny_layout.addWidget(self.sigma_space_label)

        self.sigma_space_slider = QSlider(Qt.Horizontal)
        self.sigma_space_slider.setMinimum(1)
        self.sigma_space_slider.setMaximum(200)
        self.sigma_space_slider.setValue(self.model.sigma_space)
        self.sigma_space_slider.valueChanged.connect(self.update_sigma_space)
        canny_layout.addWidget(self.sigma_space_slider)

        # Apply button
        self.apply_btn = QPushButton("Find Edges")
        self.apply_btn.clicked.connect(self.apply_edge_detection)
        canny_layout.addWidget(self.apply_btn)

        canny_group.setLayout(canny_layout)
        layout.addWidget(canny_group)

        # Actions group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()

        preview_btn = QPushButton("Preview Mask")
        preview_btn.clicked.connect(self.preview_mask)
        actions_layout.addWidget(preview_btn)

        grabcut_btn = QPushButton("Smart Selection (GrabCut)")
        grabcut_btn.clicked.connect(self.apply_grabcut)
        actions_layout.addWidget(grabcut_btn)

        save_no_bg_btn = QPushButton("Save without background")
        save_no_bg_btn.clicked.connect(self.save_without_background)
        actions_layout.addWidget(save_no_bg_btn)

        save_with_border_btn = QPushButton("Save with border")
        save_with_border_btn.clicked.connect(self.save_with_border)
        actions_layout.addWidget(save_with_border_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self.reset)
        actions_layout.addWidget(reset_btn)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()

        return panel

    def set_region_mode(self, mode):
        """Устанавливает режим обработки области"""
        self.model.region_mode = mode
        self.canvas.region_mode = mode
        self.canvas.update_display()

        if self.model.auto_update and self.model.original_image is not None and (self.model.rect or self.model.freeform_polygons):
            self.apply_edge_detection()

    def toggle_auto_update(self, state):
        """Enables/disables auto-update"""
        self.model.auto_update = (state == Qt.Checked)

        if self.model.auto_update:
            self.apply_btn.setEnabled(False)
            self.apply_btn.setText("Auto-update active")
            if self.model.original_image is not None:
                self.apply_edge_detection()
        else:
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("Find Edges")

    def toggle_auto_threshold(self, state):
        """Toggles automatic threshold calculation"""
        self.model.auto_threshold = (state == Qt.Checked)
        
        if self.model.original_image is not None:
            self.apply_edge_detection()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            if self.controller.load_image(file_path):
                self.canvas.set_image(self.model.current_image)
                self.auto_update_checkbox.setChecked(False)
            else:
                QMessageBox.warning(self, "Error", "Failed to load image!")

    def change_mode(self, mode_text):
        mode_map = {
            "View": "view",
            "Rectangular Area": "rect",
            "Freeform Area": "freeform",
            "Mark Boundaries": "keep",
            "Select": "select"
        }
        self.model.mode = mode_map[mode_text]
        self.canvas.set_mode(self.model.mode)

    def delete_selected_annotation(self):
        """Deletes the currently selected annotation from canvas and model."""
        self.canvas.delete_selected()

    def clear_current_annotations(self):
        """Очищает текущие аннотации"""
        if self.model.mode == "rect":
            self.model.rect = None
            self.canvas.start_point = None
            self.canvas.end_point = None
        elif self.model.mode == "freeform":
            self.model.freeform_polygons = []
            self.canvas.freeform_polygons = []
            self.canvas.current_polygon = []
        elif self.model.mode == "keep":
            self.model.keep_points = []
            self.model.keep_lines = []
            self.canvas.keep_lines = []
            self.canvas.current_line = []
            self.canvas.drawing_line = False  # Важно сбросить флаг

        self.canvas.update_display()

    def update_threshold1(self, value):
        self.model.threshold1 = value
        self.threshold1_label.setText(f"Lower threshold: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def update_threshold2(self, value):
        self.model.threshold2 = value
        self.threshold2_label.setText(f"Upper threshold: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def update_blur(self, value):
        if value % 2 == 0:
            value += 1
        self.model.blur_size = value
        self.blur_label.setText(f"Blur: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def update_closing(self, value):
        self.model.closing_size = value
        self.closing_label.setText(f"Closing: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def update_sigma_color(self, value):
        self.model.sigma_color = value
        self.sigma_color_label.setText(f"Sigma Color: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def update_sigma_space(self, value):
        self.model.sigma_space = value
        self.sigma_space_label.setText(f"Sigma Space: {value}")

        if self.model.auto_update and self.model.original_image is not None:
            self.apply_edge_detection()

    def apply_edge_detection(self):
        if self.model.original_image is None:
            QMessageBox.warning(self, "Error", "Please load an image!")
            return

        if not self.controller.request_edge_detection():
            QMessageBox.warning(self, "Error", "Edge detection failed to start!")

    def apply_grabcut(self):
        """Performs foreground extraction using the GrabCut algorithm based on the selected rectangle."""
        if self.model.original_image is None:
            QMessageBox.warning(self, "Error", "Please load an image!")
            return

        result = self.controller.request_grabcut()
        if result == "RECT_REQUIRED":
            QMessageBox.warning(self, "Error", "Please select an object with a rectangle first!")
        elif result == "BUSY":
            QMessageBox.warning(self, "Busy", "A processing task is already running. Please wait.")
        elif result is False:
            QMessageBox.warning(self, "Error", "GrabCut failed to start!")


    def preview_mask(self):

        """Shows mask preview"""
        if self.model.mask is None:
            QMessageBox.warning(self, "Error", "Please find edges first!")
            return

        preview = self.controller.generate_mask_preview()
        self.model.current_image = preview
        self.canvas.set_image(self.model.current_image)

        QMessageBox.information(
            self,
            "Preview",
            "Green area - will be kept\nDarkened area - will be transparent"
        )

    def save_without_background(self):
        if self.model.mask is None:
            QMessageBox.warning(self, "Error", "Please find edges first!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png)"
        )

        if file_path:
            result = self.controller.save_without_background(file_path)
            if result is True:
                QMessageBox.information(self, "Success", f"Image saved!\nPath: {file_path}")
            elif result == "EMPTY_MASK":
                QMessageBox.warning(self, "Error", "Mask is empty! Try changing the parameters.")
            else:
                QMessageBox.warning(self, "Error", "Failed to save image!")

    def save_with_border(self):
        if self.model.current_image is None:
            QMessageBox.warning(self, "Error", "No image to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg)"
        )

        if file_path:
            if self.controller.save_with_border(file_path):
                QMessageBox.information(self, "Success", "Image saved!")
            else:
                QMessageBox.warning(self, "Error", "Failed to save image!")

    def reset(self):
        """Полный сброс всех параметров и аннотаций"""
        if self.model.original_image is None:
            return

        if self.controller.reset():
            self.model.current_image = self.model.original_image.copy()
            self.canvas.set_image(self.model.current_image)
            self.canvas.clear_annotations()

            self.auto_update_checkbox.setChecked(False)
            self.auto_threshold_checkbox.setChecked(False)
            self.mode_combo.setCurrentIndex(0)
            self.include_radio.setChecked(True)

            self.threshold1_slider.setValue(self.model.threshold1)
            self.threshold2_slider.setValue(self.model.threshold2)
            self.blur_slider.setValue(self.model.blur_size)
            self.closing_slider.setValue(self.model.closing_size)
            self.sigma_color_slider.setValue(self.model.sigma_color)
            self.sigma_space_slider.setValue(self.model.sigma_space)

            self.threshold1_label.setText(f"Lower threshold: {self.model.threshold1}")
            self.threshold2_label.setText(f"Upper threshold: {self.model.threshold2}")
            self.blur_label.setText(f"Blur: {self.model.blur_size}")
            self.closing_label.setText(f"Closing: {self.model.closing_size}")
            self.sigma_color_label.setText(f"Sigma Color: {self.model.sigma_color}")
            self.sigma_space_label.setText(f"Sigma Space: {self.model.sigma_space}")