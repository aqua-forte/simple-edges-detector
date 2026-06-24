import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
from algorithms.canny import CannyEdgeDetector
from algorithms.grabcut import GrabCutSegmenter

class ImageProcessorWorker(QThread):
    """Worker thread for heavy image processing tasks to keep the UI responsive."""
    result_ready = pyqtSignal(np.ndarray, np.ndarray, str)

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "canny":
                image = self.params['image']
                keep_points = self.params['keep_points']
                region_mask = self.params['region_mask']
                keep_lines = self.params['keep_lines']
                
                detector = CannyEdgeDetector(
                    threshold1=self.params['threshold1'],
                    threshold2=self.params['threshold2'],
                    blur_size=self.params['blur_size'],
                    closing_size=self.params['closing_size'],
                    sigma_color=self.params['sigma_color'],
                    sigma_space=self.params['sigma_space']
                )
                edges, mask = detector.detect_edges(
                    image, keep_points, 0, 0, region_mask, keep_lines
                )
                self.result_ready.emit(edges, mask, "canny")
                
            elif self.task_type == "grabcut":
                image = self.params['image']
                rect = self.params['rect']
                segmenter = GrabCutSegmenter()
                edges, mask = segmenter.segment(image, rect)
                self.result_ready.emit(edges, mask, "grabcut")
        except Exception as e:
            print(f"Worker error: {e}")

class EdgeDetectionController(QObject):
    """
    Controller class that coordinates the Model and the View.
    It handles the business logic, image processing orchestration, and file I/O.
    """
    processing_finished = pyqtSignal(np.ndarray, np.ndarray, str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.worker = None
        
        # Timer for debouncing real-time updates
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._execute_async_processing)

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            return False
        
        self.model.original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.model.current_image = self.model.original_image.copy()
        self.model.rect = None
        self.model.freeform_polygons = []
        self.model.keep_points = []
        self.model.keep_lines = []
        self.model.mask = None
        self.model.auto_update = False
        return True

    def request_edge_detection(self):
        if self.model.original_image is None:
            return False

        if self.model.auto_update:
            self.update_timer.start(50)
        else:
            self._execute_async_processing()
        return True

    def request_grabcut(self):
        if self.model.original_image is None:
            return False

        if self.model.rect is None:
            return "RECT_REQUIRED"

        if self.worker is not None and self.worker.isRunning():
            return "BUSY"

        params = {
            'image': self.model.original_image,
            'rect': self.model.rect
        }
        
        self.worker = ImageProcessorWorker("grabcut", params)
        self.worker.result_ready.connect(self._on_worker_finished)
        self.worker.start()
        return True

    def _execute_async_processing(self):
        if self.model.original_image is None:
            return

        # Create region mask for processing
        region_mask = None
        if self.model.rect or self.model.freeform_polygons:
            region_mask = np.zeros(self.model.original_image.shape[:2], dtype=np.uint8)
            if self.model.rect:
                x, y, w, h = self.model.rect
                cv2.rectangle(region_mask, (x, y), (x + w, y + h), 255, -1)
            for polygon in self.model.freeform_polygons:
                if len(polygon) > 2:
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(region_mask, [pts], 255)
            if self.model.region_mode == "exclude":
                region_mask = cv2.bitwise_not(region_mask)

        # Prepare parameters
        params = {
            'image': self.model.original_image,
            'keep_points': self.model.keep_points,
            'region_mask': region_mask,
            'keep_lines': self.model.keep_lines,
            'threshold1': self.model.threshold1,
            'threshold2': self.model.threshold2,
            'blur_size': self.model.blur_size,
            'closing_size': self.model.closing_size,
            'sigma_color': self.model.sigma_color,
            'sigma_space': self.model.sigma_space
        }

        if self.model.auto_threshold:
            t1, t2 = CannyEdgeDetector.calculate_optimal_thresholds(self.model.original_image)
            params['threshold1'] = t1
            params['threshold2'] = t2

        if self.worker is not None and self.worker.isRunning():
            return

        self.worker = ImageProcessorWorker("canny", params)
        self.worker.result_ready.connect(self._on_worker_finished)
        self.worker.start()

    def _on_worker_finished(self, edges, mask, task_type):
        self.model.current_image = edges
        self.model.mask = mask
        self.processing_finished.emit(edges, mask, task_type)

    def generate_mask_preview(self):
        if self.model.mask is None:
            return None

        preview = self.model.original_image.copy().astype(float)
        overlay = np.zeros_like(preview)
        overlay[:, :, 1] = 255

        alpha = (self.model.mask / 255.0).astype(float)
        alpha = np.stack([alpha] * 3, axis=-1)

        preview = np.where(
            self.model.mask[:, :, np.newaxis] > 0,
            preview * 0.7 + overlay * 0.3,
            preview * 0.3
        ).astype(np.uint8)

        contours, _ = cv2.findContours(self.model.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (255, 0, 0), 3)
        
        return preview

    def save_without_background(self, file_path):
        if self.model.mask is None:
            return False

        if not np.any(self.model.mask):
            return "EMPTY_MASK"

        result = np.zeros((*self.model.original_image.shape[:2], 4), dtype=np.uint8)
        result[:, :, :3] = self.model.original_image
        result[:, :, 3] = self.model.mask

        result_bgra = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
        success = cv2.imwrite(file_path, result_bgra)
        return success

    def save_with_border(self, file_path):
        if self.model.current_image is None:
            return False

        result_bgr = cv2.cvtColor(self.model.current_image, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(file_path, result_bgr)
        return success

    def reset(self):
        self.update_timer.stop()
        if self.worker is not None:
            # We can't easily disconnect if we don't have the reference to the slot, 
            # but the worker will just finish and emit to a controller that's being reset.
            # It's better to let it finish or terminate it.
            self.worker.terminate()
            self.worker = None
            
        self.model.reset_parameters()
        return True
