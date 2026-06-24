import cv2
import numpy as np


class CannyEdgeDetector:
    def __init__(self, threshold1=50, threshold2=150, blur_size=5, closing_size=0, sigma_color=75, sigma_space=75):
        """
        Canny edge detector initialization.

        Args:
            threshold1 (int): Lower threshold for hysteresis.
            threshold2 (int): Upper threshold for hysteresis.
            blur_size (int): Kernel size for Gaussian blur.
            closing_size (int): Kernel size for morphological closing. 0 means disabled.
            sigma_color (int): Filter sigma in the color space.
            sigma_space (int): Filter sigma in the coordinate space.
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        self.closing_size = closing_size
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    @staticmethod
    def calculate_optimal_thresholds(image):
        """
        Calculate optimal Canny thresholds based on the median of the image intensity.

        Args:
            image (np.ndarray): Input image (RGB or grayscale).

        Returns:
            tuple: (lower_threshold, upper_threshold)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return lower, upper

    def detect_edges(self, image, keep_points=None, offset_x=0, offset_y=0, region_mask=None, keep_lines=None):
        """
        Full edge detection incorporating region masks and manual boundary corrections.
        """
        # Convert to grayscale and ensure type is uint8
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Bilateral filter for noise reduction while preserving edges
        blurred = cv2.bilateralFilter(gray, self.blur_size, self.sigma_color, self.sigma_space)

        # Basic Canny
        edges = cv2.Canny(blurred, self.threshold1, self.threshold2)

        # Morphological closing to connect gaps in edges
        if self.closing_size > 0:
            kernel = np.ones((self.closing_size, self.closing_size), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 1. Apply region mask if provided
        if region_mask is not None:
            edges = cv2.bitwise_and(edges, region_mask)

        # 2. Add manual boundary corrections (keep_lines)
        if keep_lines:
            for line in keep_lines:
                if len(line) > 1:
                    pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(edges, [pts], isClosed=False, color=255, thickness=1)

        # 3. Generate the mask
        # We use the processed edges to find the object
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour to define the main object mask
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # 4. Visualization result (green edges on original image)
        result = image.copy().astype(np.uint8)
        result[edges > 0] = [0, 255, 0]

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