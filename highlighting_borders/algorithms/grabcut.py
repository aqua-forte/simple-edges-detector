import cv2
import numpy as np

class GrabCutSegmenter:
    """
    Implementation of the GrabCut algorithm for precise foreground extraction.
    """
    def __init__(self):
        pass

    def segment(self, image, rect):
        """
        Extracts the foreground object using GrabCut.

        Args:
            image (np.ndarray): Input image in RGB.
            rect (tuple): Bounding box as (x, y, w, h).

        Returns:
            tuple: (result_image, binary_mask)
        """
        # Create a copy of the image for GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # GrabCut requires the image in BGR format for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # Run GrabCut algorithm
            cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Create a binary mask where 0 and 2 are background, 1 and 3 are foreground
            binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
            
            # Create visualization: highlight the foreground in green
            result_image = image.copy()
            result_image[binary_mask > 0] = [0, 255, 0]
            
            return result_image, binary_mask
        except Exception as e:
            print(f"GrabCut error: {e}")
            # Fallback: return empty mask and original image
            return image, np.zeros(image.shape[:2], dtype=np.uint8)
