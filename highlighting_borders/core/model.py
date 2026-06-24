import numpy as np

class EdgeDetectionModel:
    """
    Model class that holds the state of the edge detection application.
    It contains the images, masks, and parameters used for processing.
    """
    def __init__(self):
        # Images and masks
        self.original_image = None
        self.current_image = None
        self.mask = None
        
        # Selection and annotations
        self.rect = None
        self.freeform_polygons = []
        self.keep_points = []
        self.keep_lines = []
        
        # Operation modes
        self.mode = "view"
        self.region_mode = "include"
        
        # Canny parameters
        self.threshold1 = 50
        self.threshold2 = 150
        self.blur_size = 5
        self.closing_size = 0
        self.sigma_color = 75
        self.sigma_space = 75
        
        # UI state
        self.auto_update = False
        self.auto_threshold = False

    def reset_parameters(self):
        self.threshold1 = 50
        self.threshold2 = 150
        self.blur_size = 5
        self.closing_size = 0
        self.sigma_color = 75
        self.sigma_space = 75
        self.auto_update = False
        self.auto_threshold = False
        self.mode = "view"
        self.region_mode = "include"
        self.rect = None
        self.freeform_polygons = []
        self.keep_points = []
        self.mask = None
