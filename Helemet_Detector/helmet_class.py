# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import fastai
from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from pathlib import Path
import cv2

class classifier:
    
    def __init__(self, usselessint):  
        fastai.device = torch.device('cpu')
        path = Path('/home/pragyan/cv_project/darknet/object-detection-opencv')
        self.trained = load_learner(path)
        # self.trained.default.device = torch.device('cpu')

    def predict(self, image):
        img = Image(image)
        return self.trained.predict(img)

