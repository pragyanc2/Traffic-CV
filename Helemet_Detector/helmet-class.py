%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from pathlib import Path
import cv2

class classifier:
    
    def __init__(self, usselessint):  
        path = Path('/home/pragyan/cv_project/darknet/object-detection-opencv')
        self.trained = load_trainer(path)

    def predict(self, image):
        img = Image(pil2tensor(image, dtype = np.float32).div_(255))
        return self.trained.predict(img)

