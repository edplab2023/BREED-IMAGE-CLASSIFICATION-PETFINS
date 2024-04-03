from fastai.vision.all import *
import numpy as np
import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 클래스 인덱스 클래스명으로 변환
with open('breed_classes.txt', 'r') as f:
    classes = f.read().split('\n')
class_dict = {idx: clsname for idx, clsname in enumerate(classes)}




class Classifier:
    
    def __init__(self, model_path):
    
        self.model = load_learner(model_path)
        self.model.to(device)
        print('model using gpu: ', next(self.model.parameters()).is_cuda)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def preprocess(self, imgfname):
        return PILImage.create(imgfname)
    
    def postprocess(self, probs, top_k = 5):
        probabilities = np.array(probs)

        top_5_index = probabilities.argsort(-1)[::-1][:top_k]

        top_5_probabilities = probabilities[top_5_index]
        
        sum_probabilities = sum(top_5_probabilities)
        
        top_5_probabilities = [prob / sum_probabilities for prob in top_5_probabilities]
        
        
        return {class_dict[idx] : prob for idx, prob in zip(top_5_index, top_5_probabilities)}
    
    def __call__(self, img):
        # img = self.preprocess(image_path)
        pred_class, pred_idx, probs = self.predict(img)
        
        top_5_predictions = self.postprocess(probs)
        
        return top_5_predictions
