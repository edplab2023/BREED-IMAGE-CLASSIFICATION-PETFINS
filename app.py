from fastapi import FastAPI, File, UploadFile


import numpy as np
import uvicorn

from fastai.vision.core import PILImage
from PIL import Image
import io

from typing import List
import logging

from model import Classifier
import matplotlib.pyplot as plt

from datetime import datetime
from io import BytesIO
from glob import glob
import os
from werkzeug.utils import secure_filename

from contextlib import asynccontextmanager
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImageInput(BaseModel):
    fileList : list[UploadFile]


ML_MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ML_MODELS['classification'] = Classifier(model_path = 'best_weights_deploy.pth')
    yield
    ML_MODELS.clear()
    


app = FastAPI(lifespan=lifespan)


@app.get('/health')
def healthcheck():
    return 'OK'


@app.get('/classes')
def list_classes():
    return ML_MODELS['classification'].model.dls.vocab


@app.post('/inference')
async def classification(fileList: List[UploadFile] = File(...)):
    # if 'fileList' not in request:
    #     return 'No files provided in the request', 400

    # files = request.files.getlist('fileList')
    files = fileList

    if not files:
        return 'No files uploaded', 400
    
    
    # 여러장의 이미지로 추론할 시 이미지묶음 폴더 생성 (현재시간 밀리초)
    now = datetime.now().strftime('%Y%m%d%H%M%S_%f')
    inf_image_path = '/tmp/breed-classification/image/dog/'
    inf_folder_path = '/tmp/breed-classification/image/dog/' + now
    
    
    # 파일 저장
    # if len(files) > 1:
    #     # 여러장의 이미지 업로드한 폴더 생성
    #     try:
    #         os.makedir(inf_folder_path, exist_ok = True)
    #     except OSError:
    #         print ('Error: Creating directory.')
    #         return 'Error: Creating directory.'

    #     # 반복문으로 여러파일을 하나씩 저장
    #     for file in files:
    #         filename = secure_filename(file.filename)
            
    #         contents = await file.read()
    #         image = Image.open(io.BytesIO(contents))            
    #         # image.save(inf_folder_path + '/' + filename)

    #     # 새로 생성한 폴더의 이미지 모두 취합
    #     test_image_path_list = glob(inf_folder_path + '/*')
    #     voting_prob = {}
    #     image_count = len(test_image_path_list)

    #     # 여러사진 추론시작
    #     # 새로 생성한 폴더의 이미지 모두 취합
    #     test_image_path_list = glob(inf_folder_path + '/*')
    #     voting_prob = {}
    #     image_count = len(test_image_path_list)
        
    #     # 여러사진 추론시작
    #     for test_image_path in test_image_path_list :
           
    #         temp_prob = ML_MODELS['classification'](img = image)#image_path=test_image_path)
    #         for pk in temp_prob.keys() :
    #             if pk in voting_prob.keys() :
    #                 temp_value = voting_prob[pk]
    #                 voting_prob[pk] = temp_value + temp_prob[pk]
    #             else :
    #                 voting_prob[pk] = temp_prob[pk]
    #     else :
    #         for tk in voting_prob.keys() :
    #             voting_prob[tk] /= image_count
    #     prob = dict(sorted(voting_prob.items(), key=lambda x : x[1], reverse=True)[:5],)
    # else :
    filename = secure_filename(files[0].filename)
    # print(files[0])
    contents = await files[0].read()
    # print(contents)
    image = Image.open(io.BytesIO(contents)).convert("RGB")
        # image.save(inf_image_path + filename)
    # print(image.size)
    
    
    inf_image_path += filename
    # 한장의 사진 추론 시작       
    prob = ML_MODELS['classification'](img = image)#image_path=inf_image_path)

    for tk in prob.keys() :
        prob[tk] = str(round(prob[tk] * 100, 2)) + '%'

    # print(prob)

    return prob

    
    
if __name__ =='__main__':
    uvicorn.run(app, port=7000)

    
# app = Flask(__name__)


# @app.route('/health', methods = ["GET"])
# def healthcheck():
#     return 'OK'


# @app.route('/classes', methods = ['GET'])
# def list_classes():
#     return ML_MODELS['classification'].model.dls.vocab


# @app.route('/inference', methods=['POST'])
# def classification():
#     if 'fileList' not in request.files:
#         return 'No files provided in the request', 400

#     files = request.files.getlist('fileList')

#     if not files:
#         return 'No files uploaded', 400
    
    
#     # 여러장의 이미지로 추론할 시 이미지묶음 폴더 생성 (현재시간 밀리초)
#     now = datetime.now().strftime('%Y%m%d%H%M%S_%f')
#     inf_image_path = '/mnt/a/petfins-ai/image/dog/'
#     inf_folder_path = '/mnt/a/petfins-ai/image/dog/' + now
    
    
#     # 파일 저장
#     if len(files) > 1:
#         # 여러장의 이미지 업로드한 폴더 생성
#         try:
#             if not os.path.exists(inf_folder_path):
#                 os.makedirs(inf_folder_path)
#         except OSError:
#             print ('Error: Creating directory.')
#             return 'Error: Creating directory.'

#         # 반복문으로 여러파일을 하나씩 저장
#         for file in files:
#             filename = secure_filename(file.filename)
#             file.save(inf_folder_path + '/' + filename)

#         # 새로 생성한 폴더의 이미지 모두 취합
#         test_image_path_list = glob(inf_folder_path + '/*')
#         voting_prob = {}
#         image_count = len(test_image_path_list)

#         # 여러사진 추론시작
#         # 새로 생성한 폴더의 이미지 모두 취합
#         test_image_path_list = glob(inf_folder_path + '/*')
#         voting_prob = {}
#         image_count = len(test_image_path_list)

#         # 여러사진 추론시작
#         for test_image_path in test_image_path_list :
           
#             temp_prob = ML_MODELS['classification'](image_path=test_image_path)
#             for pk in temp_prob.keys() :
#                 if pk in voting_prob.keys() :
#                     temp_value = voting_prob[pk]
#                     voting_prob[pk] = temp_value + temp_prob[pk]
#                 else :
#                     voting_prob[pk] = temp_prob[pk]
#         else :
#             for tk in voting_prob.keys() :
#                 voting_prob[tk] /= image_count
#         prob = dict(sorted(voting_prob.items(), key=lambda x : x[1], reverse=True)[:5],)
#     else :
#         filename = secure_filename(files[0].filename)
        
#         files[0].save(inf_image_path + filename)
        
        
    
#         inf_image_path += filename
#         # 한장의 사진 추론 시작       
#         prob = ML_MODELS['classification'](image_path=inf_image_path)

#     for tk in prob.keys() :
#         prob[tk] = str(prob[tk]) + '%'

#     print(prob)

#     return prob

    
    
# if __name__ =='__main__':
#     app.run()
