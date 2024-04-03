from model import Classifier

from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import uvicorn


from typing import Dict
from PIL import Image
import io
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
async def classification(file: UploadFile = File(...)) -> Dict[str, str]:
    # filename = secure_filename(file.filename)
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    
    top_5_predictions = ML_MODELS['classification'](img = image)

    return top_5_predictions

    
    
if __name__ =='__main__':
    uvicorn.run(app, port=7000)