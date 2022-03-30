import sys
sys.path.append('../')
from fastapi import FastAPI, UploadFile, File
from fastapi.param_functions import Depends
from fastapi.responses import FileResponse, Response
from typing import List
from ..model.inference import inference
import uvicorn
from starlette.responses import StreamingResponse
import io
from PIL import Image
import numpy as np


app = FastAPI()

orders = []


@app.get("/")
def hello_world():
    return {"hello": "world"}


@app.post("/order", description="주문을 요청합니다")
async def make_order(files: List[UploadFile] = File(...)):
    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        sentence = inference(image)

    return sentence

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8001, reload=True,)