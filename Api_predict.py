import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from fastapi import FastAPI, HTTPException, Query,File
import httpx
import io 

app = FastAPI()
config = Cfg.load_config_from_name('vgg_transformer')
config['cnn']['pretrained']=False
config['device'] = 'cpu'
detector = Predictor(config)

@app.get("/detect_text")

async def detect_text(image_url: str = Query(..., description="URL of the image to analyze")):
   async with httpx.AsyncClient() as client:
            # Download the image from the URL
        response = await client.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download the image")

            # Open the image
        img = Image.open(io.BytesIO(response.content))
        box = (0, 55, 910, 110) # họ và tên
        box1=(910, 50,1310, 110)  #điện thoại
        img_hoten = img.crop(box)
        img_sdt=img.crop(box1)
            # Format the results
        s1 = detector.predict(img_hoten)
        s2 = detector.predict(img_sdt)
        return {"hoten": s1, "dienthoai":s2}

