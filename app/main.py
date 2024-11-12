from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import numpy as np
from PIL import Image
import traceback

app = FastAPI()
templates = Jinja2Templates(directory="templates")

TRITON_SERVER_URL = "http://triton-server:8000/v2/models/resnet50/infer"

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

def preprocess_image(image):
    """Preprocesses the input image to prepare it for Triton inference."""
    image = image.resize((224, 224))
    image = np.array(image).astype("float32") / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict", response_class=HTMLResponse)
async def predict_image(request: Request, image: UploadFile = File(...), image_base64: str = Form(None)):
    try:
        image_data = Image.open(image.file)
        input_tensor = preprocess_image(image_data)
        
        payload = {
            "inputs": [
                {
                    "name": "input__0",
                    "shape": input_tensor.shape,
                    "datatype": "FP32",
                    "data": input_tensor.flatten().tolist()
                }
            ],
            "outputs": [
                {
                    "name": "output__0"
                }
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                TRITON_SERVER_URL,
                json=payload,
                timeout=30.0
            )
        
        if response.status_code != 200:
            return templates.TemplateResponse(
                "error.html", {"request": request, "error": response.text}
            )

        result = response.json()
        output_data = result["outputs"][0]["data"]
        probabilities = np.array(output_data)
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        top5_indices = probabilities.argsort()[-5:][::-1]
        
        with open("imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        
        predictions = [
            {"class": categories[i], "probability": round(float(probabilities[i] * 100), 2)}
            for i in top5_indices
        ]
        
        return templates.TemplateResponse("form.html", {"request": request, "predictions": predictions, "image_base64": image_base64})


    except httpx.RequestError as e:
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": f"Network error: {str(e)}"}
        )
    except Exception as e:
        error_message = traceback.format_exc()
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": error_message}
        )
