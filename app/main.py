from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import numpy as np
from PIL import Image
import traceback

app = FastAPI()

templates = Jinja2Templates(directory="templates")

TRITON_SERVER_URL = "http://triton:8001/v2/model_repository/resnet18/infer"

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

def preprocess_image(image):
    """Preprocesses the input image to prepare it for Triton inference."""
    
    image = image.resize((256, 256))
    image = image.crop((16, 16, 240, 240))  
    image = np.array(image).astype("float32") / 255.0  
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = np.transpose(image, (2, 0, 1)) 
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        
        image_data = Image.open(image.file)
        input_tensor = preprocess_image(image_data)

        payload = {
            "inputs": [
                {
                    "name": "input",
                    "shape": input_tensor.shape,
                    "datatype": "FP32",
                    "data": input_tensor.flatten().tolist()
                }
            ]
        }

        # Send the inference request to Triton Server
        async with httpx.AsyncClient() as client:
            response = await client.post(TRITON_SERVER_URL, json=payload)

        if response.status_code != 200:
            return JSONResponse(content={"error": "Triton inference failed."}, status_code=500)

        # Parse the response from Triton
        result = response.json()
        output_data = result["outputs"][0]["data"]

        # Compute probabilities and get the top 5 classes
        probabilities = np.array(output_data)
        probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))
        top5_indices = probabilities.argsort()[-5:][::-1]

        with open("templates/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]

        predictions = {categories[i]: float(probabilities[i]) for i in top5_indices}

        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        error_message = traceback.format_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)
