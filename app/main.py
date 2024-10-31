from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import httpx

app = FastAPI()

templates = Jinja2Templates(directory="templates")

TRITON_SERVER_URL = "http://triton:8000/v2/models/your_model_name/infer"

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    contents = await image.read()
    async with httpx.AsyncClient() as client:
        response = await client.post(TRITON_SERVER_URL, files={"image": (image.filename, contents)})
    result = response.json()
    description = result.get("description", "No description")

    return {"description": description}