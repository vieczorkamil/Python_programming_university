from src.core.carPlatesDetector import CarPlatesDetector
from fastapi import FastAPI, UploadFile, File
from fastapi import Response
import numpy as np
import cv2


app = FastAPI(
    title="Advanced programming in Python - FastAPI"
)

detector = None
result = None


@app.get("/", tags=["Default"])
async def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    app.detector = CarPlatesDetector('src/models/lapi.weights', 'src/models/darknet-yolov3.cfg')
    app.result = False


@app.get("/test")
async def test():
    return {"Response from plate number detector": app.detector.test()}


@app.post("/uploadPhoto") #TODO: use background FastAPI task ! 
async def upload_photo(file: UploadFile = File(...)):
    contents = await file.read()
    numpy_array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    print(type(img))
    app.detector.setInputImage(img)
    app.result = app.detector.process()

    return {"Upload status": "OK"}


@app.get("/readPlateNumber")
async def read_plate_number():
    if app.result:
        [response, img1, img2, txt] = app.result
        response = {"Is result found": "True",
                    "Plate number": txt[:-1]}
    else:
        response = {"Error": "Number plate not found"}
    
    return response


@app.get("/showPlate")
async def show_plate():
    if app.result:
        [response, img1, img2, txt] = app.result
        _, responseImg = cv2.imencode('.png', img2)
        headers = {'Content-Disposition': 'inline; filename="number_plate.png"'}
        response = Response(responseImg.tobytes() , headers=headers, media_type='image/png')
    else:
        response = {"Error": "Number plate not found"}
    
    return response


@app.get("/showImage")
async def show_image():
    if app.result:
        [response, img1, img2, txt] = app.result
        _, responseImg = cv2.imencode('.png', img1)
        headers = {'Content-Disposition': 'inline; filename="image_with_highlighted_number_plate.png"'}
        response = Response(responseImg.tobytes() , headers=headers, media_type='image/png')
    else:
        response = {"Error": "Number plate not found"}
    
    return response