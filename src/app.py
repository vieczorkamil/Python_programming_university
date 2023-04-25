from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi import Response
from os import environ
import numpy as np
import uvicorn
import cv2
from core.carPlatesDetector import CarPlatesDetector


app = FastAPI(
    title="Advanced programming in Python - FastAPI"
)

detector = None
result = None
img = None


@app.get("/", tags=["Default"])
async def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    app.detector = CarPlatesDetector('./models/lapi.weights', './models/darknet-yolov3.cfg')
    app.result = "Not started yet"
    app.img = None


@app.get("/test")
async def test():
    return {"Response from plate number detector": app.detector.test()}


def backgroud_task() -> None:
    print("start background task")
    app.result = "In progress"
    app.detector.setInputImage(app.img)
    app.result = app.detector.process()
    print("end background task")


@app.post("/uploadPhoto")
async def upload_photo(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    numpy_array = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
    print(type(img))
    app.img = show_image
    background_tasks.add_task(backgroud_task)

    return {"Upload status": "OK"}


@app.get("/readPlateNumber")
async def read_plate_number():
    print(app.result[0])
    if app.result[0] == "Found":
        response = {"Process": "Success - number plate found",
                    "Plate number": app.result[3]}
    elif app.result[0] == "Not found":
        response = {"Process": "Error - number plate not found"}
    elif app.result[0] == "In progress":
        response = {"Process": "In progress"}
    else:
        response = {"Process": app.result}
    return response


@app.get("/showPlate")
async def show_plate():
    if app.result[0] == "Found":
        _, responseImg = cv2.imencode('.png', app.result[2])
        headers = {'Content-Disposition': 'inline; filename="number_plate.png"'}
        response = Response(responseImg.tobytes(), headers=headers, media_type='image/png')
    elif app.result[0] == "Not found":
        response = {"Process": "Error - number plate not found"}
    elif app.result[0] == "In progress":
        response = {"Process": "In progress"}
    else:
        response = {"Process": app.result}
    return response


@app.get("/showImage")
async def show_image():
    if app.result[0] == "Found":
        _, responseImg = cv2.imencode('.png', app.result[1])
        headers = {'Content-Disposition': 'inline; filename="image_with_highlighted_number_plate.png"'}
        response = Response(responseImg.tobytes(), headers=headers, media_type='image/png')
    elif app.result[0] == "Not found":
        response = {"Process": "Error - number plate not found"}
    elif app.result[0] == "In progress":
        response = {"Process": "In progress"}
    else:
        response = {"Process": app.result}
    return response


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=int(environ.get("PORT", 5000)))
