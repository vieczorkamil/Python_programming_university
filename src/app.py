from fastapi import FastAPI
from src.core.carPlatesDetector import CarPlatesDetector

app = FastAPI(
    title="Advanced programming in Python - FastAPI"
)

detector = None

# app.include_router(ep_2.router)
# app.include_router(ep_1.router)
# app.include_router(ep_3.router)

@app.get("/", tags=["Default"])
async def read_root():
    return {"Hello": "World"}


@app.on_event("startup")
async def startup_event():
    app.detector = CarPlatesDetector('src/models/lapi.weights', 'src/models/darknet-yolov3.cfg')


@app.get("/test")
async def test():
    return {"Response from plate number detector": app.detector.test()}