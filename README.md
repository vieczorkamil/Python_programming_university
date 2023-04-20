# PRE ACTION INSTALL
## Model of Convolutional Neural Network (YOLO)
Before usage download Yolo weights (lapi.weights), Yolo config (darknet-yolov3.cfg) and put it into src/models folder. Pre-trained model download [Link](https://www.kaggle.com/achrafkhazri/yolo-weights-for-licence-plate-detector?select=lapi.weights)
## Add a virtual environment
```
python -m venv venv
```
## Activate virtual environment
```
.\venv\Scripts\activate.bat
```
## Install packages
```
pip install -r src/requirements.txt
```

# RUN SERVER LOCALLY
```
uvicorn src.app:app --reload
```