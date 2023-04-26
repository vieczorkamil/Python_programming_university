FROM python:3.8.0

WORKDIR /src
COPY . .

# RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN apt-get update  && apt install wget && wget --no-check-certificate -P ./src "https://pjreddie.com/media/files/yolov3.weights"

CMD ["python", "app.py"]
EXPOSE 5000