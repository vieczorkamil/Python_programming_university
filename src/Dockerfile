FROM python:3.8.0-slim

WORKDIR /src
COPY . .

# RUN pip3 install --upgrade pip
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python", "src/app.py"]
EXPOSE 5000