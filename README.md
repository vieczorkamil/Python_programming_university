# Install
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
pip install -r requirements.txt
```

# Run server locally 
```
uvicorn src.app:app --reload
```