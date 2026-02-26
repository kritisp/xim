import os

class Config:
    DEBUG = True
    MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8001/embed")
    PORT = int(os.getenv("PORT", 8080))

config = Config()
