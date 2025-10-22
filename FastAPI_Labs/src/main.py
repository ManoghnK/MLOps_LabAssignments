from fastapi import FastAPI, status, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict import predict_data
from pathlib import Path


app = FastAPI()

# Mount static files (CSS/JS) served from ./static
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float


class IrisResponse(BaseModel):
    response: int


@app.get("/", response_class=FileResponse)
async def ui_index():
    """Serve the single-page UI."""
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        # Note: model expects features in order: sepal_length, sepal_width, petal_length, petal_width
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                     iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
