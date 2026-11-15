from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from predict import load_bundle, predict_data

MAX_BATCH_SIZE = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so a missing model fails fast, not on the first request."""
    load_bundle()
    yield


app = FastAPI(
    title="California Housing Price API",
    description="Predicts the median house value of a California census block group "
    "with a Gradient Boosting model trained on the 1990 census.",
    version="1.0.0",
    lifespan=lifespan,
)


class HousingData(BaseModel):
    """
    Pydantic BaseModel describing one California census block group.

    Field order matches the column order the model was trained on.
    """
    median_income: float = Field(
        ..., gt=0, description="Median household income, in tens of thousands of dollars (8.3 = $83,000)"
    )
    house_age: float = Field(..., ge=0, description="Median age of houses in the block group, in years")
    avg_rooms: float = Field(..., gt=0, description="Average number of rooms per household")
    avg_bedrooms: float = Field(..., gt=0, description="Average number of bedrooms per household")
    population: float = Field(..., ge=0, description="Number of people living in the block group")
    avg_occupancy: float = Field(..., gt=0, description="Average number of people per household")
    latitude: float = Field(..., ge=32.5, le=42.0, description="Latitude (must be inside California)")
    longitude: float = Field(..., ge=-124.5, le=-114.0, description="Longitude (must be inside California)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "median_income": 8.3252,
                    "house_age": 41.0,
                    "avg_rooms": 6.98,
                    "avg_bedrooms": 1.02,
                    "population": 322.0,
                    "avg_occupancy": 2.56,
                    "latitude": 37.88,
                    "longitude": -122.23,
                }
            ]
        }
    }

    def to_features(self):
        return [
            self.median_income, self.house_age, self.avg_rooms, self.avg_bedrooms,
            self.population, self.avg_occupancy, self.latitude, self.longitude,
        ]


class HousingResponse(BaseModel):
    median_house_value: float = Field(..., description="Predicted median house value, in $100,000s")
    median_house_value_usd: int = Field(..., description="Predicted median house value, in dollars")


class ModelMetrics(BaseModel):
    r2: float
    mae: float
    rmse: float


class ModelInfo(BaseModel):
    model_type: str
    target: str
    feature_names: list[str]
    metrics: ModelMetrics
    n_train: int
    n_test: int
    n_iterations: int
    sklearn_version: str
    trained_at: str


def to_response(value: float) -> HousingResponse:
    return HousingResponse(
        median_house_value=round(float(value), 4),
        median_house_value_usd=round(float(value) * 100_000),
    )


"""Modern web apps use a technique named routing. This helps the user remember the URLs.
For instance, instead of having /booking.php they see /booking/. Instead of /account.asp?id=1234/
they’d see /account/1234/."""


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    """Health check used by load balancers and uptime monitors."""
    return {"status": "healthy"}


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    """Return the model type, features and held-out test metrics recorded at training time."""
    bundle = load_bundle()
    return ModelInfo(**{k: v for k, v in bundle.items() if k != "model"})


# Prediction is CPU-bound, so these routes use plain `def`: FastAPI runs them in a
# thread pool instead of blocking the event loop the way `async def` would.
@app.post("/predict", response_model=HousingResponse)
def predict_housing(housing_features: HousingData):
    """
    Predict the median house value of one census block group.

    Example:
        POST /predict
        {
            "median_income": 8.3252, "house_age": 41.0, "avg_rooms": 6.98,
            "avg_bedrooms": 1.02, "population": 322.0, "avg_occupancy": 2.56,
            "latitude": 37.88, "longitude": -122.23
        }
        Response:
        {"median_house_value": 4.1463, "median_house_value_usd": 414634}

    Raises:
        HTTPException: 500 with error details if prediction fails.
    """
    try:
        prediction = predict_data([housing_features.to_features()])
        return to_response(prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=list[HousingResponse])
def predict_housing_batch(houses: list[HousingData]):
    """
    Predict median house values for up to 1000 block groups in one request.

    Raises:
        HTTPException: 400 if the list is empty or too long; 500 if prediction fails.
    """
    if not houses:
        raise HTTPException(status_code=400, detail="Send at least one record.")
    if len(houses) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, detail=f"Batch size {len(houses)} exceeds the limit of {MAX_BATCH_SIZE}."
        )
    try:
        predictions = predict_data([h.to_features() for h in houses])
        return [to_response(p) for p in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
