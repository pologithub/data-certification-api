from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint


@app.get("/predict")
def predict(acousticness,  # 0.654
        danceability,      # 0.499
        duration_ms,       # 219827
        energy,            # 0.19
        explicit,          # 0
        id,                # 0B6BeEUd6UwFlbsHMQKjob
        instrumentalness,  # 0.00409
        key,               # 7
        liveness,          # 0.0898
        loudness,          # -16.435
        mode,              # 1
        name,              # Back%20in%20the%20Goodle%20Days
        release_date,      # 1971
        speechiness,       # 0.0454
        tempo,             # 149.46
        valence,           # 0.43
        artist             # John%20Hartford
        ):  # 1

    """
        the pipeline expects to be trained with a DataFrame containing
        the following data types in that order
        ```
        acousticness        float64
        danceability        float64
        duration_ms           int64
        energy              float64
        explicit              int64
        id                   object
        instrumentalness    float64
        key                   int64
        liveness            float64
        loudness            float64
        mode                  int64
        name                 object
        release_date         object
        speechiness         float64
        tempo               float64
        valence             float64
        artist               object
        ```
    """

    # step 1 : convert params to dataframe

    X_pred = pd.DataFrame({ 'acousticness'     : [float(acousticness)],
                            'danceability'     : [float(danceability)],
                            'duration_ms'      : [int(duration_ms)],
                            'energy'           : [float(energy)],
                            'explicit'         : [int(explicit)],
                            'id'               : [str(id)],
                            'instrumentalness' : [float(instrumentalness)],
                            'key'              : [int(key)],
                            'liveness'         : [float(liveness)],
                            'loudness'         : [float(loudness)],
                            'mode'             : [int(mode)],
                            'name'             : [str(name)],
                            'release_date'     : [str(release_date)],
                            'speechiness'      : [float(speechiness)],
                            'tempo'            : [float(tempo)],
                            'valence'          : [float(valence)],
                            'artist'           : [str(artist)]
                                    })


    # print(X_pred)
    # print(X_pred.columns)
    # print(X_pred.dtypes)



    # step 2 : load the trained model
    pipeline = joblib.load("model.joblib")
    # print(pipeline)

    # step 3 : make a prediction
    y_pred = pipeline.predict(X_pred)
    # print(type(y_pred))

    # step 4 : return the prediction (extract the prediction value from the ndarray)
    # print(y_pred)
    prediction = y_pred[0]

    return {"artist": artist, "name": name, "popularity": prediction}
