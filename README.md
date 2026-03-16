# CropIQ 🌿
#### A machine learning web application for crop recommendation, fertilizer suggestions, and plant disease detection.

## DISCLAIMER ⚠️
This is a POC (Proof of concept) project. The data used here comes with no guarantee from the creator. Please don't use it for making farming decisions. If you do so, the creator is not responsible for anything. However, this project demonstrates how we can use ML/DL in precision farming when developed at large scale with authentic and verified data.

## MOTIVATION 💪
- Farming is one of the major sectors that influences a country's economic growth.

- In countries like India, the majority of the population depends on agriculture for their livelihood. Many new technologies, such as Machine Learning and Deep Learning, are being implemented into agriculture so that it is easier for farmers to grow and maximize their yield.

- In this project, I present a website in which the following applications are implemented: Crop recommendation, Fertilizer recommendation, and Plant disease prediction.

    - **Crop Recommendation**: Users can provide soil data and the application will predict which crop should be grown.

    - **Fertilizer Recommendation**: Users can input soil data and crop type, and the application will predict what the soil lacks or has excess of and recommend improvements.

    - **Plant Disease Detection**: Users can upload an image of a plant leaf, and the application will predict the disease and provide suggestions for treatment.

## DATA SOURCE 📊
- [Crop recommendation dataset ](https://www.kaggle.com/atharvaingle/crop-recommendation-dataset) (custom built dataset)
- [Fertilizer suggestion dataset](https://github.com/Gladiator07/Harvestify/blob/master/Data-processed/fertilizer.csv) (custom built dataset)
- [Disease detection dataset](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## Notebooks 📓
##### I have also published the corresponding code on Kaggle Notebooks.
- [Crop Recommendation](https://www.kaggle.com/atharvaingle/what-crop-to-grow)
- [Disease Detection](https://www.kaggle.com/atharvaingle/plant-disease-classification-resnet-99-2)

# Built with 🛠️
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/python/python.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/html/html.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/css/css.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/javascript/javascript.png"></code>
<code><img height="30" src="https://github.com/tomchen/stack-icons/raw/master/logos/bootstrap.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
<code><img height="30" src="https://symbols.getvecta.com/stencil_80/56_flask.3a79b5a056.jpg"></code>
<code><img height="30" src="https://cdn.iconscout.com/icon/free/png-256/heroku-225989.png"></code>

<code><img height="30" src="https://raw.githubusercontent.com/numpy/numpy/7e7f4adab814b223f7f917369a72757cd28b10cb/branding/icons/numpylogo.svg"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pandas-dev/pandas/761bceb77d44aa63b71dda43ca46e8fd4b9d7422/web/pandas/static/img/pandas.svg"></code>
<code><img height="30" src="https://matplotlib.org/_static/logo2.svg"></code>
<code><img height="30" src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1280px-Scikit_learn_logo_small.svg.png"></code>
<code><img height="30" src="https://raw.githubusercontent.com/pytorch/pytorch/39fa0b5d0a3b966a50dcd90b26e6c36942705d6d/docs/source/_static/img/pytorch-logo-dark.svg"></code>

## DEPLOYMENT 🚀

#### This project can be deployed using various platforms like Heroku, Railway, or Render.
#### For deployment, use the `app/` directory which contains the Flask application.

## How to use 💻
- Crop Recommendation system ==> enter the corresponding nutrient values of your soil, state and city. Note that, the N-P-K (Nitrogen-Phosphorous-Pottasium) values to be entered should be the ratio between them. Refer [this website](https://www.gardeningknowhow.com/garden-how-to/soil-fertilizers/fertilizer-numbers-npk.htm) for more information.
Note: When you enter the city name, make sure to enter mostly common city names. Remote cities/towns may not be available in the [Weather API](https://openweathermap.org/) from where humidity, temperature data is fetched.

- Fertilizer suggestion system ==> Enter the nutrient contents of your soil and the crop you want to grow. The algorithm will tell which nutrient the soil has excess of or lacks. Accordingly, it will give suggestions for buying fertilizers.

- Disease Detection System ==> Upload an image of leaf of your plant. The algorithm will tell the crop type and whether it is diseased or healthy. If it is diseased, it will tell you the cause of the disease and suggest you how to prevent/cure the disease accordingly.
Note that, for now it only supports following crops

<details>
  <summary>Supported crops
</summary>

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Pepper
- Orange
- Peach
- Potato
- Soybean
- Strawberry
- Tomato
- Squash
- Raspberry
</details>

## How to run locally 🛠️
- Make sure you have [Python](https://www.python.org/downloads/) installed on your system
- Clone this project:
  ```bash
  git clone https://github.com/SnehaRathi132/CROP_IQ.git
  cd CROP_IQ
  ```
- Create a virtual environment:
  ```bash
  python -m venv venv
  # On Windows:
  venv\Scripts\activate
  # On macOS/Linux:
  source venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Optional: Set your OpenWeatherMap API key for live weather:
  ```bash
  # On Windows:
  set OPENWEATHER_API_KEY="your_key_here"
  # On macOS/Linux:
  export OPENWEATHER_API_KEY="your_key_here"
  ```
- Run the application:
  ```bash
  python app/app.py
  ```
- Open your browser and go to `http://localhost:5000`

## Updated Training Pipeline (Archive Datasets)
- Train the crop + fertilizer models from the `archive/` datasets:
  ```
  python app/scripts/train_models.py
  ```
- Generate EDA summaries (saved to `reports/eda_summary.json`):
  ```
  python app/scripts/eda_report.py
  ```
- The Flask app reads the latest models from `app/models/`.

## API Endpoints
- `GET /api/weather?lat=...&lon=...` or `GET /api/weather?city=...` for live weather (server-side key)
- `POST /api/crop` with JSON: `nitrogen`, `phosphorous`, `pottasium`, `ph`, `rainfall`, plus either `city` or `temperature` + `humidity`
- `POST /api/fertilizer` with JSON: `nitrogen`, `phosphorous`, `pottasium`, `moisture`, `soil_type`, `crop_type`, plus either `city` or `temperature` + `humidity`

