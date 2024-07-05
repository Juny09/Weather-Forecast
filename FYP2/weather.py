import streamlit as st
import requests
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import datetime
import folium
from streamlit_folium import folium_static

import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


api_key = "506de6f6c7efce49dade9e9d28dc3421"

aqi_mapping = {
    1: 'Good',
    2: 'Fair',
    3: 'Moderate',
    4: 'Poor',
    5: 'Very Poor'
}

def fetch_and_prepare_data(lat, lon):
    # url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&appid={api_key}"
    # response = requests.get(url)

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Automatically set end_date to today's date
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2008-12-13",
        "end_date": "2015-12-13",
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", "weather_code", "pressure_msl", "cloud_cover",  "wind_speed_10m"]
    }
    responses = openmeteo.weather_api(url, params=params)

    if responses:
        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()

        hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
        hourly_rain = hourly.Variables(3).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
        hourly_weather_code = hourly.Variables(5).ValuesAsNumpy()
        hourly_pressure_msl = hourly.Variables(6).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(7).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()


        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s"),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m

        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["rain"] = hourly_rain
        hourly_data["snowfall"] = hourly_snowfall

        hourly_data["weather_code"] = hourly_weather_code
        hourly_data["pressure_msl"] = hourly_pressure_msl
        hourly_data["cloud_cover"] = hourly_cloud_cover
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m


        df = pd.DataFrame(data = hourly_data)
        
        # Drop rows with NaNs
        df.dropna(inplace=True)

        open_meteo_weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snowfall",
            73: "Moderate snowfall",
            75: "Heavy snowfall",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }
        # Define the function to extract weather
        def extract_weather_main(row):
            if 'weather_code' in row and row['weather_code'] is not None:
                return open_meteo_weather_codes.get(row['weather_code'], 'Unknown')
            return 'Unknown'
        
        # Apply the function to create the label column
        df['weather_condition'] = df.apply(extract_weather_main, axis=1)


        # Convert the 'label' column to categorical type
        df['weather_condition'] = pd.Categorical(df['weather_condition'])

        # Feature engineering
        features = [
            'temperature_2m', 
            'relative_humidity_2m', 
            'wind_speed_10m', 
            'pressure_msl', 
            'rain', 
            'snowfall',
            'cloud_cover',
            'precipitation'

        ]
        X = df[features]
        y = df['weather_condition']

        # Standardizing the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Applying PCA
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=42)

        # Model training
        clf = KNeighborsClassifier(n_neighbors=13)

        #setting instance of MLP and some basic parameter to decrease the training and search time
        # from sklearn.neural_network import MLPClassifier
        # clf = MLPClassifier(activation = 'tanh', alpha = 0.05, learning_rate = 'constant', solver = 'adam',
        #                     hidden_layer_sizes = (700, 700, 700, 700), max_iter=10000)

        clf.fit(X_train,y_train)

        # Model evaluation
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)
        print('Prediction distribution on test set:', np.unique(y_pred, return_counts=True))
        report = classification_report(y_test, y_pred)
        print(report)

        # Save models (scaler, PCA, classifier)
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        pickle.dump(pca, open('pca_model.pkl', 'wb'))
        pickle.dump(clf, open('model.pkl', 'wb'))

        return df, accuracy, report
    else:
        st.error(f"Failed to fetch data")
        return pd.DataFrame()

def get_hourly_forecast(city, api_key):
        url = f"https://pro.openweathermap.org/data/2.5/forecast/hourly?q={city}&appid={api_key}"
        response = requests.get(url)

        if response.status_code == 200:
            forecast_data = response.json()
            forecast_list = []
            for hour in forecast_data['list']:
                dt = datetime.datetime.fromtimestamp(hour['dt'])
                date = dt.strftime('%Y-%m-%d')
                hour_of_day = dt.strftime('%H:%M')  # Add hour of the day
                temp = round(hour['main']['temp'] - 273.15)  # Convert Kelvin to Celsius
                humidity = hour['main']['humidity']
                wind_speed = hour['wind']['speed']
                clouds = hour['clouds']['all']
                pressure = hour['main']['pressure']
                feels_like = round(hour['main']['feels_like'] - 273.15)  # Convert Kelvin to Celsius
                rain = hour.get('rain', {}).get('1h', 0)
                snow = hour.get('snow', {}).get('1h', 0)
                precipation = hour.get('pop', 0)
                icon = hour['weather'][0]['icon']

                forecast_list.append([date, hour_of_day, temp, humidity, wind_speed, clouds, pressure, feels_like, rain, snow, precipation,icon])
                # Include hour_of_day in the DataFrame
            return pd.DataFrame(forecast_list, columns=['Date', 'Time', 'Temp', 'Humidity', 'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Rain', 'Snow', 'Precipation','Icon'])
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
            return pd.DataFrame()
        
def preprocess_and_predict(forecast_df, scaler, pca, model):
    # Ensure the DataFrame is sorted by date and time if not already
    forecast_df.sort_values(by=['Date', 'Time'], inplace=True)

    # Initialize an empty list for predictions
    predictions = []

    # Loop through the DataFrame and make predictions for each hour
    for _, row in forecast_df.iterrows():
        # Extract features for the current row
        features = np.array([
            row['Temp'],  
            row['Humidity'],    
            row['Wind Speed'],   
            row['Pressure'],
            row['Rain'],
            row['Snow'],
            row['Clouds'],
            row['Precipation']  
        ]).reshape(1, -1)

        # Scale and transform the features
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Make a prediction and add it to the list
        prediction = model.predict(features_pca)
        predictions.append(prediction[0])

    # Print the predictions (optional, can be removed for production)
    print(predictions)

    return predictions

# Fetching and preparing data
def Dfetch_and_prepare_data(lat, lon):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        # Automatically set end_date to today's date
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")
        # API request parameters
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2003-12-13",
            "end_date": today_date,
            "daily": ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                    "precipitation_sum", "rain_sum", "snowfall_sum",
                    "wind_speed_10m_max", "wind_gusts_10m_max"]
        }

        responses = openmeteo.weather_api(url, params=params)

        if responses:
                response = responses[0]

                daily = response.Daily()
                daily_weather_code = daily.Variables(0).ValuesAsNumpy()
                daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
                daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
                daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
                daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
                daily_rain_sum = daily.Variables(5).ValuesAsNumpy()
                daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
                daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()

                daily_data = {"date": pd.date_range(
                    start = pd.to_datetime(daily.Time(), unit = "s"),
                    end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
                    freq = pd.Timedelta(seconds = daily.Interval()),
                    inclusive = "left"
                )}

                daily_data["weather_code"] = daily_weather_code
                daily_data["temperature_2m_max"] = daily_temperature_2m_max
                daily_data["temperature_2m_min"] = daily_temperature_2m_min
                daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
                daily_data["precipitation_sum"] = daily_precipitation_sum
                daily_data["rain_sum"] = daily_rain_sum
                daily_data["snowfall_sum"] = daily_snowfall_sum
                daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max

                # Create DataFrame
                df = pd.DataFrame(data=daily_data)

                open_meteo_weather_codes = {
                    0: "Clear sky",
                    1: "Mainly clear",
                    2: "Partly cloudy",
                    3: "Overcast",
                    45: "Fog",
                    48: "Depositing rime fog",
                    51: "Light drizzle",
                    53: "Moderate drizzle",
                    55: "Dense drizzle",
                    56: "Light freezing drizzle",
                    57: "Dense freezing drizzle",
                    61: "Slight rain",
                    63: "Moderate rain",
                    65: "Heavy rain",
                    66: "Light freezing rain",
                    67: "Heavy freezing rain",
                    71: "Slight snowfall",
                    73: "Moderate snowfall",
                    75: "Heavy snowfall",
                    77: "Snow grains",
                    80: "Slight rain showers",
                    81: "Moderate rain showers",
                    82: "Violent rain showers",
                    85: "Slight snow showers",
                    86: "Heavy snow showers",
                    95: "Thunderstorm",
                    96: "Thunderstorm with slight hail",
                    99: "Thunderstorm with heavy hail"
                }
                # Define the function to extract weather
                def extract_weather_main(row):
                    if 'weather_code' in row and row['weather_code'] is not None:
                        return open_meteo_weather_codes.get(row['weather_code'], 'Unknown')
                    return 'Unknown'

                # Apply the function to create the label column
                df['weather_condition'] = df.apply(extract_weather_main, axis=1)

                # Convert the 'label' column to categorical type
                df['weather_condition'] = pd.Categorical(df['weather_condition'])

                # Feature engineering
                features = [
                    'temperature_2m_mean',
                    'temperature_2m_max',
                    'temperature_2m_min',
                    'wind_speed_10m_max',
                    'rain_sum',
                    'snowfall_sum'
                ]

                X = df[features]
                y = df['weather_condition']

                # Splitting the dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

                # Standardizing the features
                Dscaler = StandardScaler()
                X = Dscaler.fit_transform(X_train)

                # Model training
                Dclf = RandomForestClassifier(n_estimators=100)
                Dclf.fit(X_train, y_train)

                # base_clf = RandomForestClassifier(n_estimators=100)
                # clf = AdaBoostClassifier(base_estimator=base_clf, n_estimators=100, random_state=42)
                # clf.fit(X_train, y_train)

        
                # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
                # clf.fit(X_train, y_train)


                # Model evaluation
                y_pred = Dclf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print('Accuracy:', accuracy)
                print('Prediction distribution on test set:', np.unique(y_pred, return_counts=True))
                report = classification_report(y_test, y_pred)
                print(report)

                # Save models (scaler, PCA, classifier)
                pickle.dump(Dscaler, open('dailyscaler.pkl', 'wb'))
                #pickle.dump(pca, open('dailypca_model.pkl', 'wb'))
                pickle.dump(Dclf, open('dailymodel.pkl', 'wb'))


                return df, accuracy, report
        else:
            st.error(f"Failed to fetch data")
            return pd.DataFrame()
        
# Function to get daily weather forecast
def get_daily_forecast(city, api_key):
    url = f"https://api.openweathermap.org/data/2.5/forecast/daily?q={city}&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        forecast_data = response.json()
        forecast_list = []
        for day in forecast_data['list']:
            date = datetime.datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
            sunrise = datetime.datetime.fromtimestamp(day['sunrise']).strftime('%H:%M:%S')
            sunset = datetime.datetime.fromtimestamp(day['sunset']).strftime('%H:%M:%S')
            temp = round(day['temp']['day'] - 273.15)  # Convert Kelvin to Celsius
            humidity = day['humidity']
            wind_speed = day['speed']
 
            clouds = day['clouds']
            pressure = day['pressure']
            feels_like = round(day['feels_like']['day'] - 273.15)

            rain = day.get('rain', 0)  # Default to 0 if 'rain' is not present
            snow = day.get('snow', 0)  # Default to 0 if 'snow' is not present
            temp_min = round(day['temp']['min'] - 273.15)
            temp_max = round(day['temp']['max'] - 273.15)

            forecast_list.append([date, sunrise, sunset, temp, humidity, wind_speed, clouds, pressure, feels_like, temp_min, temp_max, rain, snow])
        return pd.DataFrame(forecast_list, columns=['Date', 'Sunrise', 'Sunset', 'Temp', 'Humidity', 'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Temp_Min', 'Temp_Max', 'Rain', 'Snow'])
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return pd.DataFrame()

def daily_preprocess_and_predict(Dforecast_df, Dscaler, Dmodel):
    predictions = []
    for _, row in Dforecast_df.iterrows():
        features = np.array([
            row['Temp'],
            row['Wind Speed'],
            row['Rain'],
            row['Snow'],
            row['Temp_Min'],
            row['Temp_Max'],

        ]).reshape(1, -1)
        features_scaled = Dscaler.transform(features)
        #features_pca = pca.transform(features_scaled)
        prediction = Dmodel.predict(features_scaled)
        predictions.append(prediction[0])
    return predictions

def daily_page():
    st.title('Daily Forecast')

    # Sidebar for user input
    city = st.sidebar.text_input('Enter a city name:', 'Cyberjaya')
    submit_button = st.sidebar.button('Get Weather Data')

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
            weather_data = response.json()

            # Get the current date
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # Display the date in a header
            st.header(f"{city} {current_date}")

            # Display weather data using columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                 st.metric(label="🌡️ Temperature", value=f"{round(weather_data['main']['temp'])} °C")
                 st.metric(label="💨 Wind Speed", value=f"{weather_data['wind']['speed']} m/s")
            with col2:
                 st.metric(label="🌡️ Feels like", value=f"{round(weather_data['main']['feels_like'])}°C")
                 st.metric(label="🔽 Pressure", value=f"{weather_data['main']['pressure']} hPa")
            with col3:
                 st.metric(label="☁️ Clouds", value=f"{weather_data['clouds']['all']}%")
                 st.metric(label="👓 Visibility", value=f"{weather_data['visibility']/1000} KM")
            with col4:
                 st.metric(label="💧 Humidity", value=f"{weather_data['main']['humidity']}%")
    if submit_button:
        if city:
            # Fetch weather data using OpenWeatherMap API
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
            geocode_response = requests.get(geocode_url)
            geocode_data = geocode_response.json()
            if geocode_data:
                lat = geocode_data[0]['lat']
                lon = geocode_data[0]['lon']
                df,accuracy, report = Dfetch_and_prepare_data(lat, lon)

                url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                response = requests.get(url)
   
                # Display accuracy on the Streamlit page
                st.write(f'Model Accuracy: {accuracy:}')
                st.text('Classification Report:')
                st.text(report)


                if response.status_code == 200:

                    # Load models
                    loaded_scaler = pickle.load(open('dailyscaler.pkl', 'rb'))
                    #loaded_pca = pickle.load(open('dailypca_model.pkl', 'rb'))
                    loaded_model = pickle.load(open('dailymodel.pkl', 'rb'))



                    # Fetch 30 days forecast
                    Dforecast_df = get_daily_forecast(city, api_key)
                    if not Dforecast_df.empty:
                        # Preprocess and predict
                        # predictions = daily_preprocess_and_predict(Dforecast_df, loaded_scaler, loaded_model)
                        # Dforecast_df['Predictions'] = predictions
                        # columns_to_display = ['Date', 'Sunrise', 'Sunset', 'Temp', 'Humidity',
                        #                       'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Predictions']

                        # st.header("7 Days Weather Forecast with Predictions")
                        # st.dataframe(Dforecast_df[columns_to_display])
                        Dforecast_df = Dforecast_df.head(96)  # Assuming the dataframe is hourly
                        predictions = daily_preprocess_and_predict(Dforecast_df, loaded_scaler, loaded_model)
                        Dforecast_df['Predictions'] = predictions
                        columns_to_display = ['Date', 'Sunrise', 'Sunset', 'Temp', 'Humidity',
                                                'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Predictions']

                        st.header("7 Days Weather Forecast with Predictions")
                        st.dataframe(Dforecast_df[columns_to_display])
                        # Display the formatted forecast layout
                        display_daily_forecast_layout(Dforecast_df)


                    else:
                        st.error("Failed to retrieve forecast data.")
                elif submit_button and not city:
                    st.error("Please enter a city name.")
            pass

def news_page():
    st.title('Latest News')

    # Sidebar for user input
    query = st.sidebar.text_input("Enter a keyword", "Weather")
    from_date = st.sidebar.date_input("From Date", datetime.date.today() - datetime.timedelta(days=7))
    to_date = st.sidebar.date_input("To Date", datetime.date.today())

    # Mediastack API endpoint and API key
    api_key = "df5fbc6bfe9663cdcdd45aa0514377d3"  # Replace with your Mediastack API key
    endpoint = "http://api.mediastack.com/v1/news"

    # API call to Mediastack API
    params = {
        "access_key": api_key,
        "keywords": query,
        "languages": "en",  # Fetching only English news
        "sort": "published_desc",
    }

    # Adjust the date parameter based on user input
    if from_date is not None and to_date is not None:
        params['date'] = f"{from_date.isoformat()},{to_date.isoformat()}"

    response = requests.get(endpoint, params=params)
    news_data = response.json()

    # Display news cards
    if "data" in news_data:
        articles = news_data["data"]
        for article in articles:
            with st.container():
                col1, col2 = st.columns([1, 3])  # Adjust column ratio as needed
                with col1:
                    if article.get('image'):
                        st.image(article['image'], use_column_width=True)
                with col2:
                    st.subheader(article.get('title'))
                    st.write(f"**Source:** {article.get('source')}")
                    st.write(f"**Published At:** {article.get('published_at')}")
                    st.write(article.get('description'))
                    st.write(f"[Read More]({article.get('url')})", unsafe_allow_html=True)
    else:
        st.error("Failed to fetch news data. Please check your API key and try again.")

# Function to fetch air pollution data
def fetch_air_pollution_data(lat, lon):
    air_pollution_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        response = requests.get(air_pollution_url)
        air_pollution_data = response.json()
        return air_pollution_data
    except Exception as e:
        st.error(f"Failed to fetch air pollution data: {e}")
        return None



# Function to display map
def display_map(location, layer, zoom):
    geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={api_key}"
    geocode_response = requests.get(geocode_url)
    geocode_data = geocode_response.json()

    if geocode_data:
        lat = geocode_data[0]['lat']
        lon = geocode_data[0]['lon']
        map = folium.Map(location=[lat, lon], zoom_start=zoom)

        tile_url = f'https://tile.openweathermap.org/map/{layer}/{{z}}/{{x}}/{{y}}.png?appid={api_key}'
        folium.TileLayer(tile_url, attr='OpenWeatherMap').add_to(map)

        folium_static(map)
    else:
        st.error("Could not find the location.")

def display_forecast_layout(forecast_df):
    # Group the DataFrame by the 'Date' column
    grouped = forecast_df.groupby('Date') 
    
    for date, group in grouped:
        # Parse the date string to a datetime object
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d') # Convert string to datetime object
        # Format the datetime object to get the day name
        day_of_week = date_obj.strftime('%A') 
        
        # Display the day of the week and the date
        st.subheader(f"{day_of_week} , {date}")  # Display the day of the week and the date
        
        
        for _, row in group.iterrows():
            with st.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f"<h3 style='text-align: center; margin-bottom:0;'>{row['Time']}</h3>", unsafe_allow_html=True)
                    #st.image(f"https://openweathermap.org/img/wn/{row['Icon']}.png")
                with col2:

                    st.metric(label="🌡️ Temperature", value=f"{row['Temp']} °C")
                    st.metric(label="🌡️ Feels Like", value=f"{row['Feels like']} °C")
                    st.markdown(f"<h8 style='text-align: center; margin-bottom:0;'>🔽 Pressure</h8>    <h3 style='text-align: center;'>{row['Pressure']} hPa</h3>", unsafe_allow_html=True)
                    #st.metric(label="🔽 Pressure", value=f"{row['Pressure']} hPa")
                with col3:
                    st.metric(label="💨 Wind Speed", value=f"{row['Wind Speed']} m/s")
                    st.metric(label="☁️ Clouds", value=f"{row['Clouds']}%")
                with col4:
                    
                    st.metric(label="💧 Humidity", value=f"{row['Humidity']}%")
                    st.metric(label="🌧️ Prob. of Precip.", value=f"{row['Precipation'] * 100:.0f}%")
                with col5:
                    st.markdown(f"<h6 style='text-align: center; margin-bottom:0;'>🌤️ Predictions</h6> <h3 style='text-align: center;'>{row['Predictions']}</h3>", unsafe_allow_html=True)
                st.markdown("---")                    

def display_daily_forecast_layout(Dforecast_df):
    # Group the DataFrame by the 'Date' column
    grouped = Dforecast_df.groupby('Date') 
    
    for date, group in grouped:
        date_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
        day_of_week = date_obj.strftime('%A')
        st.subheader(f"{day_of_week}, {date}")
        
        for _, row in group.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="Sunrise", value=f"{row['Sunrise']}")
                    st.metric(label="🌡️ Temperature", value=f"{row['Temp']} °C")
                    st.metric(label="🌡️ Feels Like", value=f"{row['Feels like']} °C")
                with col2:
                    st.metric(label="Sunset", value=f"{row['Sunset']}")
                    st.metric(label="💨 Wind Speed", value=f"{row['Wind Speed']} m/s")
                    st.metric(label="☁️ Clouds", value=f"{row['Clouds']}%")
                with col3:
                    st.metric(label="💧 Humidity", value=f"{row['Humidity']}%")
                    st.metric(label="🔽 Pressure", value=f"{row['Pressure']} hPa")
                with col4:
                    st.markdown(f"<h6 style='text-align: center; margin-bottom:0;'>🌤️ Predictions</h6> <h3 style='text-align: center;'>{row['Predictions']}</h3>", unsafe_allow_html=True)
                    pass
                st.markdown("---")


def map_page():
    st.title('Weather Map')
    st.header("Map Settings")
    layer_options = {
            'Precipitation': 'precipitation_new',
            'Temperature': 'temp_new',
            'Wind Speed': 'wind_new',
            'Clouds': 'clouds_new',
            'Sea Level Pressure': 'pressure_new'
    }
    layer = st.selectbox("Select Layer", list(layer_options.keys()))
    zoom = st.slider("Zoom Level", 1, 18, 14)
    # Sidebar for user inputs
    with st.sidebar:
        location = st.text_input('Enter a location:', 'Cyberjaya')
        submit_button = st.button('Get Map')

    if submit_button:
        if location:

            st.header(f"{layer} Map")
            map_col, _ = st.columns([1, 1])  # Adjust ratio as needed
            with map_col:
                display_map(location, layer_options[layer], zoom)
        else:
            st.error("Please enter a location.")

def air_pollution_page():
    st.title('Air Pollution')

    # Sidebar for user input
    city = st.sidebar.text_input('Enter a city name:', 'Cyberjaya')
    date = st.sidebar.date_input("Select a date for Solar", datetime.date.today())
    submit_button = st.sidebar.button('Get Air Pollution Data')

    if submit_button:
        if city:
            # Fetch weather data using OpenWeatherMap API
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
            geocode_response = requests.get(geocode_url)
            geocode_data = geocode_response.json()
            if geocode_data:
                lat = geocode_data[0]['lat']
                lon = geocode_data[0]['lon']
                air_pollution_data = fetch_air_pollution_data(lat, lon)
                aqi = air_pollution_data['list'][0]['main']['aqi']
                aqi_mapping = {
                    1: "Good",
                    2: "Fair",
                    3: "Moderate",
                    4: "Poor",
                    5: "Very Poor"
                }
                
                aqi_text = aqi_mapping.get(aqi, "Unknown")
                latest_data = air_pollution_data['list'][0]
                components = latest_data['components']

                def get_aqi_description(aqi):
                    if aqi == 1:
                        return "Air quality is considered satisfactory, and air pollution poses little or no risk."
                    elif aqi == 2:
                        return "Air quality is acceptable; however, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
                    elif aqi == 3:
                        return "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
                    elif aqi == 4:
                        return "Health alert: everyone may experience more serious health effects."
                    elif aqi == 5:
                        return "Health warnings of emergency conditions. The entire population is more likely to be affected."
                    else:
                        return "Unknown air quality index."
                    
                if air_pollution_data:
                    # Display the date in a header
                    st.header(f"{city} Air Pollution Data")
                    aqi_description = get_aqi_description(aqi)
                    st.markdown(f"<h1 style='font-size: 24px;'>Air Quality Index (AQI): {aqi_text}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 16px;margin-bottom: 50px;'>{aqi_description}</div>", unsafe_allow_html=True)
                    
                    # Display air pollution data using columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="🌡️ CO", value=f"{air_pollution_data['list'][0]['components']['co']} μg/m³")
                        st.metric(label="🔽 SO₂", value=f"{air_pollution_data['list'][0]['components']['so2']} μg/m³")
                        st.metric(label="🌧️ PM₁₀", value=f"{air_pollution_data['list'][0]['components']['pm10']} μg/m³")
                    with col2:
                        st.metric(label="💨 NO", value=f"{air_pollution_data['list'][0]['components']['no']} μg/m³")
                        st.metric(label="💧 O₃", value=f"{air_pollution_data['list'][0]['components']['o3']} μg/m³")
                        st.metric(label="🌧️ NH₃", value=f"{air_pollution_data['list'][0]['components']['nh3']} μg/m³")
                    with col3:
                        st.metric(label="☁️ NO₂", value=f"{air_pollution_data['list'][0]['components']['no2']} μg/m³")
                        st.metric(label="🌧️ PM₂.₅", value=f"{air_pollution_data['list'][0]['components']['pm2_5']} μg/m³")
                    st.markdown("---")
                else:
                    st.error("Failed to retrieve air pollution data.")
            else:
                st.error("Could not find the location.")
        else:
            st.error("Please enter a city name.")


# Streamlit app
def main():
    # Sidebar for user inputs
    with st.sidebar:
        choice = st.sidebar.selectbox("Choose a page", ["Weather Forecast", "Daily Forecast", "Weather Map", "Air Pollution", "News"])

    if choice == "Weather Forecast":
        st.title('Weather Forecast')

        with st.sidebar:
            city = st.text_input('Enter a city name:', 'Cyberjaya')
            submit_button = st.button('Get Weather Data')

            # Add a button to toggle the visibility of the 4-day forecast
            #show_forecast_button = st.sidebar.checkbox("Show 4 Day Forecast")

            forecast_day = st.sidebar.radio("Select Forecast Day:", ["Day 1", "Day 2", "Day 3", "Day 4", "4 Days"])

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            weather_data = response.json()

            # Get the current date
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # Display the date in a header
            st.header(f"{city} {current_date}")

            # Display weather data using columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                 st.metric(label="🌡️ Temperature", value=f"{round(weather_data['main']['temp'])} °C")
                 st.metric(label="💨 Wind Speed", value=f"{weather_data['wind']['speed']} m/s")
            with col2:
                 st.metric(label="🌡️ Feels like", value=f"{round(weather_data['main']['feels_like'])}°C")
                 st.metric(label="🔽 Pressure", value=f"{weather_data['main']['pressure']} hPa")
            with col3:
                 st.metric(label="☁️ Clouds", value=f"{weather_data['clouds']['all']}%")
                 st.metric(label="👓 Visibility", value=f"{weather_data['visibility']/1000} KM")
            with col4:
                 st.metric(label="💧 Humidity", value=f"{weather_data['main']['humidity']}%")
            st.markdown("---")
            
        if submit_button:
            if city:
                # Fetch weather data using OpenWeatherMap API
                geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
                geocode_response = requests.get(geocode_url)
                geocode_data = geocode_response.json()
                if geocode_data:
                    lat = geocode_data[0]['lat']
                    lon = geocode_data[0]['lon']
                    df,accuracy, report = fetch_and_prepare_data(lat, lon)

                    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
                    response = requests.get(url)

                    # Display accuracy on the Streamlit page
                    st.write(f'Model Accuracy: {accuracy:}')
                    st.text('Classification Report:')
                    st.text(report)

                    if response.status_code == 200:
                        weather_data = response.json()

                        geocode_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={api_key}"
                        geocode_response = requests.get(geocode_url)
                        geocode_data = geocode_response.json()
                        if geocode_data:
                            lat = geocode_data[0]['lat']
                            lon = geocode_data[0]['lon']
                            air_pollution_data = fetch_air_pollution_data(lat, lon)
                            if air_pollution_data and 'list' in air_pollution_data:
                                latest_data = air_pollution_data['list'][0]  # Assuming we want the latest data
                                aqi = latest_data['main']['aqi']
                                aqi_text = aqi_mapping.get(aqi, "Unknown")
                                components = latest_data['components']
                                    
                                # Display AQI in a header
                                st.markdown(f"<h1 style='font-size: 24px;'>Air Quality Index (AQI): {aqi_text}</h1>", unsafe_allow_html=True)

                                # Subtitle for pollutant concentrations
                                st.markdown("<p style='font-size: 18px; font-family: Arial;'>Pollutant Concentrations:</p>", unsafe_allow_html=True)

                                # Create columns for pollutants
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.markdown(f"<p style='font-size: 16px;'>SO2 (Sulfur Dioxide): {components.get('so2', 'N/A')} µg/m³</p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='font-size: 16px;'>NO2 (Nitrogen Dioxide): {components.get('no2', 'N/A')} µg/m³</p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='font-size: 16px;'>PM10 (Particulate Matter 10): {components.get('pm10', 'N/A')} µg/m³</p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='font-size: 16px;'>PM2.5 (Particulate Matter 2.5): {components.get('pm2_5', 'N/A')} µg/m³</p>", unsafe_allow_html=True)

                                with col2:
                                    st.markdown(f"<p style='font-size: 16px;'>O3 (Ozone): {components.get('o3', 'N/A')} µg/m³</p>", unsafe_allow_html=True)
                                    st.markdown(f"<p style='font-size: 16px;'>CO (Carbon Monoxide): {components.get('co', 'N/A')} µg/m³</p>", unsafe_allow_html=True)

                                # Add a horizontal line to separate the sections
                                st.markdown("---")
                            else:
                                st.error("No air pollution data found for the selected location.")
                        else:
                            st.error("Could not find the location for air pollution data.")

                        # Load models
                        loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))
                        loaded_pca = pickle.load(open('pca_model.pkl', 'rb'))
                        loaded_model = pickle.load(open('model.pkl', 'rb'))

                        # Fetch the hourly forecast data
                        forecast_df = get_hourly_forecast(city, api_key)
                        
                        # if forecast_df.empty:
                        #     st.error("Failed to retrieve forecast data.")
                        # else:
                        #     if show_forecast_button:
                        #         # Filter the dataframe to include only the first 96 hours (4 days)
                        #         four_day_forecast_df = forecast_df.head(96)  # Assuming the dataframe is hourly
                        #         predictions = preprocess_and_predict(four_day_forecast_df, loaded_scaler, loaded_pca, loaded_model)
                        #         four_day_forecast_df['Predictions'] = predictions
                        #         columns_to_display = ['Date', 'Time', 'Temp', 'Humidity', 'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Predictions']
                        #         st.header("4 Days Hourly Weather Forecast with Predictions")
                        #         st.dataframe(four_day_forecast_df[columns_to_display])
                        #         # Display the formatted forecast layout
                        #         display_forecast_layout(four_day_forecast_df)
                        #     else:
                        #         # Show 24-hour forecast
                        #         one_day_forecast_df = forecast_df.head(24)  # Assuming the dataframe is hourly
                        #         predictions = preprocess_and_predict(one_day_forecast_df, loaded_scaler, loaded_pca, loaded_model)
                        #         one_day_forecast_df['Predictions'] = predictions
                        #         columns_to_display = ['Date', 'Time', 'Temp', 'Humidity', 'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Predictions']
                        #         st.header("24 Hour Weather Forecast with Predictions")
                        #         st.dataframe(one_day_forecast_df[columns_to_display])
                        #         # Display the formatted forecast layout for 24 hours
                        #         display_forecast_layout(one_day_forecast_df)

                    if not forecast_df.empty:
                        # Determine the slice of the dataframe to display based on selected forecast day
                        if forecast_day == "Day 1":
                            forecast_slice = forecast_df.head(24)
                        elif forecast_day == "Day 2":
                            forecast_slice = forecast_df.iloc[24:48]
                        elif forecast_day == "Day 3":
                            forecast_slice = forecast_df.iloc[48:72]
                        elif forecast_day == "Day 4":
                            forecast_slice = forecast_df.iloc[72:96]
                        elif forecast_day == "4 Days":
                            forecast_slice = forecast_df.iloc[0:96]

                        predictions = preprocess_and_predict(forecast_slice, loaded_scaler, loaded_pca, loaded_model)
                        forecast_slice['Predictions'] = predictions
                        columns_to_display = ['Date', 'Time', 'Temp', 'Humidity', 'Wind Speed', 'Clouds', 'Pressure', 'Feels like', 'Predictions']
                        st.header(f"{forecast_day} Weather Forecast with Predictions")
                        st.dataframe(forecast_slice[columns_to_display])
                        # Display the formatted forecast layout
                        display_forecast_layout(forecast_slice)

                else:
                     # Error in geocode API response
                    st.error("Failed to get city name. Please use the correct city name.")
                    
            elif submit_button and not city:
                st.error("Please enter a city name.")
        pass

    elif choice == "Daily Forecast":
        daily_page()

    elif choice == "Weather Map":
        map_page()

    elif choice == "Air Pollution":
        air_pollution_page()

    elif choice == "News":
        news_page()

if __name__ == "__main__":
    main()
