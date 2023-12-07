from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def measure_performance(y_real, y_pred, model_name):
    mae = mean_absolute_error(y_real, y_pred)
    print(f'The Mean Absolute Error (MAE) for the {model_name} is: {mae}')

def get_flight_data():
    flight_hours = float(input("Enter the flight hours: "))
    distance_input = input("Enter the flight distance (in km): ")
    distance = float(''.join(c for c in distance_input if c.isdigit() or c == '.'))
    passengers = float(input("Enter the number of passengers: "))
    strike = float(input("Is there a strike by employees? (1 for Yes, 0 for No): "))
    adverse_weather = float(input("Are there adverse weather conditions? (1 for Yes, 0 for No): "))
    plane_size = float(input("Enter the plane size (in meters): "))
    plane_weight = float(input("Enter the plane weight (in tons): "))
    return [flight_hours, distance, passengers, strike, adverse_weather, plane_size, plane_weight]

user_data = get_flight_data()

X_train = [
    [3.0, 2500, 120, 0, 0, 35, 160],
    [2.5, 2800, 150, 1, 0, 30, 180],
    [2.8, 3200, 180, 0, 1, 40, 200],
    [2.0, 2700, 135, 1, 1, 27, 150],
    [3.2, 3100, 165, 0, 1, 32, 170],
    [3.5, 3600, 200, 1, 1, 37, 220],
]

y_train = [35, 42, 38, 47, 40, 55]

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)

linear_model = LinearRegression()
linear_model.fit(X_train_normalized, y_train)

user_data_normalized = scaler.transform([user_data])
linear_prediction = linear_model.predict(user_data_normalized)
linear_prediction_rounded = round(linear_prediction[0], 2)
print(f'The delay prediction (linear model) for the provided data is: {linear_prediction_rounded} hours')

polynomial_model = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
polynomial_model.fit(X_train_normalized, y_train)

user_data_normalized = scaler.transform([user_data])
polynomial_prediction = polynomial_model.predict(user_data_normalized)
polynomial_prediction_rounded = round(polynomial_prediction[0], 2)
print(f'The delay prediction (polynomial model) for the provided data is: {polynomial_prediction_rounded} hours')

measure_performance([30], [linear_prediction_rounded], "Linear Model")
measure_performance([30], [polynomial_prediction_rounded], "Polynomial Model")
