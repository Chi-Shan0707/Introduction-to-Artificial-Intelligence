import random

CITIES = ["Beijing", "Tokyo", "London", "New York", "Sydney", "Paris"]
CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy", "Windy", "Foggy"]

def generate_forecast(city):
    temp = random.randint(-5, 38)
    humidity = random.randint(20, 95)
    condition = random.choice(CONDITIONS)
    return {
        "city": city,
        "temperature": temp,
        "humidity": humidity,
        "condition": condition,
    }

def display_forecast(forecast):
    print(f"  {forecast['city']}: {forecast['temperature']}°C, "
          f"{forecast['humidity']}% humidity, {forecast['condition']}")

if __name__ == "__main__":
    print("=== Weather Forecast ===")
    for city in CITIES:
        display_forecast(generate_forecast(city))
