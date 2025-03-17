from tensorflow import keras

# Load your model
model = keras.models.load_model("stock_model.keras")

# Print model summary
model.summary()