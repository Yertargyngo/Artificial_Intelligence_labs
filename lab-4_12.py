import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Загрузка предобученной модели ResNet50
model = ResNet50(weights='imagenet')

# Функция для предобработки изображения
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Функция для предсказания класса
def predict_class(img_path):
    processed_image = preprocess_image(img_path)
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0]

# Пример использования
img_path = 'C:/Users/Yertargyn/Desktop/Picture Medicine/ggg.jpeg'
predicted_class = predict_class(img_path)

print(f'Прогнозируемый класс: {predicted_class[1]}, Вероятность: {predicted_class[2]}')
