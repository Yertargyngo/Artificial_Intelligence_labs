# 5. Распознавание цифр: Используя датасет MNIST, создайте CNN для классификации
# рукописных цифр

# Сандарды тану: mnist деректер жиынын пайдаланып, жіктеу үшін CNN жасаңыз
# қолмен жазылған сандар

import tensorflow as tf
from tensorflow.keras import layers, models

# TensorFlow-нейрондық желілерді құруға және оқытуға арналған кітапхана.
# layers және models-нейрондық желі архитектурасын анықтауға мүмкіндік беретін Keras кітапханасының құрамдас бөліктері.

# MNIST мәліметтерін жүктеу
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Бұл жолдар MNIST деректер жинағын жүктейді.
# Train_images және test_images суреттерді қамтиды,
# ал train_label және text_labels сәйкес белгілерді (0 - ден 9-ға дейінгі сандар) қамтиды.

# Мәліметтерді дайындау
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# 0 мен 1 аралығында пиксельдер мәндеріне нормализация жасау
train_images, test_images = train_images / 255.0, test_images / 255.0

# Мұнда біз массивтердің пішінін CNN үшін күтілетін форматқа сәйкес өзгертеміз.
# Соңғы өлшем (1) кескіннің тереңдігін білдіреді (ақ-қара сурет).
# Кескін пиксельдері 0-ден 1-ге дейін болатындай етіп қалыпқа келтіріледі.

# CNN модельін құру
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu')
])

# Кодтың бұл бөлігі тізбектелген қабаттар жиынтығы болып табылатын сериялық модельді (Секвенциялық модель) жасайды.
# Біз белгілерді алу үшін конволюциялық қабаттарды, сондай-ақ субдискретизация қабатын (пудинг) қосамыз.

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Конволюциялық қабаттардан кейін біз жіктеу үшін толық байланысқан қабаттарды қосамыз.
# Flatten () көп өлшемді деректерді бір өлшемді деректерге түрлендіреді.
# Dance-Rely (rectified Linear Unit) белсендіру функциясын пайдаланатын 64 нейроны бар толық байланысқан қабат.
# Соңғы қабатта активациясыз 10 Нейрон бар, өйткені бұл 10 сыныпты жіктеу міндеті.

# Модель компиляциясы
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Мұнда біз оңтайландырғышты (бұл жағдайда adam) және
# модельді оқытуға арналған шығын функциясын көрсетеміз.
# Біз сондай-ақ модельдің өнімділігін бағалау үшін метриканы таңдаймыз.

# Модельды оқыту
model.fit(train_images, train_labels, epochs=5)

# Бұл жол Оқу процесін бастайды. Біз жаттығу суреттерін және оларға сәйкес белгілерді береміз

# Модельдің дәлдікті бағалауы
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nТочность на тестовых данных: {test_acc*100:.2f}%')

# Біз сынақ деректер жиынтығында модельдің өнімділігін бағалаймыз және дәлдікті шығарамыз.
