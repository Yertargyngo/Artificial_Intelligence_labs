# 5. Прогнозирование потребления электроэнергии: Используйте RNN для
# прогнозирования потребления электроэнергии в определенном районе на основе
# исторических данных

# 5. Электр энергиясын тұтынуды болжау: RNN пайдаланыңыз
# белгілі бір ауданда электр энергиясын тұтынуды болжау
# тарихи деректер

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# Мұнда қажетті кітапханалар импортталады:
# сандық деректермен жұмыс істеу үшін NumPy,
# визуализация үшін Matplotlib, модельді құру және оқыту үшін TensorFlow және Keras.


# Синтетикалық деректерді құру

def generate_data():
    time = np.arange(0, 400, 0.1)
    amplitude = np.sin(time) + np.random.normal(scale=0.3, size=len(time))
    return amplitude

# Бұл функция синтетикалық уақыт қатарларын жасайды.
# Бұл жағдайда бұл кездейсоқ Шу қосылған синусоид.

# Оқу үшін мәліметтер тізбегін құру
def create_sequences(data, seq_length):
    sequences = []
    target = []
    for i in range(len(data)-seq_length):
        seq = data[i:i+seq_length]
        label = data[i+seq_length:i+seq_length+1]
        sequences.append(seq)
        target.append(label)
    return np.array(sequences), np.array(target)

# Бұл мүмкіндік оқу үшін деректер тізбегін жасайды.
# Әрбір реттілікте seq_length элементтері бар, ал оған сәйкес "жауап" келесі элемент болып табылады.

# Гиперпараметры
seq_length = 20  # Длина последовательности (прошлых наблюдений)
num_epochs = 10
batch_size = 64

# Мұнда реттілік ұзындығы, дәуір саны және пакет өлшемі
# (batch size) сияқты модельді оқыту параметрлері орнатылады.

# Синтетикалық деректерді құру
data = generate_data()

# Синтетикалық уақыт қатарлары жасалады.

# Реттілік құру
sequences, target = create_sequences(data, seq_length)

# Оқыту реттілігі және оларға сәйкес "жауаптар"қалыптасады.

# Оқу және тест жиынтықтарына бөлу
split_idx = int(0.8 * len(sequences))
X_train, y_train = sequences[:split_idx], target[:split_idx]
X_test, y_test = sequences[split_idx:], target[split_idx:]
# Деректер оқу және сынақ жинақтары болып бөлінеді.

# RNN моделін құру және оқыту
model = Sequential([
    SimpleRNN(units=64, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)
# Мұнда бір қайталанатын қабаты (SimpleRNN) және
# бір толық байланысқан қабаты (Dense) бар RN моделі жасалады және құрастырылады.


# Модельді бағалау
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Ошибка на тестовых данных: {test_loss}')
# Модель сынақ деректер жиынтығында бағаланады.

# Болжау
predictions = model.predict(X_test)
# Модель сынақ деректер жиынтығында болжау үшін қолданылады.


# Нәтижелерді көру
plt.figure(figsize=(15, 6))
plt.plot(data, label='Original Data', alpha=0.5)
plt.plot(range(split_idx+seq_length, len(data)), predictions, label='Predictions', alpha=0.7)
plt.legend()
plt.show()

# Мұнда нәтижелер көрсетіледі, мұнда көк сызық бастапқы деректерді,
# ал қызғылт сары түс болжамды мәндерді білдіреді.
