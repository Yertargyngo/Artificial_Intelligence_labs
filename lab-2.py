# 5. Прогноз погоды:
# - Данные: Метеорологические данные за прошлые периоды.
# - Задача: Прогнозирование погоды на следующий день или неделю.

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# pandas - қуатты деректер құрылымдары мен талдауларын ұсынатын деректермен жұмыс істейтін кітапхана.
# matplotlib-графиктер жасауға және деректерді визуализациялауға арналған кітапхана.
# statsmodels - статистикалық модельдерді бағалауға және статистикалық сынақтарды орындауға арналған кітапхана.

# Ақпараттарды жүктеу
data = pd.read_csv('C:/Users/Yertargyn/Desktop/csv_files/wth.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# CSV файлынан деректерді жүктеу. Файл жолы жолда көрсетілген.
# 'Date' бағанын күн форматына түрлендіру.
# 'date' бағанын индекс ретінде орнату.

# Ақпараттарлы дайындау
train_data = data['temperature'].iloc[:-7]
test_data = data['temperature'].iloc[-7:]

# модельді оқытуға арналған деректерді іріктеу, бұл жағдайда бұл соңғы 7 күннен басқа барлық деректер.
# модельді сынау үшін деректерді іріктеу, бұл жағдайда бұл соңғы 7 күн.

# Арима моделін оқытамыз
model = ARIMA(train_data, order=(5,1,0))
model_fit = model.fit()

# деректерге байланысты таңдалатын параметрлерді (p, d,q) көрсететін ARIMA моделін құру.
# модельді оқыту

forecast = model_fit.forecast(steps=7)  # болжамды 7 күнге орындау

# Нәтижелердің визуализация сы
plt.figure(figsize=(10, 5))
plt.plot(train_data.index, train_data.values, label='Оқытылатын ақпараттар')
plt.plot(test_data.index, test_data.values, label='Нақты ақпараттар')
plt.plot(test_data.index, forecast, label='Болжам')
plt.xlabel('Күн')
plt.ylabel('Температура')
plt.legend()
plt.show()

# plt.figure (Сурет=(10, 5)) - өлшемдері 10х5 дюйм болатын жаңа графикалық терезе жасау.
# plot.plot () - график құру. Мұнда оқыту деректерін, нақты деректерді және болжамды көрсету үшін қолданылады.
# plt.xlabel () және plt.ylabel () - сәйкесінше X және Y осьтеріне дайындау.
# plt.legend () - графиктерге аңыз қосу.
# plt.show () - графикті көрсету.