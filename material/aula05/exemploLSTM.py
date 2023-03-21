# Numpy
import numpy as np

# Pandas
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import metrics

from sklearn.model_selection import train_test_split
# Gráficos
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as dates

df = pd.read_csv("petr4_treinamento.csv",index_col='Date',parse_dates=True)
pd.set_option('display.float_format','{:.2f}'.format)
df.isnull().sum()
df = df.dropna()

df_open = df.iloc[:,1:2].values
min_scaler = MinMaxScaler(feature_range=(0,1))
df_open = min_scaler.fit_transform(df_open)

x_train, y_train= [],[]
x_days = 60

for i in range(x_days,len(df_open)):
  x_train.append(df_open[i-x_days:i,0])
  y_train.append(df_open[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

x_train_spl, x_test_spl, y_train_spl, y_test_spl = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

taxa_esquecimento = 0.25

modelo = Sequential()

modelo.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
modelo.add(Dropout(taxa_esquecimento))

modelo.add(LSTM(units= 50,return_sequences=True,activation = 'tanh'))
modelo.add(Dropout(taxa_esquecimento))

modelo.add(LSTM(units= 50,return_sequences=True,activation = 'tanh'))
modelo.add(Dropout(taxa_esquecimento))

modelo.add(LSTM(units= 50,return_sequences=False,activation = 'tanh'))
modelo.add(Dropout(taxa_esquecimento))

# Alterar Ativação
modelo.add(Dense(units=1,activation = 'sigmoid'))

# Alterara Otimizador
modelo.compile(optimizer='adam',loss='mean_squared_error', metrics = [metrics.RootMeanSquaredError(name="root_mean_squared_error")])
modelo.fit(x_train_spl, y_train_spl,epochs=70,batch_size=32)

gabarito = y_test_spl
y_test_spl = modelo.predict(x_test_spl)

gabarito = gabarito.reshape(237, 1)
y_test_spl = y_test_spl.reshape(237, 1)

gabarito = min_scaler.inverse_transform(gabarito)
y_test_spl = min_scaler.inverse_transform(y_test_spl)

fig, ax = plt.subplots(1,1, figsize=(12,8))
ax.plot(gabarito, 'g-', label='Valor Real')
ax.plot(y_test_spl, 'r-',label='Previsto')
ax.set_xlabel('Quantidade de dias')
ax.set_ylabel('Preço')
ax.set_title('Evolução do valor de Open')
ax.legend()
plt.rcParams.update({'font.size': 12})
fig.gca().spines['top'].set_visible(False)
fig.gca().spines['right'].set_visible(False)
plt.show()