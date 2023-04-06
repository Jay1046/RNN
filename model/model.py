# def get_model():
#     model = Sequential()
#     model.add(LSTM(units=40, activation='relu', return_sequences=True, input_shape=(1,5)))
#     model.add(LSTM(units=80, activation='relu', return_sequences=True))
#     model.add(LSTM(units=40, activation='relu', return_sequences=True))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     return model

import tensorflow as tf

__all__ = ['MyModel']

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(units=40, activation='relu', return_sequences=True, input_shape=(1,5))
        self.lstm2 = tf.keras.layers.LSTM(units=80, activation='relu', return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(units=40, activation='relu', return_sequences=True)
        self.d1 = tf.keras.layers.Dense(1)
        
        
    def call(self, x):
        out = self.lstm1(x)
        out = self.lstm2(out)
        out = self.lstm3(out)
        out = self.d1(out)
        return out
        
    
    
    
    
    
    
    