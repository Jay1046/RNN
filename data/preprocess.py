from sklearn.preprocessing import MinMaxScaler
import numpy as np

def set_data(data, step):
    
    data = data.set_index('Date')
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X = scaled.tolist()
    
    x, y = [], []
    
    for i in range(len(X)-5):
        d = i + step
        x.append(X[i:d])
        y.append(X[d])
        
    train_size = int(len(X) * 0.8) 
    
    X_train = np.array(x[0:train_size]).reshape(-1, 1, step)
    y_train = np.array(y[0:train_size])

    X_test = np.array(x[train_size:]).reshape(-1, 1, step)
    y_test = np.array(y[train_size:])
    
    return (X_train, y_train), X_test, y_test
    
    