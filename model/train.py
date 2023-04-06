from model import MyModel
from data.load import load_data
from data.preprocess import set_data
import matplotlib.pyplot as plt


def train_model(path, epochs):
    
    data = load_data(path)
    
    
    train, X_test, y_test = set_data(data,step=5)
    
    model = MyModel()
    
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mse'])
    
    print("train starts")
    
    model.fit(*train,epochs=epochs)
    print("train ends")
    
    y_pred = model.predict(X_test)
    
    plt.plot(y_test, color='red', label='Real Y')
    plt.plot(y_pred.reshape(-1), color='blue', label='Predicted Y')
    plt.legend()
    plt.show()
    
    
    

    