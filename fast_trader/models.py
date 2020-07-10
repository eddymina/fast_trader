from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from .utils import TraderError,TraderWarning,isTrader
import numpy as np 

class LinReg:
    def __init__(self,X,y):
        if isTrader(X,y).res:
            self.X= np.stack([X.data,np.ones(X.data.shape[0])],axis=1)
            self.y= y.data
            
        else:
            self.X= np.stack([X,np.ones(X.shape[0])],axis=1)
            self.y= y
        
        self.Xt= self.X.T
        
    def _inv(self,A):
        a,b,c,d= A.flatten()
        return 1/(a*d-b*c)*np.array([[d,-b],[-c,a]])
    
    def fsolve(self):
        return self._inv(self.Xt@self.X)@self.Xt@self.y

    def solve(self,plot=True):
        sol = self.fsolve()
        self.yhat= sol@self.Xt
        diff =(self.yhat-self.y)
        if plot:
            plt.title('Eq= {:2f}x+{:2f}'.format(sol[0],sol[1]))
            plt.scatter(self.X[:,0],self.y)
            plt.plot(self.X[:,0],self.yhat,color='red')
            plt.legend(['Raw Data','Approximate Sol'])
        return {'sol':sol,'mse':(diff**2).mean(),
        'mae':(np.abs(diff)).mean()}


class tLSTM:
    def __init__(self,num_time_steps=50,val_split=None):
        
        self.num_time_steps=num_time_steps
        self.val_split=val_split
        
    def _model(self,input_size):
        model = Sequential()
        model.add(LSTM(units=50,return_sequences=True,input_shape=input_size))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50,return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        return model

    def _batchify (self,datum):
        datum = np.expand_dims(datum.data,axis=1)
        num_timesteps=self.num_time_steps+1
        X_train = []
        y_train = []
        for i in range(num_timesteps, len(datum)):
            X_train.append(datum[i-num_timesteps:i, 0])
            y_train.append(datum[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, y_train
            
    def fit(self,tarr,epochs=10,batch_size=32):
        isTrader(tarr).error()
        self._omin= tarr.min
        self._omax= tarr.max
        
        train = tarr.minmax_scale(0,1)
        val_data= None
        
        if self.val_split:
            train,val= train.split_on(self.val_split)
            val_data= self._batchify(val)

        X_train, y_train= self._batchify(train)
        
        
        self.lstm=self._model(input_size=(X_train.shape[1], 1))
        
        self.lstm.compile(optimizer='adam',loss='mse')
        self.lstm.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,
                     validation_data=val_data)