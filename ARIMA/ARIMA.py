from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


class myARIMA:
    """
    Simple ARIMA model
    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data: timeseries
            data to fit
        
        """
        self.X = data
        self.model = ARIMA(self.X, order=(1,1,0))

    def fit(self, **kwargs):
        """
        Model fitting
        Parameters
        ----------
        **kwargs: dict
            Key-words parameters for fitting.
        """
        
        self.results = self.model.fit(disp=-1, trend='nc')
        
        
    def predict(self, steps=1):
        """
        Model prediciton
        Parameters
        ----------
        steps: int
            Number of prediction steps
        """

        y_pred = np.empty((steps, ), dtype=np.float64)
        for step in range(steps):
            y_pred[step] = self.results.predict(len(self.X) + step, len(self.X) + step)[-1]
            self.X[:-1] = self.X[1:]
            self.X[-1] = y_pred[step]
            self.model = ARIMA(self.X, order=(1,1,0))
            self.results = self.model.fit(disp=-1, trend='nc')
        return y_pred
