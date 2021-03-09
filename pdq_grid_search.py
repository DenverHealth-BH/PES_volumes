import itertools
import statsmodels.api as sm


def pdq_grid_search(y, p, d, q, seasonal):
    """
    generates and tests several SARIMAX(pdq)x(PDQ)_Seasonal models of y
    returns best fitting model parameters and AIC value

    If return is [0,1000000], grid search did not identify a model with an AIC lower than that. Likely due to something
    being broken
    :param y: series with datetime index to model
    :param p: max possible value for SARIMA(p)
    :param d: max possible value for SARIMA(d)
    :param q: max possible value for SARIMA(q)
    :param seasonal: Seasonal value, single int. Look at autocorrelation ksplot for best estimate
    :return: list [pdqxPDQ, lowest AIC value obtained
    """
    p = range(0, p)
    d = range(0, d)
    q = range(0, q)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seasonal) for x in pdq]

    arima_model = [0, 0, 1000000]
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit(disp=0)
                if results.aic < arima_model[2]:
                    arima_model[0] = param
                    arima_model[1] = param_seasonal
                    arima_model[2] = results.aic
            except:
                continue
    return arima_model
