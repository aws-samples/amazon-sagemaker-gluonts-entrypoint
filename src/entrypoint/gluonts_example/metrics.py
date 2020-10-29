import numpy as np


def mape(y_true, y_pred, version=0) -> float:
    r"""
    .. math::
        mape = mean(|Y - Y_hat| / |Y|))

    Args:
        y_true (np.array): Ground truths
        y_pred (np.array): Forecasts
        version (int, optional): Version 0 is as-is implementation of formula. Version 1 & 2 deal with div-by-0, but
            version=1 ignores those with y=0 from calculation. Defaults to 0.

    Returns:
        array of floats: mape values
    """

    if version == 0:
        return np.mean(np.abs((y_true - y_pred) / y_true))
    elif version == 1:
        # This version takes care of div-by-0, and ignore 0-y in nominator.
        # See: https://github.com/awslabs/gluon-ts/pull/725
        denominator = np.abs(y_true)
        flag = denominator == 0
        return np.mean((np.abs(y_true - y_pred) * (1 - flag)) / (denominator + flag))
    elif version == 2:
        # This version takes care of div-by-0, and include 0-y in nominator.
        denominator = np.abs(y_true)
        flag = denominator == 0
        return np.mean(np.abs(y_true - y_pred) / (denominator + flag))

    raise ValueError(f"Unknown mape version: {version}")


def wmape(actual, forecast, version=0) -> float:
    r"""
    .. math::
        wmape = mape * (actual / sum(actual))

    This implementation assumes actual are positives -- FIXME: would it be
    better to enforce this assumption by using sum(abs(actual))?

    Args:
        actual (np.array): Ground truths
        forecast (np.array): Forecasts
        version (int, optional): mape version to use. See mape(). Defaults to 0 (which
            may perform division-by-zero).

    Returns:
        array of floats: wmape values
    """
    # we take two series and calculate an output a wmape from it.
    # - NOTE: as-is implementation from Logbooks/Med_Low/Weekly_AutoArima.ipynb

    # make a series called mape
    se_mape = mape(actual, forecast, version=version)

    # get a float of the sum of the actual
    # - NOTE: this as-is implementation assumes actual are positives. Would it
    # be better to enforce this assumption by using np.sum(np.abs(actual))?
    ft_actual_sum = actual.sum()

    # get a series of the multiple of the actual & the mape
    se_actual_prod_mape = actual * se_mape

    # summate the prod of the actual and the mape
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()

    # float: wmape of forecast
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum

    # return a float
    return ft_wmape_forecast
