import numpy as np
import os
from typing import Union

from scipy.signal import butter, filtfilt
from scipy.integrate import simpson

from .Utils import norm, plot_traces


def sklearn_available(
        X_train : np.array, 
        X_test : np.array, 
        path_save : str, 
        function, 
        custom_name : Union[None, str] = None,
        **param
    ):
    """

    Execute a dimensionality reduction available in the scikit-learn library.
    If the method has a transformation, it will be trained using X_train the 
    output will be the transformation of X_test. Otherwise the X_test will directly
    be transformed using the `fit_transform` method.

    Parameters
    ----------
    X_train : ndarray
        Training dataset.
    X_test : ndarray
        Testing dataset.
    path_save : str
        Path when the results are stored and loaded.
    function : Sklearn method
        Dimensionality reduction method that is executed.
    param : kwargs
        Parameters used in `function`.

    Returns
    -------
    X_low : ndarray
        Low-dimensional representation of the samples.
    """

    method = function(**param)
    if custom_name is None:
        file_name = f"{function}_{param}"
        file_name = f"{path_save}/{''.join(file_name.split('.')[3:])}.npy"
        file_name = file_name.replace('<', '')
        file_name = file_name.replace('>', '')
        file_name = file_name.replace(':', '')
        file_name = file_name.replace(r"'", '')
    else:
        file_name = f"{path_save}/{custom_name}"

    if os.path.isfile(file_name):
        X_low = np.load(file_name)
    else:
        if hasattr(method, 'transform'):
            trained = method.fit(X_train)
            X_low = trained.transform(X_test)
        else:
            print('Method does not have a transform function')
            X_low = method.fit_transform(X_test)

        X_low = norm(X_low)
        np.save(file_name, X_low)

    return X_low


def area(
        X_high : np.array,
        filtering : bool = False,
        critical_frequency : float = 0.1,
        threshold : float = None,
        plot_filter : bool = False,
        save_path : Union[str,None] = None, 
    ):
    """
    Area of signals.

    Parameters
    ----------
    X_high : ndarray
    filter : bool
        Filter traces.
    plot_traces : bool
        Plot filtered traces (if filtered).

    Returns
    -------
    Area : ndarray

    """
    file_name = f'{save_path}/AREA.npy'

    if os.path.isfile(file_name):
        X_low = np.load(file_name)
    else:
        if filtering:
            b, a = butter(5, critical_frequency, 'low')
            X_high = filtfilt(b, a, X_high)

        if threshold is not None:
            X_high[X_high < threshold] = threshold

        if plot_filter:
            plot_traces(X_high)

        X_low = simpson(X_high).reshape(-1,1)
        X_low = norm(X_low).reshape(-1,1)

        if save_path is not None:
            np.save(file_name, X_low)
        else:
            pass

    return X_low


def max_value(
        X_high : np.array,
        critical_frequency : float = 0.1,
        filtering : bool = False,
        save_path : Union[str,None] = None):
    """

    Maximum value of signals.

    Parameters
    ----------
    X_high : ndarray

    Returns
    -------
    norm : ndarray

    """
    file_name = f'{save_path}/MAX.npy'

    if os.path.isfile(file_name):
        X_low = np.load(file_name)
    else:
        if filtering:
            b, a = butter(5, critical_frequency, 'low')
            X_high = filtfilt(b, a, X_high)

        X_low = norm(X_high.max(axis = 1)).reshape(-1,1)

        if save_path is not None:
            np.save(file_name, X_low)
        else:
            pass

    return X_low




