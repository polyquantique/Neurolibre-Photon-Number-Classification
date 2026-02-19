import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import poisson
import polars as pl
import os
import re
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

Average = [
    7.08211260e06,
    5.15056588e06,
    3.72436481e06,
    2.71572185e06,
    1.97472603e06,
    1.42408918e06,
    1.02232317e06,
    7.32125310e05,
    5.24090642e05,
    3.74998464e05,
    2.66916821e05,
    1.89078950e05,
    1.35700261e05,
    9.74077327e04,
    6.94691976e04,
    4.90518881e04,
    3.50311232e04,
    2.50723929e04,
    1.77261578e04,
    1.26385561e04,
    9.03448510e03,
    6.34776305e03,
    4.44228612e03,
    3.11561879e03,
    2.20105919e03,
    1.55472974e03,
    1.10119877e03,
    7.74547737e02,
    5.46214434e02,
    3.86814856e02,
    2.77270797e02,
    1.97954232e02,
    1.39667657e02,
    9.96796628e01,
    7.12159618e01,
    5.03875460e01,
    3.56871128e01,
    2.54771649e01,
    1.80622346e01,
    1.27380046e01,
    9.02367143e00,
    6.40704597e00,
    4.54269070e00,
    3.20272751e00,
    2.26309906e00,
]

dB = [
    7.0,
    7.5,
    8.0,
    8.5,
    9.0,
    9.5,
    10.0,
    10.5,
    11.0,
    11.5,
    12.0,
    12.5,
    13.0,
    13.5,
    14.0,
    14.5,
    15.0,
    15.5,
    16.0,
    16.5,
    17.0,
    17.5,
    18.0,
    18.5,
    19.0,
    19.5,
    20.0,
    20.5,
    21.0,
    21.5,
    22.0,
    22.5,
    23.0,
    23.5,
    24.0,
    24.5,
    25.0,
    25.5,
    26.0,
    26.5,
    27.0,
    27.5,
    28.0,
    28.5,
    29.0,
]


def stand(X: np.array):
    """

    Standardize an array

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    norm : ndarray

    """
    return (X - X.mean()) / X.std()


def find_repo_root(marker='myst.yml'):
    """
    Use an anchor to determine repo root 
    for passing absolute path
    """
    path = os.path.abspath('')
    while True:
        if os.path.exists(os.path.join(path, marker)):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError(f"Could not find {marker}")
        path = parent

def dataset_dat(
    weights,
    path_data: str,
    path_random_index: str,
    signal_size: int = 8192,
    interval: list = [0, 270],
    n_photon_number: int = 100,
    standardize: bool = False,
    plot_traces: bool = False,
    plot_expected: bool = False,
    SKIP: int = 1,
):

    init_size = len(weights)
    weights = np.array([0] * (len(Average) - init_size) + weights)
    weights = weights / weights.max()
    dB_ = [str(i) for i in dB]

    print('==============================================================>>>> DEBUG DIRS')
    print(path_data)
    print(path_random_index)
    os.listdir('/home/jovyan')
    os.listdir('/home/jovyan/data')
    # os.listdir(path_data)
    # os.listdir(f"{path_data}/data")
    # os.listdir(f"{path_data}/data/PIEdataRaw1")
    # os.listdir(f"{path_data}/data/PIEdataRaw2")

    files = []
    for root, _, filenames in os.walk(path_data):
        for f in filenames:
            if f.endswith(".det.daq"):
                files.append(os.path.join(root, f))

    files = sorted(files)
    print('==============================================================>>>> DEBUG files')
    print(files)

    # filename example:
    # 2010-08-05_1616_TES_A2_tomo_saturation_1kHz__1550.0nm_att17.0+17.0+17.0dB-bias1.000V-thr00001.01.det.daq
    re_db = re.compile(r"att\s*(?:\s*[\d\.]+\s*\+\s*)*(\d+\.?\d*)\s*dB", re.IGNORECASE)
    re_last_int = re.compile(r"(\d+)(?=\.det)")

    X_test, X_train = [], []
    X_dB_test, X_dB_train = [], []

    for file_ in files:
        fname = os.path.basename(file_)

        m_db = re_db.search(fname)
        if not m_db:
            print(f"[WARNING] Could not extract dB from: {fname}")
            continue

        db_string = m_db.group(1).replace(" ", "")
        if db_string not in dB_:
            print(f"[WARNING] dB value '{db_string}' not in dB list. File: {fname}")
            continue

        w = int(weights[dB_.index(db_string)] * 1024)
        if w <= 1:
            continue

        m_last = re_last_int.search(fname)
        if not m_last:
            print(f"[WARNING] Could not extract last int from: {fname}")
            continue

        last_int = int(m_last.group(1))

        print('==============================================================>>>> DEBUG')
        print(last_int)

        try:
            raw = np.fromfile(file_, dtype=np.float16)
            raw = raw.reshape(-1, signal_size)
            raw = raw[:w, interval[0] : interval[1]]
        except Exception as e:
            print(f"[ERROR] Failed reading {file_}: {e}")
            continue

        if last_int < 11:  # TRAIN
            X_train.append(raw[::SKIP])
            X_dB_train.append(np.full(w, db_string))
        else:  # TEST
            X_test.append(raw)
            X_dB_test.append(np.full(w, db_string))

    X_test = -1 * np.concatenate(X_test).astype(float)
    X_train = -1 * np.concatenate(X_train).astype(float)
    X_dB_test = np.concatenate(X_dB_test)
    X_dB_train = np.concatenate(X_dB_train)

    if standardize:
        data_train = stand(X_train)
        data_test = stand(X_test)
    else:
        data_train = X_train
        data_test = X_test

    if path_random_index is not None:
        random_index = np.load(path_random_index)
        data_train = data_train[random_index]
        data_test = data_test[random_index]
        X_dB_train = X_dB_train[random_index]
        X_dB_test = X_dB_test[random_index]

    expected_prob = np.zeros(n_photon_number)
    n_arr = np.arange(n_photon_number)

    for avg_, amp_ in zip(Average, weights):
        expected_prob += amp_ * poisson(mu=avg_).pmf(n_arr)

    expected_prob /= expected_prob.sum()

    if plot_expected:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(6, 3), dpi=100)
            plt.bar(n_arr, expected_prob, alpha=0.5, zorder=2)
            plt.xlabel("Photon number")
            plt.ylabel("Probability")
            plt.show()

    if plot_traces:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(6, 3), dpi=100)
            plt.plot(data_train[::10].T, alpha=0.05, linewidth=1)
            plt.plot(data_test[::10].T, alpha=0.05, linewidth=1)
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()

    return data_train, data_test, expected_prob, X_dB_train, X_dB_test


import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from typing import Union


def dataset_csv(
    path_data: str,
    files: Union[list, None] = None,
    plot_traces: bool = False,
    SKIP: int = 1,
    extensions=(".csv"),
):
    """
    Load CSV-like files from a folder (recursively), convert to arrays,
    apply preprocessing, and split train/test.

    - Searches all subdirectories.
    - Keeps only files matching allowed extensions.
    """
    if files is None:
        discovered_files = []
        for root, _, filenames in os.walk(path_data):
            for f in filenames:
                if extensions is None or f.lower().endswith(extensions):
                    discovered_files.append(os.path.join(root, f))
        files = discovered_files
    else:
        files = [os.path.join(path_data, f) for f in files]

    if len(files) == 0:
        raise RuntimeError(
            f"No files found in '{path_data}' " f"with extensions {extensions}"
        )

    files = sorted(files)
    print(f"[INFO] Found {len(files)} files.")

    data = []

    for file_ in files:
        fname = os.path.basename(file_)

        if len(fname) <= 15:
            continue

        try:
            df = pl.read_csv(file_, has_header=False, separator=",")
        except Exception as e:
            print(f"[WARNING] Could not read {file_}: {e}")
            continue

        arr = df.to_numpy().astype(np.float16)

        processed = (arr[:, ::3] - arr[:, :10].mean())[::SKIP]

        data.append(processed)

    if len(data) == 0:
        raise RuntimeError(
            "No usable CSV-like files were loaded — check filenames or folder structure."
        )

    data = np.concatenate(data, axis=0)
    data_train = data[::2]
    data_test = data[1::2]

    if plot_traces:
        with plt.style.context("seaborn-v0_8"):
            plt.figure(figsize=(6, 3), dpi=100)
            plt.plot(data_train[::10].T, alpha=0.05, linewidth=1)
            plt.plot(data_test[::10].T, alpha=0.05, linewidth=1)
            plt.xlabel("Time (a.u.)")
            plt.ylabel("Voltage (a.u.)")
            plt.show()

    return data_train, data_test
