"""
author : Anantha Rao
date : 19 Aug, 2021
details : This file contains the bulk of the classes and methods implemented
for GSoC 2021 project titled "Decoding quantum states through
Nuclear Magnetic Resonance" for the ML4SCI organization.

More details : https://summerofcode.withgoogle.com/projects/#6588988095201280

"""

import pandas as pd
import os
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.patches as mpatches
import matplotlib as mpl


class Dataset:
    """
    The Dataset class is used to represent a Dataset object which is the output
    of an insilico NMR spin-echo experiment. The files the directory must
    follow a particular naming convention. We follow this naming convention
    to read data files and pre-process them for further downstream propcessing.

    Attributes:
    ----------
    data_directory_path : str
        A string that specifies the full/relative path of the directory
        containing the simulation files and results.

    Methods:
    --------
    load_data()
        Loads the rawdata file by combining the real and imaginary parts of the
        echo curves and outputs the absolute value of magentization as a
        function of time as a pandas DataFrame object.

    get_window()
        A static method that is used to obtain a particular time-window
        within the entire timeframe of the experiment.

    load_params()
        Loads the parameters used for the experiment and
        returns a pandas DataFrame object.

    load_wlist()
        Loads the kernel_integral values for each simulaton and
        returns a pandas DataFrame object.

    get_yclasses()
        Loads the predictors ("αx","αz". "len_scale") useful for
        regression analysis

    """

    def __init__(self, data_directory_path: str):
        self.datadirectory = data_directory_path

    def load_data(self) -> np.ndarray:
        """Given the path of the directory containing the dataset,
        the respective Magnetization curves timeseris data is
        loaded and returned"""
        readfile = lambda x: pd.read_csv(
            os.path.join(self.datadirectory, x), delimiter=" ", header=None
        )
        echos_i, echos_r = readfile("echos_i"), readfile("echos_r")
        print("Finished loading rawdata into numpy array")
        return np.abs(echos_i.values + 1j * echos_r.values)

    def load_params(self) -> pd.DataFrame:
        """Given the directory path, loads the input parameter files
        for the simulations"""

        cols = "αx αy αz ξ pow Γ3 stencil_type s p d pulse90 pulse180".split()
        readfile = lambda x: pd.read_csv(
            os.path.join(self.datadirectory, x),
            delimiter=" ",
            header=None,
            dtype=np.float32,
            names=cols,
        )
        print("Finished loading parameters file")
        return readfile("echo_params.txt")

    def load_wlist(self) -> pd.DataFrame:
        """Given the path of the directory containing the simulation files,
        load the kernel-integrals file aka "w_list.txt" and return
        a Data frame"""
        print("Finished loading kernel-integrals file.")
        return pd.read_csv(
            os.path.join(self.datadirectory, "w_list.txt"),
            header=None,
            dtype=np.float64,
        )

    def get_yclasses(self) -> pd.DataFrame:
        """Given the parameters data frame and the kernel integrals data frame,
        here we compute the parameters to be predicted for regression,
        namely αx, αz, len_scale (ie sqrt(w_list/(2*αx+αz)))
        Returns: y_classes data frame"""
        params, ker_integrals = self.load_params(), self.load_wlist()
        y_classes = params[["αx", "αz"]].copy()
        y_classes["w_list"] = ker_integrals.values

        def get_len_scale(ax, az, w_list):
            return np.sqrt((w_list) / (2 * ax + az))

        y_classes["len_scale"] = y_classes.apply(
            lambda row: get_len_scale(row["αx"], row["αz"], row["w_list"]), axis=1
        )
        y_classes.drop("w_list", inplace=True, axis=1)
        return y_classes

    @staticmethod
    def get_window(
        data: np.ndarray,
        center_ratio: float,
        width: float,
        rescale: bool = True,
        as_df: bool = True,
    ):
        """Returns a subset of the given array with only
        those data points between [center - width , center + width]
        for all rows/examples"""
        start = int((center_ratio) * data.shape[1])
        print("The Echo pulse occurs at timestep:", start)
        output = data[:, start - width : start + width]
        if rescale and not as_df:
            return output / np.max(output, axis=1, keepdims=True), start
        elif rescale and as_df:
            return (
                pd.DataFrame(
                    output / np.max(output, axis=1, keepdims=True),
                    columns=[f"feat_{i}" for i in range(len(output[0]))],
                ),
                start,
            )
        elif not rescale and as_df:
            return (
                pd.DataFrame(
                    output, columns=[f"feat_{i}" for i in range(len(output[0]))]
                ),
                start,
            )
        else:
            return output, start


class PolynomialFeatures:
    """
     The PolynomialFeatuers class represents a PolynomialFeatures object that
     performs multiscale polynomial regression on the 2D numpy array and
     returns the value of the coefficients of the fitted polynomials.

     Attributes
     ----------
     n_splits: list
         A list of integers that specifies that number of equal partitions
     the dataset has to be bifurcated into. The number of elements in
     `order_fits` must be equal to the number of elements in `n_splits`

     order_fits: list
         A list of integers that specifies the order of the polynomial to-be fit
         in each partition of the 2D array. The number of elements in
         `order_fits` must be equal to the number of elements in `n_splits`

     Methods
     -------
     extract(dataset: np.ndarray, as_df: bool = True) -> np.ndarray:
         Extracts the polynomial features for each n_split and order_fit
         Returns a dataframe object if `as_df` is True


    poly_featextract_todf(output_data: np.ndarray) -> pd.DataFrame:
        If the output of `extract` is chosen to a pandas DataFrame object
        then this method is used to convert the numpy array with
         relevant column names for each co-efficient.

    split_and_fit(data: np.ndarray, n_split: int, order_fit: int) -> np.ndarray
        A static method that takes a 1D numpy array as input, splits it into
        `n_split` equal halves and fits a polynomial of order `order_fit`
        on each partition. Returns a numpy array with all the coefficients
        of all the polynomials in each partition

    precompute_output(dataset: np.ndarray, n_splits: list,
     order_fits: list) -> np.ndarray:
        A static method that precomputes the output dataset for polynomial
        feature extraction and stores it in memory.
        Is called later by the `PolynomialFeatures.extract` method."""

    def __init__(self, n_splits: list, order_fits: list):
        self.n_splits = n_splits
        self.order_fits = order_fits

    def extract(self, dataset: np.ndarray, as_df: bool = True) -> np.ndarray:
        """Computes the polynomial-fit features for each example in the dataset
        Input:
            dataset: 2D array of shape mxn where m is the number of examples and
                    n is the no of features
            n_splits: The number of equal splits features of the dataset
            order_fits: Order of the polynomial to be fit for each split in n_splits

        Returns:
            2D array with all the polyfit features for the dataset
        """
        if len(self.order_fits) != len(self.n_splits):
            raise ValueError(
                "n_splits and order_fits are not equal."
                + "Please provide the order of polynomial for each split"
            )

        # Pre-compute the output_dataset for faster execution
        output_data, split_cumsum = PolynomialFeatures.precompute_output(
            dataset, self.n_splits, self.order_fits
        )

        for id_row, row in enumerate(dataset):
            for id_split, split_fit in enumerate(zip(self.n_splits, self.order_fits)):
                output_data[
                    id_row, split_cumsum[id_split] : split_cumsum[id_split + 1]
                ] = PolynomialFeatures.split_and_fit(
                    data=row, n_split=split_fit[0], order_fit=split_fit[1]
                )
        if as_df:
            return PolynomialFeatures.poly_featextract_todf(self, output_data)
        else:
            return output_data

    def poly_featextract_todf(self, output_data: np.ndarray) -> pd.DataFrame:
        """Given the output of PolynomialFeatures.extract(),
        return a Dataframe with labelled columns for each coefficient"""

        cols = []
        polyfit_features = [f"{i}_{j}" for i in self.n_splits for j in range(1, i + 1)]
        for i, j in zip(self.n_splits, self.order_fits):
            for k in polyfit_features:
                if int(k.split("_")[0]) == i:
                    cols.extend([f"{k}_{l}" for l in np.arange(j, -1, -1)])
        return pd.DataFrame(output_data, columns=cols)

    @staticmethod
    def split_and_fit(data: np.ndarray, n_split: int, order_fit: int) -> np.ndarray:
        """Given the input numpy 1D array, split it into $(n_split)$ equal halves
        and perform a polynomial fit of order $(order_fit)$ on each half.
        Returns: A new 1D array of size n_split*(order_fit+1)
        containing the coefficients of the fit for each split"""

        splitted_data = np.array(np.split(data, n_split))
        x_axis = np.arange(len(splitted_data[0])) - len(splitted_data[0]) // 2
        output_array = np.zeros(n_split * (order_fit + 1))

        for counter, row in enumerate(splitted_data):
            output_array[
                counter * (order_fit + 1) : counter * (order_fit + 1) + order_fit + 1
            ] = np.polyfit(x=x_axis, y=row, deg=order_fit)

        return output_array

    @staticmethod
    def precompute_output(
        dataset: np.ndarray, n_splits: list, order_fits: list
    ) -> tuple:
        """Precomputes the output dataset for polynomial feature extraction"""

        nelements_ = [i * (j + 1) for i, j in zip(n_splits, order_fits)]
        nelements_ = [0] + nelements_
        nelems_cumsum = np.cumsum(nelements_)
        output_data = np.zeros((dataset.shape[0], np.sum(nelements_)))
        return output_data, nelems_cumsum


class FeatureImportancePlot:
    """
    A FeatureImportancePlot class represents a matplotlib axes plot that
    plots the features from polynomial feature extraction based on the
    feature importances from a random forest model.

    Attributes:
    ----------
    length_tseries: int
        The length of the time-series that the polynomial feaures works on

    n_splits: list
        A list of integers that specifies that number of equal partitions
        the dataset has to be bifurcated into.

    kind: str ("timeseries"  or "polyfeatures" )
        The kind of dataset used by the random forest model for learning
        the classification/regression parameters.

    Methods:
    --------
    get_intervals() -> np.ndarray
        Returns the end-points or intervals of a n-fold
        (FeatureImportancePlot.n_splits) partition of a dataset

    get_fi(model, X_data: pd.DataFrame) -> pd.DataFrame
        Returns the feature importance dataset from the given ML model
        and the X dataset sorted based on feature importance


    fi_df2plot(fi_df: pd.DataFrame, ntop: int, ax) -> mpl.axes
        Plots the top {ntop} features from the feature importance data
        on the given matplotlib 'ax' based on important features dataframe

    plot_feature()
        A static method that plots a line-bar for each feature from
        polynomial feature extraction"""

    def __init__(self, length_tseries: int, n_splits: list, kind: str):
        self.length_tseries = length_tseries
        self.n_splits = n_splits
        self.kind = kind
        self.intervals = FeatureImportancePlot.get_intervals(self)

    def get_intervals(self) -> np.ndarray:
        """Gives the time-interval splits for the number of splits given
        Returns : Dictionary with the n_split as the key and the time-interval
        stamps as values"""
        split_index = [
            np.linspace(0, self.length_tseries, i + 1).astype(np.int32)
            for i in self.n_splits
        ]
        intervals = {}
        for idx, interval in zip(self.n_splits, split_index):
            intervals[idx] = interval
        return intervals

    def get_fi(self, model, X_data: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance Data Frame from fitted model"""

        if self.kind == "timeseries":
            fi = pd.DataFrame(
                np.array([X_data.columns.tolist(), model.feature_importances_]).T,
                columns=["feature", "fi"],
            )
            fi["fi"] = pd.to_numeric(fi["fi"])
            return fi.sort_values("fi", ascending=False)

        if self.kind == "polyfeatures":
            fi = pd.DataFrame(
                np.array([X_data.columns.tolist(), model.feature_importances_]).T,
                columns=["feature", "fi"],
            )
            fi["fi"] = pd.to_numeric(fi["fi"], downcast="float")
            fi["order"] = fi["feature"].apply(lambda x: x.split("_")[-1])
            fi["color"] = fi["order"].replace({"0": "r", "1": "b", "2": "g", "3": "y"})

            return fi.sort_values("fi", ascending=False)

    def fi_df2plot(self, fi_df: pd.DataFrame, ntop: int, ax):
        """Plots the top {ntop} features from the feature importance data
        based on important features dataframe"""

        feats = fi_df.head(ntop)["feature"].values.tolist()[::-1]
        feats = list(map(lambda x: x.split("_"), feats))

        ax.set(xlabel="Time")
        for idx, feature in enumerate(feats):
            FeatureImportancePlot.plot_feature(self.intervals, ax, feature, idx + 1)

        # Setting up yticks
        ax.set_yticks(np.arange(1, ntop + 1))
        ax.set_yticklabels(np.arange(1, ntop + 1)[::-1])

        # Setting up legend
        color_arr = ["#c70a39", "#698bfa", "#90e388", "y"]

        red_patch = mpatches.Patch(color=color_arr[0], label=r"$x^0$")
        bl_patch = mpatches.Patch(color=color_arr[1], label=r"$x^1$")
        green_patch = mpatches.Patch(color=color_arr[2], label=r"$x^2$")
        yl_patch = mpatches.Patch(color=color_arr[3], label=r"$x^3$")
        ax.legend(
            handles=[red_patch, bl_patch, green_patch, yl_patch],
            bbox_to_anchor=[1.01, 0.95],
        )
        return ax

    @staticmethod
    def plot_feature(intervals, axes, feature: tuple, plot_index: int):
        """Given an axes object, plots a line-bar for the given feature
        of the form (10,5,2) where:
            10 : order of partition
            5 : the bin of interest in the partition
            2 : order of the polynomial in the given bin
            We choose a color for the order of the polynomial (0th,1st,2nd,3rd)
        Returns : an axes object
        """
        color_arr = ["#c70a39", "#698bfa", "#90e388", "y"]

        axes.plot(
            intervals[int(feature[0])][int(feature[1]) - 1 : int(feature[1]) + 1],
            [plot_index, plot_index],
            lw=10,
            color=color_arr[int(feature[2])],
            label=f"$x^{int(feature[2])}$",
        )
        end_time = [max(intervals[key]) for key in intervals.keys()][0]
        axes.plot(
            [0, end_time], [plot_index + 0.5, plot_index + 0.5], color=(0, 0, 0, 0.2)
        )
        axes.legend()


class FourierTransform:
    """
    The FourierTranform class represents a fourier-transform of a
    particular time-window of the Dataset object. The selected time-window is
    smoothened with a suitable sigmoid fucntion that vanished at the boundries,
    then we pad 2^n zeros on both ends of the time-window before
    performing the fast-fourier transform.

    Attributes
    ---------
    zero_padding: int
        The number of zeros (2^{`zero_padding`}) to be added to the end of
        the smoothened time-window before performing the fourier transform.

    Methods:
    --------
    get_fft(timewindow: np.ndarray, tenvelope: np.ndarray,
        Fs: np.ndarray) -> tuple(np.ndarray, np.ndarray):
        For a 1D timewindow, apply a smoothening sigmoid function and
        return the absolute value of the imaginary part of the
        fourier tranform and the new frequency axis

    transform(ts_data: np.ndarray) -> tuple(np.ndarray, np.ndarray) :
        Performs `FourierTransform.get_fft` for each row in the 2D ts_data and
        returns a tuple containing the new freqeucny axis and
        imaginary part of the fourier tranform.

    get_tenvelope(t_max: float, N: int) -> tuple(np.ndarray, np.ndarray):
        A static method that returns the smoothening time-window that is
        applied to all time-series under the
        `FourierTransform.get_fft()` method

    """

    def __init__(self, zero_padding: int):
        self.zero_padding = zero_padding

    def get_fft(
        self, timewindow: np.ndarray, tenvelope: np.ndarray, Fs: np.ndarray
    ) -> tuple:
        """Given a 1D timewindow, apply a smoothening sigmoid function and
        return the absolute value of the imaginary part of the
        fourier tranform adn the new frequency axis"""
        signal = tenvelope * timewindow
        fullsignal = np.pad(signal, (2 ** self.zero_padding), "constant")
        fullsignal_size = fullsignal.shape[0]
        delta_f = Fs / fullsignal_size
        freq_ax = (
            np.linspace(0, (fullsignal_size - 1) * delta_f, int(fullsignal_size))
            - (fullsignal_size - 1) * delta_f / 2
        )

        freq_signal = fftshift(fft(fftshift(fullsignal)))

        return abs(freq_signal.imag), freq_ax

    def transform(self, ts_data: np.ndarray) -> tuple:
        """Returns the 1D Fourier transform of a 2D array (row-wise)
        after applying a smoothening sigmoid function and zero-padding"""
        ft_dataframe = np.zeros(
            (ts_data.shape[0], int(ts_data.shape[1] + pow(2, self.zero_padding + 1)))
        )
        fx_dataframe = ft_dataframe.copy()
        N_samples = ts_data.shape[1]
        t_envelope, Fs = FourierTransform.get_tenvelope(
            t_max=(N_samples / 948) * N_samples, N=N_samples
        )
        for row_no in range(len(ft_dataframe)):
            ft_dataframe[row_no, :], fx_dataframe[row_no, :] = FourierTransform.get_fft(
                self, ts_data[row_no, :], t_envelope, Fs=Fs
            )
        return ft_dataframe, fx_dataframe

    @staticmethod
    def get_tenvelope(t_max: float, N: int) -> tuple:
        """Given the length of the time-window (t_max) and its sampling rate (N),
        return a smoothening time-window and the sampling frequency as a tuple
        """
        t = np.linspace(0, t_max, N)
        t_envelope = 1 / (1 + np.exp(-0.35 * (t - 35)))
        t_envelope *= t_envelope[::-1]
        Fs = 1 / (t[1] - t[0])
        return t_envelope, Fs
