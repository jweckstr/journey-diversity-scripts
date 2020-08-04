import os
from itertools import combinations_with_replacement

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

from gtfspy.util import timeit, makedirs
from research.route_diversity.diversity_settings import FIGS_DIRECTORY
from research.route_diversity.measures_and_colormaps import get_colormap

from research.route_diversity.rd_utils import round_sigfigs, drop_nans, flatten_2d_array, \
    get_differences_between_dataframes, stops_within_buffer, filter_stops_spatially


class JourneyAnalyzeComparer:
    def __init__(self, route_analyzers, targets):
        self.route_analyzers = route_analyzers
        self.targets = targets
        self.city = route_analyzers[0].get_name_year()[0]
        self._podas = None

    def get_years(self):
        return [y.get_name_year()[1] for y in self.route_analyzers]

    def collect_pickles(self, route_analyzer):
        temp_dfs = []
        generator = route_analyzer.diversity_pickle_generator(self.targets)
        pickles = [x for x in generator]
        if len(self.targets) - len(pickles) > 0:
            print("WARNING! pickles for {} targets missing".format(len(self.targets) - len(pickles)))
        else:
            print("All pickles available")
        for temp_df in pickles:
            temp_dfs.append(temp_df)
        df = pd.concat(temp_dfs, ignore_index=True, sort=True)
        df.set_index(["from_stop_I", "to_stop_I"], inplace=True)
        return df

    def collect_pickles_using_buffer(self, route_analyzer, **kwargs):
        df = self.collect_pickles(route_analyzer)
        if "geometry" in kwargs:
            df.reset_index(inplace=True)
            df = filter_stops_spatially(df, self.route_analyzers[0].gtfs, **kwargs)
            df.set_index(["from_stop_I", "to_stop_I"], inplace=True)
        return df

    @timeit
    def get_podas(self, **kwargs):
        if self._podas:
            return self._podas
        else:
            dfs = {}
            for route_analyzer in self.route_analyzers:
                df = self.collect_pickles_using_buffer(route_analyzer, **kwargs)
                dfs[route_analyzer.get_name_year()[1]] = df

            dfs, names = get_differences_between_dataframes(dfs)
            podas = [PairwiseODAnalyzer(df, namepair) for df, namepair in zip(dfs, names)]
            return podas

    @timeit
    def stress_test(self, target, **kwargs):
        """
        What to test:
        - How long it takes to process 100 - 1000 targets
        - how much hd and memory space is needed for processing and storage
        - how long it takes to match the Dataframes from two sources

        Pseudo code:
        Process one routing pickle to performance measure dataframe
        create copies with altered results
        add copies to master dataframe
        process the other route_analyzer in a similar way
        match results and calculate difference
        store pickle
        perform searches
        :param city:
        :param target:
        :param kwargs:
        :return:
        """
        dfs = {}
        prev_targets = None
        print("target:", target)
        for route_analyzer in self.route_analyzers:
            store_dict = route_analyzer.find_pickle(target)
            df = route_analyzer.calculate_diversity(**store_dict)
            temp_dfs = []
            for i in range(0, 100):
                temp_df = df.copy(deep=True)
                temp_df["to_stop_I"] = i
                temp_dfs.append(temp_df)
                # route_analyzer.append_to_performance_measure_df(temp_df)

            df = pd.concat(temp_dfs, ignore_index=True)
            print(df)
            targets = list(df["to_stop_I"].unique())
            if prev_targets:
                assert targets == prev_targets
            prev_targets = targets
            df.set_index(["from_stop_I", "to_stop_I"], inplace=True)
            dfs[route_analyzer.get_name_year()[1]] = df

        return get_differences_between_dataframes(dfs)

    def plot_od_heatmap(self, measure_x, measure_y, x_lims, y_lims, **kwargs):
        """
        Plots a heatmap on a o-d basis
        :param measure_y: measure on y axis
        :param measure_x: measure on x axis
        :return:
        """

        for poda in self.get_podas(**kwargs):
            density = kwargs.pop("density", True)
            df = poda.get_df_without_infs_and_nans()
            name_b = poda.name1
            name_a = poda.name2

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

            ranges = [x_lims, y_lims]   # [[xmin, xmax], [ymin, ymax]]
            bins = (15, 15)
            a, xedges, yedges = np.histogram2d(df[measure_x + "|" + name_a], df[measure_y + "|" + name_a],
                                               bins=bins, range=ranges, density=density)
            x_ticks = np.linspace(ranges[0][0], ranges[0][1], 5)
            y_ticks = np.linspace(ranges[1][0], ranges[1][1], 5)
            b, _, _ = np.histogram2d(df[measure_x + "|" + name_b], df[measure_y + "|" + name_b],
                                     bins=bins, range=ranges, density=density)
            c = b - a
            vmax = round_sigfigs(np.amax(a), 1)

            cmap, norm = get_colormap(data=flatten_2d_array(a), percentiles=(0, 100))
            plot = ax1.imshow(a.T, cmap=cmap, norm=norm, aspect='auto', extent=[x_ticks.min(), x_ticks.max(), y_ticks.max(), y_ticks.min(), ])
            plt.colorbar(plot, ax=ax1)
            plot = ax2.imshow(b.T, cmap=cmap, norm=norm, aspect='auto', extent=[x_ticks.min(), x_ticks.max(), y_ticks.max(), y_ticks.min(), ])
            plt.colorbar(plot, ax=ax2)

            cmap, norm = get_colormap(data=flatten_2d_array(c), percentiles=(0, 100))
            plot = ax3.imshow(c.T, cmap=cmap, norm=norm, aspect='auto', extent=[x_ticks.min(), x_ticks.max(), y_ticks.max(), y_ticks.min(), ])
            plt.colorbar(plot, ax=ax3)

            ax1.set_ylabel(measure_y)
            ax2.set_xlabel(measure_x)

            ax1.set_title(name_a)
            ax2.set_title(name_b)
            ax3.set_title("difference")
            # plt.scatter(df["number_of_most_common_journey_variant_2017"], df["time_weighted_simpson_2017"])  # , bins=(30, 30), cmap=plt.cm.Reds)
            # plt.xlim([0, 12])
            # plt.ylim([0, 0.5])
            # g = pairplot(df.dropna(axis=0), plot_kws={"s": 6})
            #plt.show()
            fig_format = kwargs.get("fig_format", "png")
            folder = kwargs.get("folder", "")
            fname = kwargs.get("fname", "heatmap" + "_" + str(measure_y) + "_|_" + str(measure_x) + "_" + self.city
                               + "." + fig_format)
            plt.savefig(os.path.join(makedirs(folder), fname), format=fig_format, dpi=300)

    def plot_difference_between_feeds(self, target=None, **kwargs):
        ra_plots = self.route_analyzers[0]
        for df in self.get_aggregated_feed_difference_generator(**kwargs):
            if target:
                ra_plots.plot_multiple_measures_on_maps(df, target=target, figs_directory=FIGS_DIRECTORY,
                                                        fname_suffix="_" + str(target))
            else:
                ra_plots.plot_multiple_measures_on_maps(df, figs_directory=FIGS_DIRECTORY, **kwargs)
                #ra_plots.plot_sample_stops(targets)

    def get_aggregated_feed_difference_generator(self, **kwargs):
        ra_plots = self.route_analyzers[0]
        for poda in self.get_podas(**kwargs):
            poda.get_df_without_infs_and_nans()
            df = poda.aggragate_dataframe(agg=kwargs.pop("agg"))
            df = ra_plots.gtfs.add_coordinates_to_df(df)
            yield df

    def get_feed_difference_generator(self, walking_distance=None, threshold_measure=None, threshold=None, **kwargs):
        for poda in self.get_podas(**kwargs):
            df = poda.get_df_without_infs_and_nans()
            init_len = len(df.index)
            if threshold_measure and threshold:
                df = poda.apply_pairwise_threshold(measure=threshold_measure, threshold=threshold)
            if walking_distance:
                df = df.loc[self.remove_walking_ods(df, walking_distance=walking_distance)].copy()
            len_after_clean = len(df.index)
            print("Attention! Of {init} rows, {removed} rows not fitting threshold were removed"
                  .format(init=init_len, removed=init_len - len_after_clean))
            yield df

    def thresholded_counts(self, condition_tuples, feed, **kwargs):
        df = DataFrame()
        condition_strs = []
        for poda in self.get_podas(**kwargs):
            for i in self.replacements(condition_tuples):
                condition_str = "+".join([x[0]+x[1]+str(x[2]) for x in i])
                condition_strs.append(condition_str)
                conditions = [(x[0]+"|"+feed, x[1], x[2]) for x in i]
                print(conditions)
                cur_df = poda.add_count_by_condition(conditions, col_name=condition_str)
                if df.empty:
                    df = cur_df
                else:
                    df = df.merge(cur_df, left_index=True, right_index=True)
        return df, condition_strs

    @staticmethod
    def replacements(condition_tuples):
        def replace_sign(sign):
            if sign == "==":
                return "!="
            elif sign == "<=":
                return ">"
            elif sign == "<":
                return ">="
            elif sign == ">=":
                return "<"
            elif sign == ">":
                return "<="
            else:
                return

        l_conditions = len(condition_tuples)
        for special_case in [True, False]:
            if special_case:
                yield [(x[0], replace_sign(x[1]), x[2]) for x in condition_tuples]
            else:
                for i in combinations_with_replacement(condition_tuples, l_conditions):
                    negatives = [(x[0], replace_sign(x[1]), x[2]) for x in condition_tuples if x not in i]
                    positives = [x for x in condition_tuples if x in i]
                    yield negatives + positives

    def remove_walking_ods(self, df, walking_distance=0):
        ra_plots = self.route_analyzers[0]
        return df.apply(lambda x: not ra_plots.gtfs.get_stop_walk_distance(x.from_stop_I, x.to_stop_I) or
                                  ra_plots.gtfs.get_stop_walk_distance(x.from_stop_I, x.to_stop_I) > walking_distance,
                        axis=1)

    def plot_threshold_ods(self, measure_sets, threshold_sets=None, **kwargs):
        if not isinstance(measure_sets[0], list):
            measure_sets = [measure_sets]
            threshold_sets = [threshold_sets]
        ra_plots = self.route_analyzers[0]
        for poda in self.get_podas(**kwargs):
            for i, measures in enumerate(measure_sets):
                dfs = {}
                for name in poda.names():
                    poda.reset_df()
                    measures_with_year = [measure + "|" + name for measure in measures]
                    dfs[name] = poda.apply_thresholds_and_count(measures_with_year, threshold_sets[i])

                dfs, names = get_differences_between_dataframes(dfs)
                for df, name in zip(dfs, names):
                    df = ra_plots.gtfs.add_coordinates_to_df(df)

                    ra_plots.plot_multiple_measures_on_maps(df, figs_directory=FIGS_DIRECTORY, cmap_using_data=True,
                                                            percentiles=(2, 98))


class PairwiseODAnalyzer:
    def __init__(self, df, names):
        self.df = df
        self._orig_df = df
        (self.name1, self.name2) = names

    def names(self):
        return self.name1, self.name2

    def reset_df(self):
        self.df = self._orig_df

    def get_df_without_infs_and_nans(self):
        self.df = drop_nans(self.df)
        return self.df

    def apply_thresholds_and_count(self, measures, thresholds, col_name="count"):
        df = self.df

        all_stops_df = DataFrame({"stop_I": df["from_stop_I"].unique()})

        df = df.reset_index()
        df = self.apply_thresholds(df, measures, thresholds)

        df = drop_nans(df)
        df[col_name] = 1
        df = df[["from_stop_I", col_name]].groupby("from_stop_I").count()
        df = df.reset_index()
        df = df.rename(index=str, columns={'from_stop_I': 'stop_I'})
        df = all_stops_df.merge(df, how="left", left_on="stop_I", right_on="stop_I")
        df = df.fillna(value=0)
        df = df.set_index("stop_I")
        self.df = df
        return df

    def add_count_by_condition(self, condition_tuples, col_name="count"):
        df = self.df
        all_stops_df = DataFrame({"stop_I": df["from_stop_I"].unique()})

        df = df.reset_index()
        df = self.apply_conditions(df, condition_tuples)

        df = drop_nans(df)
        df[col_name] = 1
        df = df[["from_stop_I", col_name]].groupby("from_stop_I").count()
        df = df.reset_index()
        df = df.rename(index=str, columns={'from_stop_I': 'stop_I'})
        df = all_stops_df.merge(df, how="left", left_on="stop_I", right_on="stop_I")
        df = df.fillna(value=0)
        df = df.set_index("stop_I")
        return df

    def aggragate_dataframe(self, agg='mean'):
        df = self.df
        targets = set(df["to_stop_I"].unique())
        df = df.drop(["to_stop_I"], axis=1)
        df_grouped = df.groupby(["from_stop_I"]).agg([agg])
        df_grouped.columns = df_grouped.columns.map('|'.join)

        df = df_grouped.reset_index()
        df = df.rename(index=str, columns={'from_stop_I': 'stop_I'})
        self.df = df
        return df

    def apply_pairwise_threshold(self, measure, threshold):
        measures = [x for x in self.df.columns if measure in x and (self.name1 in x or self.name2 in x)]
        thresholds = [threshold for x in measures]
        return self.apply_thresholds(self.df, measures, thresholds)

    @staticmethod
    def apply_thresholds(df, measures, thresholds):
        cut_points = {k: v for k, v in zip(measures, thresholds)}

        conditions = [(df[k] <= v) for k, v in cut_points.items()]
        print("cut points:", cut_points)
        df = df[np.logical_and.reduce(conditions)]
        return df.copy()

    @staticmethod
    def apply_conditions(df, condition_tuples):
        """
        Applies a list of column based conditions on a Pandas DataFrame, returns a copy of the DataFrame
        :param df: Pandas DataFrame
        :param condition_tuples: list of tuples (column name in df, string; condition sign, string;
        threshold value, numeric)
        :return: Pandas DataFrame
        """
        conditions = []
        for m, c, v in condition_tuples:
            if c == "==":
                condition = (df[m] == v)
            elif c == "<=":
                condition = (df[m] <= v)
            elif c == "<":
                condition = (df[m] < v)
            elif c == ">=":
                condition = (df[m] >= v)
            elif c == ">":
                condition = (df[m] > v)
            else:
                return
            conditions.append(condition)

        print("cut points:", condition_tuples)
        df = df[np.logical_and.reduce(conditions)]
        return df.copy()
