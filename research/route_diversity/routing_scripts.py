import sys
import itertools
from collections import OrderedDict

import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from pandas import DataFrame

from gtfspy.smopy_plot_helper import *


from gtfspy.plots import plot_trip_counts_hourly
from research.route_diversity.journey_analyze_comparison import JourneyAnalyzeComparer

from research.route_diversity.journey_analyze_pipeline import *
from research.route_diversity.journey_analyze_plots import JourneyAnalyzePlots
from research.route_diversity.rd_utils import stop_sample, split_into_equal_length_parts, drop_nans, tidy_label
from research.route_diversity.measures_and_colormaps import *
from research.route_diversity.static_route_type_analyzer import RouteTypeAnalyzer, get_most_intense_centroid
from research.route_diversity.static_network_scripts import get_route_type_analyzer


def get_sample_stops(gtfs, **kwargs):
    return stop_sample(gtfs, **kwargs)


def run_multifeed_diversity(city, target, **kwargs):
    for feed in CITIES[city]:
        run_plot_diversity(feed, target, **kwargs)


def run_multifeed_stop2stop(origin, target, city):
    for feed in CITIES[city]:
        run_stop_to_stop(feed, origin, target)


def run_routing_slurm_batch(feed, slurm_array_i, slurm_array_length, **kwargs):
    assert (slurm_array_i < slurm_array_length)
    feed_dict = get_feed_dict(feed)
    print("feed: ", feed)
    sample_stops = get_sample_stops(feed_dict["gtfs"], **kwargs)
    parts = split_into_equal_length_parts(sample_stops, slurm_array_length)
    targets = parts[slurm_array_i]
    run_routing(feed, targets)


def run_routing(feed, targets):
    assert TRACK_ROUTE
    feed_dict = get_feed_dict(feed)
    print("feed: ", feed)
    ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
    ra_pipeline.loop_trough_targets_and_run_routing_with_route(targets, "")


def routing_to_njpa(feed, targets=None, slurm_array_i=None, slurm_array_length=None, **kwargs):
    assert TRACK_ROUTE
    assert (slurm_array_i is not None and slurm_array_length is not None) or targets is not None

    feed_dict = get_feed_dict(feed)
    print("feed: ", feed)
    if not targets:
        sample_stops = get_sample_stops(feed_dict["gtfs"], **kwargs)
        parts = split_into_equal_length_parts(sample_stops, slurm_array_length)
        targets = parts[slurm_array_i]
    targets = [int(x) for x in targets]
    ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
    ra_pipeline.run_everything_to_diversity(targets)


def plot_journeys_to_target(feed, target):
    assert TRACK_ROUTE
    feed_dict = get_feed_dict(feed)
    print("feed: ", feed, "target", target)
    ra_pipeline = JourneyAnalyzePlots(**feed_dict)
    ra_pipeline.plot_journeys_to_target(target)


def run_plot_diversity(feed, target, **kwargs):
    print(feed, "target:", target)
    feed_dict = get_feed_dict(feed)
    feed_dict.update(**kwargs)
    ra_pipeline = JourneyAnalyzePlots(**feed_dict)
    ra_pipeline.plot_multiple_measures_on_maps(target=target, figs_directory=FIGS_DIRECTORY)


def run_calculate_diversity(feed, target, **kwargs):

    print(feed, "target:", target)
    feed_dict = get_feed_dict(feed)
    feed_dict.update(**kwargs)
    ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
    ra_pipeline.calculate_diversity_for_target(target)


def run_stop_to_stop(feed, origin, target, **kwargs):
    origin_name = None
    target_name = None
    try:
        origin = int(origin)
    except ValueError:
        origin_name = origin
        origin = None
    try:
        target = int(target)
    except ValueError:
        target_name = target
        target = None

    feed_dict = get_feed_dict(feed)
    feed_dict.update(kwargs)
    ra_pipeline = JourneyAnalyzePlots(**feed_dict)
    ra_pipeline.stop_to_stop_routes(origin_id=origin, origin_name=origin_name, target_id=target,
                                    target_name=target_name, folder=os.path.join(MAPS_DIRECTORY, "s2s_tp"), **kwargs)


def run_generic(feed, **kwargs):

    print(feed)
    feed_dict = get_feed_dict(feed)
    plot_trip_counts_hourly(feed_dict["gtfs"], show=True)


def get_journey_analyze_comparer(feeds, **kwargs):
    feed_dicts = [get_feed_dict(feed) for feed in feeds]
    sample_stops = get_sample_stops(feed_dicts[0]["gtfs"], **kwargs)
    jac = JourneyAnalyzeComparer([JourneyAnalyzePlots(**feed_dict) for feed_dict in feed_dicts], sample_stops)
    return jac


def od_heatmap(feeds, **kwargs):

    print(feeds)
    jac = get_journey_analyze_comparer(feeds, **kwargs)
    measure1 = largest_headway_gap
    measure2 = number_of_most_common_journey_variant
    x_lims = MEASURE_PLOT_PARAMS[measure1]["lims"]
    y_lims = MEASURE_PLOT_PARAMS[measure2]["lims"]
    folder = os.path.join(FIGS_DIRECTORY, "heatmaps")
    jac.plot_od_heatmap(measure1, measure2, x_lims=x_lims, y_lims=y_lims, folder=folder)
    # TODO: what is the percentage of stops/places reached by each route?


def run_pair_plots(feeds, **kwargs):

    print(feeds)
    jac = get_journey_analyze_comparer(feeds, **kwargs)
    years = jac.get_years()
    hubs = [x.gtfs.calculate_hubs(200, x.day_start, 0 * 3600, 24 * 3600) for x in jac.route_analyzers]
    hubs = [hub[["stop_I", trips_in_area]] for hub in hubs]

    select_cols = [mean_temporal_distance, mean_trip_n_boardings, time_weighted_simpson, trips_in_area]
    for df in jac.get_feed_difference_generator():
        for hub, year in zip(hubs, years):
            trips_in_area_o = 'trips_in_area_o_{}'.format(year)
            trips_in_area_d = 'trips_in_area_d_{}'.format(year)
            prod_name = trips_in_area + "_prod_{}".format(year)

            df = df.merge(hub, how="left", left_on="from_stop_I", right_on="stop_I")
            df = df.rename(index=str, columns={'trips_in_area': trips_in_area_o})
            df = df.merge(hub, how="left", left_on="to_stop_I", right_on="stop_I")

            df = df.rename(index=str, columns={'trips_in_area': trips_in_area_d})
            df = df.fillna(value=0)

            df[prod_name] = df.apply(lambda row: 0 if row[trips_in_area_d]*row[trips_in_area_o] == 0 else
            np.log(row[trips_in_area_d]*row[trips_in_area_o]), axis=1)
            gc.collect()
        df = df.drop(['from_stop_I', 'to_stop_I'], axis=1)
        cols = [col for col in df.columns if ("|diff" in col or years[1] in col) and
                any([True for y in select_cols if y in col])]
        df = df[cols]
        fig = plt.figure()
        g = sns.PairGrid(df, diag_sharey=False)
        g.map_diag(plt.hist, bins=20)
        gc.collect()

        def pairgrid_heatmap(x, y, **kws):
            cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
            plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

        g.map_offdiag(pairgrid_heatmap, bins=20)
        plt.savefig(os.path.join(FIGS_DIRECTORY, jac.city + "_core_pair_plot.png"), format="png", dpi=300)


def plot_means_on_map(feeds, **kwargs):
    kwargs["agg"] = "mean"
    kwargs["fname_simple"] = True
    kwargs["separate_colorbar"] = True
    kwargs["title"] = False
    rac = get_journey_analyze_comparer(feeds, **kwargs)
    rac.plot_difference_between_feeds(**kwargs)


def plot_ods(feeds, **kwargs):
    rac = get_journey_analyze_comparer(feeds, **kwargs)
    measures = [time_weighted_simpson, number_of_most_common_journey_variant]
    thresholds = [0.4, 3]
    rac.plot_threshold_ods(measures, thresholds)


def plot_diff_distributions(feeds=None, buffer=50000, **kwargs):
    threshold_measure = kwargs.get("threshold_measure", None)
    threshold = kwargs.get("threshold", None)
    figs = {}
    for ci, city in enumerate(CITIES.keys(), start=0):
        print(city)
        subfeeds = CITIES[city]
        centroid_gdf = get_most_intense_centroid(subfeeds[0])

        if not feeds or set(subfeeds) <= set(feeds):
            rac = get_journey_analyze_comparer([subfeeds[1], subfeeds[0]], **kwargs)
            for df in rac.get_feed_difference_generator(threshold_measure=threshold_measure, threshold=threshold,
                                                        **{"cols": ["from_stop_I", "to_stop_I"],
                                                           "geometry": [centroid_gdf, centroid_gdf],
                                                           "buffer": [buffer, buffer]}):
                df["city"] = tidy_label(city, capitalize=True)
                for i, column in enumerate(df.columns):
                    if "diff" in column:
                        fig = figs.get(column)
                        if fig is not None:
                            plt.figure(fig.number)
                            ax = plt.gca()
                        else:
                            fig, ax = plt.subplots()
                        flierprops = dict(marker='o', markersize=1)

                        sns.violinplot(x="city", y=column, data=df, ax=ax, order=[tidy_label(city, capitalize=True) for city in CITIES.keys()], flierprops=flierprops,  # scale="area",
                                       linewidth=0.25)
                        figs[column] = fig

    for col, fig in figs.items():
        print("plot:", col)
        plt.figure(fig.number)
        plt.grid(True)
        plt.ylim(MEASURE_PLOT_PARAMS[col.replace(diff_suffix, "")]["diff_lims"])
        plt.xticks(rotation=20)
        plt.ylabel(tidy_label(MEASURE_ALIASES[col.replace(diff_suffix, "")]))
        plt.savefig(os.path.join(FIGS_DIRECTORY, col+"_box_distribution_{buffer}.png".format(
                buffer=str(buffer))), format="png", dpi=600)


def plot_split_distributions(feeds=None, buffer=50000, **kwargs):
    all_temp_dfs = []
    for ci, city in enumerate(CITIES.keys(), start=0):
        print(city)
        subfeeds = CITIES[city]

        if not feeds or set(subfeeds) <= set(feeds):
            rac = get_journey_analyze_comparer([subfeeds[0], subfeeds[1]], **kwargs)
            city_temp_dfs = []
            centroid_gdf = get_most_intense_centroid(subfeeds[0])

            for route_analyzer, year in zip(rac.route_analyzers, ["Before", "After"]):
                temp_df = rac.collect_pickles_using_buffer(route_analyzer, **{"cols": ["from_stop_I", "to_stop_I"],
                                                                              "geometry": [centroid_gdf, centroid_gdf],
                                                                              "buffer": [buffer, buffer]})
                temp_df["year"] = year
                temp_df["city"] = city
                temp_df = drop_nans(temp_df)
                city_temp_dfs.append(temp_df)

            intersect_index = list(set(city_temp_dfs[0].index.to_list()).intersection(
                set(city_temp_dfs[1].index.to_list())))

            city_temp_dfs = [x.loc[intersect_index] for x in city_temp_dfs]

            all_temp_dfs += city_temp_dfs
    print("concat")
    df = pd.concat(all_temp_dfs, ignore_index=True, sort=True)
    df["city"] = df.apply(lambda x: tidy_label(x.city, True), axis=1)
    order = [tidy_label(city, True) for city in CITIES.keys()]
    print("plotting")
    for column in df.columns:
        if column not in ["year", "city"]:
            fig, ax = plt.subplots()
            print("plot:", column)
            sns.violinplot(x="city", y=column, hue="year", split=True, data=df, ax=ax, order=order,
                           # scale="area",
                           inner='quartile',
                           scale_hue=True,
                           linewidth=0.25)
            plt.grid(True)
            plt.xticks(rotation=20)
            plt.ylim(MEASURE_PLOT_PARAMS[column]["lims"])
            plt.ylabel(tidy_label(MEASURE_ALIASES[column]))
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), loc=5)
            plt.savefig(os.path.join(FIGS_DIRECTORY, column+"_split_box_distribution_{buffer}.png".format(
                buffer=str(buffer))), format="png", dpi=600)


def confidence_intervals_using_bootstrap(feeds=None, buffer=50000, **kwargs):
    import bootstrapped.bootstrap as bs
    import bootstrapped.stats_functions as bs_stats
    all_temp_dicts = {}
    for ci, city in enumerate(CITIES.keys(), start=0):
        subfeeds = CITIES[city]

        rac = get_journey_analyze_comparer([subfeeds[0], subfeeds[1]], **kwargs)

        centroid_gdf = get_most_intense_centroid(subfeeds[0])

        for route_analyzer, year, subfeed in zip(rac.route_analyzers, ["Before", "After"], [subfeeds[0], subfeeds[1]]):
            temp_df = rac.collect_pickles_using_buffer(route_analyzer, **{"cols": ["from_stop_I", "to_stop_I"],
                                                                          "geometry": [centroid_gdf, centroid_gdf],
                                                                          "buffer": [buffer, buffer]})
            temp_df = drop_nans(temp_df)
            print(subfeed)

            for measure in [mean_temporal_distance, mean_trip_n_boardings, time_weighted_simpson]:
                all_temp_dicts.setdefault(measure, [])
                samples = temp_df[measure].to_numpy()
                bs_result = bs.bootstrap(samples, stat_func=bs_stats.mean, num_iterations=1000, iteration_batch_size=1)

                bs_dict = {"upper": bs_result.upper_bound,
                           "lower": bs_result.lower_bound,
                           "value": bs_result.value,
                           "diff_upper": bs_result.upper_bound - bs_result.value,
                           "diff_lower": bs_result.lower_bound - bs_result.value}
                bs_dict = {k: round(v, 5) for k, v in bs_dict.items()}
                bs_dict["feed"] = subfeed
                all_temp_dicts[measure].append(bs_dict)
                print(measure, bs_dict)
    for k, v in all_temp_dicts.items():
        print(k)
        print(DataFrame(v))


def base_data_generator(feeds=None, **kwargs):
    for ci, city in enumerate(CITIES.keys(), start=0):
        print(city)
        subfeeds = CITIES[city]
        if not feeds or set(subfeeds) <= set(feeds):
            rac = get_journey_analyze_comparer([subfeeds[1], subfeeds[0]], **kwargs)
            feed_dict = get_feed_dict(subfeeds[0])
            for df in rac.get_feed_difference_generator():
                yield city, feed_dict, df


def plot_outlier_ods(feeds=None, **kwargs):
    for ci, city in enumerate(CITIES.keys(), start=0):
        print(city)
        subfeeds = CITIES[city]

        if not feeds or set(subfeeds) <= set(feeds):
            rac = get_journey_analyze_comparer([subfeeds[1], subfeeds[0]], **kwargs)
            feed_dict = get_feed_dict(subfeeds[0])
            gtfs = feed_dict["gtfs"]
            city_coords = feed_dict["city_coords"]
            for df in rac.get_feed_difference_generator():
                df = gtfs.add_coordinates_to_df(df)
                df = gtfs.add_coordinates_to_df(df)

                for i, column in enumerate(df.columns):
                    if "diff" in column:
                        temp_df = df[["from_lat", "to_lat", "from_lon", "to_lon", column]].copy()
                        q1, q3 = np.percentile(temp_df[column], [25, 75])
                        iqr = q3 - q1
                        lower_bound = q1 - (3 * iqr)
                        upper_bound = q3 + (3 * iqr)
                        print(lower_bound, upper_bound)
                        temp_df = temp_df[(temp_df[column] < lower_bound) | (temp_df[column] > upper_bound)]

                        for direction, color in zip(["from_", "to_"], ["red", "blue"]):

                            fig = plt.figure()
                            ax = fig.add_subplot(111, projection="smopy_axes")
                            bbox = get_custom_spatial_bounds(**city_coords)
                            ax.set_map_bounds(**bbox)
                            ax.set_plot_bounds(**bbox)

                            agg_df = temp_df[[direction+"lat", direction+"lon", column]].groupby([direction+"lat",
                                                                                                  direction+"lon"])
                            agg_df = agg_df.agg({column: 'count'}, axis=1)
                            agg_df = agg_df.reset_index()
                            ax.scatter(agg_df[direction+"lon"].to_list(), agg_df[direction+"lat"].to_list(), c=color,
                                       s=[0.1 * x for x in agg_df[column].to_list()])
                            ax.set_title("{direction}{col}, min: {min}, max: {max}".format(direction=direction,
                                                                                           col=column,
                                                                                           min=min(agg_df[column]),
                                                                                           max=max(agg_df[column])))
                            plt.savefig(os.path.join(makedirs(DEBUG_FIGS), city+"_"+direction+column+"_far_out_map.png"),
                                        format="png", dpi=600)


def get_max_and_min_stops(**kwargs):
    for city, feed_dict, df in base_data_generator(**kwargs):
        cols = list(df.columns)
        cols = [x for x in cols if "stop_I" not in x and "diff" in x]
        df = df.set_index(["from_stop_I", "to_stop_I"])
        print("------------------{city}----------------------".format(city=city))
        print(df[cols].idxmax(axis=0))


def plot_4_category_map(feeds, **kwargs):
    rac = get_journey_analyze_comparer(feeds, **kwargs)
    condition_tuples = [(largest_headway_gap, "<=", 20), (number_of_most_common_journey_variant, ">=", 6)]
    #condition_tuples = [(time_weighted_simpson, ">=", 0.8), (mean_temporal_distance, "<=", 60)]
    #condition_tuples = [(time_weighted_simpson, ">=", 0.7), (largest_headway_gap, "<=", 15)]

    for feed in feeds:
        df, condition_strs = rac.thresholded_counts(condition_tuples, feed[-4:], **kwargs)
        df["max_column"] = df.idxmax(axis=1)
        df = df.reset_index()
        #max_categories = set(df["max_column"].to_list())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="smopy_axes")

        feed_dict = get_feed_dict(feeds[0])
        gtfs = feed_dict["gtfs"]
        city_coords = feed_dict["city_coords"]
        bbox = get_custom_spatial_bounds(**city_coords)
        ax.set_map_bounds(**bbox)
        ax.set_plot_bounds(**bbox)

        df = gtfs.add_coordinates_to_df(df)
        color_dict = {k: v for k, v in zip(condition_strs, ["purple", "red", "plum", "blue"])}
        for cat, color in color_dict.items():
            temp_df = df.loc[df["max_column"] == cat].copy()
            ax.scatter(temp_df["lon"].to_list(), temp_df["lat"].to_list(), c=color, s=1)
        plot_custom_label_point(color_dict)
        plt.savefig(os.path.join(makedirs(MAPS_CATEGORICAL_DIRECTORY),
                                 feed + "+".join([x[0] + "_" + str(x[2]) for x in condition_tuples]) + "_4_cat.png"),
                    format="png", dpi=600)


def plot_custom_label_point(types_color_dict):
    handles = []
    labels = []

    for type, color in types_color_dict.items():
        point = Patch(color=color, label=type)
        handles.append(point)
        labels.append(type)

    plt.figlegend(labels=labels, handles=handles, loc='upper right', ncol=1)


# TODO: route variant density (spatially)

# TODO: Quantify trip/mode presence on fastest paths
# TODO: measures for headway distribution: expected waiting time at highest % stop

# TODO: introduce kwargs for plot names/folders and recalculation True
# TODO: test duplicate multiple stats Dataframes together and calculate mean as a stress test

# TODO: routing with maximum of the minimum frequency as pareto criteria:
#       For each stop-to-stop section set min(previous_min_frequency, current_frequncy), at each domintates, maximize
"""
for VARIABLE in 780 1249 1280 518 354 1016
do
echo $VARIABLE
python3 research/route_diversity/routing_scripts.py cc helsinki_2018 helsinki_2014 -t $VARIABLE
done
"""


def main():
    parser = argparse.ArgumentParser()
    main_command = parser.add_mutually_exclusive_group(required=True)
    main_command.add_argument("-rr", "--run_routing", action='store_true', help="runs routing, outputs pickle file")
    main_command.add_argument("-rs", "--run_slurm_routing", action='store_true',
                              help="runs routing for slurm, requires -sai and -sal input")
    main_command.add_argument("-njpa", "--run_routing_to_njpa", action='store_true', help="")
    main_command.add_argument("-hm", "--heatmap", action='store_true', help="")
    main_command.add_argument("-mom", "--means_on_map", action='store_true', help="")
    main_command.add_argument("-ods", "--thresholded_ods", action='store_true', help="")
    main_command.add_argument("-d", "--diversity", action='store_true', help="")
    main_command.add_argument("-ss", "--stop2stop", action='store_true', help="")
    main_command.add_argument("-pp", "--pair_plot", action='store_true', help="")
    main_command.add_argument("-pj", "--plot_journeys", action='store_true',
                              help="plots all optimal journeys to specified target")
    main_command.add_argument("-st", "--stress_test", action='store_true', help="")
    main_command.add_argument("-pd", "--plot_distributions", action='store_true', help="")
    main_command.add_argument("-psd", "--plot_split_distributions", action='store_true', help="")
    main_command.add_argument("-po", "--plot_outliers", action='store_true', help="")
    main_command.add_argument("-mm", "--min_max", action='store_true', help="")
    main_command.add_argument("-4cat", "--plot_4_category_map", action='store_true', help="")
    main_command.add_argument("-ci", "--confidence_intervals", action='store_true', help="")

    parser.add_argument("feeds", type=str, help="name of the feed or multiple feeds", nargs='*')
    parser.add_argument("-t", "--targets", type=int, help="id(s) of the target(s) stop(s)", nargs='*')
    parser.add_argument("-o", "--origin", type=int, help="id of the origin stop")
    parser.add_argument("-sai", "--slurm_array_i", type=int,
                        help="slurm array index, required when using run_slurm_routing")
    parser.add_argument("-sal", "--slurm_array_length", type=int,
                        help="slurm array length, required when using run_slurm_routing")
    parser.add_argument("-s", "--sample_size", type=int, const=SAMPLE_SIZE, default=None, nargs="?",
                        help="number of routing destinations")
    parser.add_argument("-sf", "--sample_fraction", type=float, const=SAMPLE_FRACTION, default=None, nargs="?",
                        help="routing destinations as a fraction of all stops")
    parser.add_argument("-td", "--tesselation_distance", type=int, default=TESSELATION_DISTANCE,
                        help="")
    parser.add_argument("-tps", "--transfer_penalty_seconds", type=int, default=TRANSFER_PENALTY_SECONDS,
                        help="")
    parser.add_argument("--fig_format", type=str, help="name of the feed or multiple feeds")
    parser.add_argument("--run_if_missing", action='store_true', help="")


    """
    parser.add_argument("-rc", "--recalculate", action='store_true',
                        help="use to force recalculation of results TODO")
    parser.add_argument("-fn", "--file_name", type=str, help="output file name")
    parser.add_argument("-dir", "--directory", type=str, help="output directory")

    """
    args = parser.parse_args()
    kwargs = {"sample_size": args.sample_size, "sample_fraction": args.sample_fraction,
              "tesselation_distance": args.tesselation_distance,
              "transfer_penalty_seconds": args.transfer_penalty_seconds,
              "fig_format": args.fig_format,
              "run_if_missing": args.run_if_missing}
    print(args)
    if any([args.run_slurm_routing, args.run_routing_to_njpa, args.diversity, args.heatmap, args.means_on_map,
            args.thresholded_ods]) and (args.sample_size is None and args.sample_fraction is None):
        parser.error("define sample size either using sample size or sample fraction")

    if args.run_routing:
        for feed in args.feeds:
            run_routing(feed=feed, targets=args.targets)
    elif args.run_slurm_routing:
        run_routing_slurm_batch(feed=args.feeds[0],
                                slurm_array_i=args.slurm_array_i,
                                slurm_array_length=args.slurm_array_length, **kwargs)
    elif args.run_routing_to_njpa:
        routing_to_njpa(feed=args.feeds[0], targets=args.targets, slurm_array_i=args.slurm_array_i,
                        slurm_array_length=args.slurm_array_length, **kwargs)
    elif args.diversity:
        run_plot_diversity(feed=args.feeds[0], target=args.targets[0], **kwargs)
    elif args.stop2stop:
        run_stop_to_stop(feed=args.feeds[0], origin=args.origin, target=args.targets[0], **kwargs)
    elif args.plot_journeys:
        plot_journeys_to_target(feed=args.feeds[0], target=args.targets[0])
    elif args.heatmap and args.feeds:
        od_heatmap(feeds=args.feeds, **kwargs)
    elif args.means_on_map and args.feeds:
        plot_means_on_map(feeds=args.feeds, **kwargs)
    elif args.thresholded_ods:
        plot_ods(feeds=args.feeds, **kwargs)
    elif args.plot_distributions:
        plot_diff_distributions(feeds=args.feeds, **kwargs)
    elif args.plot_split_distributions:
        plot_split_distributions(feeds=args.feeds, **kwargs)
    elif args.plot_outliers:
        plot_outlier_ods(feeds=args.feeds, **kwargs)
    elif args.min_max:
        get_max_and_min_stops(feeds=args.feeds, **kwargs)
    elif args.confidence_intervals:
        confidence_intervals_using_bootstrap(feeds=args.feeds, **kwargs)
    else:
        for city, feeds in CITIES.items():
            print(city)
            feeds.sort(reverse=True)
            if args.heatmap:
                od_heatmap(feeds=feeds, **kwargs)
            if args.pair_plot:
                run_pair_plots(feeds=feeds, **kwargs)
            if args.means_on_map:
                plot_means_on_map(feeds=feeds, **kwargs)
            if args.plot_4_category_map:
                plot_4_category_map(feeds=feeds, **kwargs)

    """
    elif args.stress_test:
        stress_test(feeds=args.feeds, target=args.targets[0])
    """


if __name__ == "__main__":
    main()
