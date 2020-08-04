import sys

import argparse
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from gtfspy.util import df_to_utm_gdf
from research.route_diversity.measures_and_colormaps import *
from research.route_diversity.journey_analyze_pipeline import *
from research.route_diversity.static_route_type_analyzer import RouteTypeAnalyzer, StaticGTFSStats, \
    get_route_type_analyzer
from research.route_diversity.rd_utils import check_day_start, apply_suffix, tidy_label
from research.route_diversity.static_navigability import StaticNavigabilityAnalyzer
from research.route_diversity.radar_chart import make_spider, parallel_coordinates


def run_static_network_measures(**kwargs):
    """
    Collects performance measures:
    For both all routes and frequent network/trunk routes
        Length: trajectory, segment
        Kilometrage
    Route overlap
    avg_segment_frequency
    *Resource (=kilometrage) allocation measure:
        coverage vs. frequency (frequency = trunk routes)
        route types
        time of day?
    Weighted mean service hours 'weighted_mean_service_hours',
    'long_service_hour_kms',
    'long_service_hour_prop'
    Network similarity over day (Jaccard)
    Number of route variants
    :param kwargs:
    :return:
    """
    datas = []
    for feeds in CITIES.values():
        for feed in feeds:
            print(feed.split("_"))
            #city, year = feed.split("_")
            fname_or_conn = os.path.join(CITIES_DIR, feed, "week" + SQLITE_SUFFIX)
            gtfs = GTFS(fname_or_conn)
            day_start = check_day_start(gtfs, 2)
            stats_dict = {}

            # Retrieve measures using RouteTypeAnalyzer
            rta = get_route_type_analyzer(feed)
            all_routes = rta.get_all_route_types()

            cross_routes = all_routes.loc[all_routes.cross_route].sum(axis=0).n_trips / all_routes.loc[
                all_routes.goes_to_hub].sum(axis=0).n_trips

            stats_dict.update({"city": feed,
                               number_of_route_variants: len(all_routes.index),
                               ratio_name(cross_route): cross_routes})

            # Retrieve measures using StaticGTFSStats
            # Global network stats
            sg = StaticGTFSStats(day_start, fname_or_conn=fname_or_conn)
            s_stats = sg.calculate_network_stats(frequency_threshold=0)
            stats_dict.update(s_stats)

            # Trunk vs. all stats
            for time_of_day in [peak, day]:
                trunk_params = {"frequency_threshold": 4}
                trunk_params.update(time_dict[time_of_day])
                p_stats = sg.calculate_prop_stats(trunk_params, **time_dict[time_of_day])
                p_stats = apply_suffix(p_stats, time_of_day)
                stats_dict.update(p_stats)

                # Stats relevant for peak hour
                h_stats = sg.calculate_hour_stats(**time_dict[time_of_day])
                h_stats = apply_suffix(h_stats, time_of_day)

                stats_dict.update(h_stats)

            # Retrieve measures using StaticNavigabilityAnalyzer
            sna = get_static_navigability_analyzer(feed)
            stats_dict.update(sna.time_variation_navigability_measures())

            datas.append(stats_dict)
    df = DataFrame(datas)

    df.to_pickle(NETWORK_STATS_PICKLE_FNAME)


def parallel_plot():

    df = pandas.read_pickle(NETWORK_STATS_PICKLE_FNAME)
    df = df.loc[df.city.isin(FEED_LIST)]
    ax = parallel_coordinates(df, "city")


def radar():
    """

    :return:
    """
    n_rows = 4
    n_cols = 3
    included_measures_and_flip = {number_of_route_variants: True,
                                  avg_segment_frequency + "_" + peak: False,
                                  prop_length + "_" + peak: False,
                                  prop_length + "_" + day: False,
                                  route_overlap + "_" + peak: True,
                                  route_overlap + "_" + day: True,
                                  weighted_mean_service_hours: False}
    """
    included_measures_and_flip = {cross_route_ratio: False, weighted_mean_service_hours: False,
                                  "prop_kilometrage_day": False, 'route_overlap_peak': True,
                                  'avg_segment_frequency_peak': False} #,
                                  #'route_length_per_capita': False, 'route_overlap': True
    """
    alphabet = "ABCDEFGHIJKL"
    letter_dict = {measure: letter for measure, letter in zip(included_measures_and_flip.keys(), alphabet)}
    included_measures_and_flip = {letter_dict[key]: value for key, value in included_measures_and_flip.items()}
    letter_tuples = [(measure, letter) for measure, letter in letter_dict.items()]
    letter_tuples = sorted(letter_tuples, key=lambda x: x[1])

    df = pandas.read_pickle(NETWORK_STATS_PICKLE_FNAME)

    df = df.rename(index=str, columns=letter_dict)
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    cols = list(df)
    cols = [col for col in cols if col not in ['city', 'id', 'name', 'feeds', 'download_d', 'population']]
    cols = list(included_measures_and_flip.keys())
    for col, flip in included_measures_and_flip.items():
        if flip:
            df[col] = 1 + -1 * (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        else:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        #df[col] = (df[col] - df[col].mean()) / df[col].std()
    # initialize the figure
    my_dpi = 90
    fig = plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)
    gs = fig.add_gridspec(n_rows, n_cols)
    df = df[cols+["city"]]
    # Loop to plot
    for row, (city, feeds) in enumerate(CITIES.items()):
        temp_df = df.loc[df["city"].isin(feeds)]
        temp_df = temp_df.set_index("city") #drop("city", axis=1)
        value_lists, colors, labels = [], [], []
        for feed, color in zip(feeds, ["b", "r"]):
            values = temp_df.loc[feed].values.tolist()
            value_lists.append(values)
            colors.append(color)
            if feed[-4:].isnumeric():
                labels.append(feed[-4:])
            else:
                labels.append(None)
        make_spider(value_lists=value_lists, row=row,
                    title=city,
                    colors=colors, measures=cols, ylim=1, n_cols=n_rows, n_rows=n_cols, labels=labels)
    ax = plt.subplot(gs[-1, :])
    for i, l_tuple in enumerate(letter_tuples):

        text = apply_static_alias(l_tuple[0]).replace("_", " ").capitalize()
        inverted_indicator = included_measures_and_flip[l_tuple[1]]
        #if len(text) >= 20:
            #text = split_by_char_closest_to_middle(text, delimiter=" ", filler="\n    ")

        ax.text(-0, 1 - 0.25 * i, l_tuple[1]+": "+text + ("*" if inverted_indicator else ""), fontsize=15, color="dimgrey")
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(makedirs(FIGS_DIRECTORY) + "network_measures_radar.pdf", format="pdf")

    #plt.show()


def get_static_navigability_analyzer(feed):
    fname_or_conn = os.path.join(CITIES_DIR, feed, "week" + SQLITE_SUFFIX)
    gtfs = GTFS(fname_or_conn)
    day_start = check_day_start(gtfs, 2)
    return StaticNavigabilityAnalyzer(gtfs, day_start, feed)


def get_avg_jaccard(feed):
    sna = get_static_navigability_analyzer(feed)
    sna.time_variation_navigability_measures()


def plot_route_category_maps(feed, **kwargs):
    rta = get_route_type_analyzer(feed, **kwargs)
    rta.plot_route_category_maps(**kwargs)


def get_distributions(feed, **kwargs):
    # TODO: headways (trajectory, route label)
    """get distributions of specific measures"""
    rta = get_route_type_analyzer(feed, **kwargs)
    all_routes = rta.get_all_route_types()
    for route_type in ROUTE_TYPE_COLORS.keys():
        all_routes.loc[all_routes[route_type], "route_type"] = route_type

    all_routes = all_routes[["route_type", "n_trips", segment_kilometrage]].groupby(["route_type", "n_trips"]).agg(
        {segment_kilometrage: 'sum'}, axis=1)
    all_routes = all_routes.unstack(level=-1)
    all_routes.columns = all_routes.columns.droplevel()
    all_routes = all_routes.T

    all_routes = all_routes.loc[:, ROUTE_TYPE_COLORS.keys()]

    all_routes = all_routes.reindex(range(1, 21)).fillna(0)
    plt.figure(figsize=(12, 8))

    all_routes.plot(kind='bar', color=ROUTE_TYPE_COLORS.values(), stacked=True, legend=False)
    plt.ylim([0, 7100])
    plt.ylabel("veh. km.")
    plt.tight_layout()
    plt.savefig(os.path.join(makedirs(os.path.join(MAPS_DIRECTORY, "distributions")),
                             feed+"_frequency_by_route_type.svg"), format="svg", transparent=True)

    fig = plt.figure(figsize=(2, 2))
    # Create legend handles manually
    handles = [Patch(color=ROUTE_TYPE_COLORS[x], label=tidy_label(x, capitalize=True)) for x in ROUTE_TYPE_COLORS.keys()]
    # Create legend
    legend = plt.legend(handles=handles)
    # Get current axes object and turn off axis
    plt.gca().set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(makedirs(os.path.join(MAPS_DIRECTORY, "distributions")),
                             "legend.pdf"), format="pdf", bbox_inches="tight", transparent=True)


def get_service_area(feed, buffer, **kwargs):
    rta = get_route_type_analyzer(feed, **kwargs)
    stops = rta.stops()
    gdf, crs = df_to_utm_gdf(stops)
    gdf["geometry"] = gdf.buffer(buffer)
    gdf["all"] = 1
    gdf = gdf.dissolve("all")
    convex_hull = gdf["geometry"].convex_hull
    area = gdf.area
    coverage = area / convex_hull.area
    print(feed, "area:", area / 10**6, "coverage", coverage)


# TODO: route variant density (spatially)

# TODO: measures for headway distribution: expected waiting time at highest % stop

# TODO: introduce kwargs for plot names/folders and recalculation True


def main():
    parser = argparse.ArgumentParser()
    main_command = parser.add_mutually_exclusive_group(required=True)
    main_command.add_argument("-cm", "--calculate_measures", action='store_true', help="")
    main_command.add_argument("-rp", "--radar_plot", action='store_true', help="")
    main_command.add_argument("-pp", "--parallel_plot", action='store_true', help="")
    main_command.add_argument("-tj", "--time_jaccard", action='store_true', help="")
    main_command.add_argument("-rtp", "--route_type_plot", action='store_true', help="")
    main_command.add_argument("-sa", "--service_area", action='store_true', help="")
    main_command.add_argument("-d", "--distributions", action='store_true', help="")


    parser.add_argument("--feeds", type=str, help="name of the feed or multiple feeds", nargs='+')
    parser.add_argument("--suffix", type=str, help="suffix used in filename", default="")

    parser.add_argument("-ft", "--frequency_threshold", type=int, help="minimum frequency of routes or segments to be "
                                                                       "included in calculations")
    parser.add_argument("-pft", "--plot_frequency_threshold", type=int, default=0,
                        help="minimum frequency of routes or segments to be colored in plots")
    parser.add_argument("-day", action='store_true', help="uses start and end time specified for day in settings")
    parser.add_argument("--plot_center", action='store_true', help="")
    parser.add_argument("--legend", action='store_true', help="")


    args = parser.parse_args()

    print(args)
    time_tag = "peak"
    if args.day:
        time_tag = "day"
    kwargs = {}
    kwargs.update(time_dict[time_tag])
    kwargs.update({"suffix": args.suffix, "plot_frequency_threshold": args.plot_frequency_threshold,
                   "plot_center": args.plot_center, "plot_legend": args.legend, "format": "svg"})
    if args.calculate_measures:
        run_static_network_measures()
    elif args.radar_plot:
        radar()
    elif args.parallel_plot:
        parallel_plot()
    else:
        if not args.feeds:
            feeds = FEED_LIST
        else:
            feeds = args.feeds
        for feed in feeds:
            print(feed)
            if args.time_jaccard:
                get_avg_jaccard(feed)
            elif args.route_type_plot:
                plot_route_category_maps(feed, **kwargs)
            elif args.service_area:
                get_service_area(feed, 1000, **kwargs)
            elif args.distributions:
                get_distributions(feed, **kwargs)


if __name__ == "__main__":
    main()

