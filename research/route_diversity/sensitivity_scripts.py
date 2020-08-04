import itertools
import matplotlib.pyplot as plt

from research.route_diversity.journey_analyze_pipeline import *
from research.route_diversity.journey_analyze_plots import JourneyAnalyzePlots


def sensitivity_analysis_of_routing_parameters(city, target, return_plot=False):
    settings = {"transfer_margin": [0, 180, 360],
                "walk_speed": [round(35.0/60, 2), round(70.0/60, 2), round(140.0/60, 2)],
                "walk_distance": [500, 1000, 2000]}
    combinations = itertools.product(settings["transfer_margin"], settings["walk_speed"], settings["walk_distance"])
    settings_list = [{"transfer_margin": x, "walk_speed": y, "walk_distance": z} for x, y, z in combinations]
    base_settings = {key: value[1] for key, value in settings.items()}
    settings_list.insert(0, base_settings)
    feed = CITIES[city][0]
    base_df = None
    for i, setting in enumerate(settings_list):
        print(feed, "target:", target)
        feed_dict = get_feed_dict(feed)
        feed_dict.update(**setting)
        ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
        if i == 0:
            base_df = ra_pipeline.calculate_diversity_for_target(target)
            base_df.set_index("stop_I", inplace=True)
        else:
            df = ra_pipeline.calculate_diversity_for_target(target)
            df.drop(columns=["lat", "lon"], inplace=True)
            df.set_index("stop_I", inplace=True)
            diff_df = subtract_dataframes(df, base_df)

            diff_df.reset_index(inplace=True)
            if return_plot:
                for fig in ra_pipeline.plot_multiple_measures_on_maps(diff_df, target, "",
                                                                      figs_directory=SENSITIVITY_DIRECTORY):
                    yield fig, setting
            else:
                ra_pipeline.plot_multiple_measures_on_maps(diff_df, target, "", figs_directory=SENSITIVITY_DIRECTORY)
            # TODO: also timerange sensitivity


def simple_sensitivity_analysis_of_routing_parameters(city, target, return_plot=False):
    test_ranges = {"transfer_margin": [0, 360],
                   "walk_speed": [round(35.0/60, 2), round(140.0/60, 2)],
                   "walk_distance": [500, 2000]}

    base_settings = {"transfer_margin": 180,
                     "walk_speed": round(70.0/60, 2),
                     "walk_distance": 1000}
    measures = ["number_of_journey_variants",
                "number_of_fp_journeys",
                "mean_temporal_distance",
                "mean_trip_n_boardings"]

    feed = CITIES[city][0]

    feed_dict = get_feed_dict(feed)
    feed_dict.update(**base_settings)
    ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
    base_df = ra_pipeline.calculate_diversity_for_target(target, measures=measures)
    base_df.set_index("stop_I", inplace=True)

    for i, (key, values) in enumerate(test_ranges.items()):
        main_df = base_df.copy()
        for v in values:
            setting = base_settings.copy()
            setting.update({key: v})

            print(feed, "target:", target)

            feed_dict = get_feed_dict(feed)
            feed_dict.update(**setting)
            ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
            df = ra_pipeline.calculate_diversity_for_target(target, measures=measures)
            df.drop(columns=["lat", "lon"], inplace=True)
            df.set_index("stop_I", inplace=True)
            diff_df = subtract_dataframes(df, base_df)
            diff_df = diff_df.rename(index=str, columns={x: x+"_|{key}={value}".format(key=key, value=str(v)) for x in
                                                         list(diff_df) if x not in ["stop_I", "lat", "lon"]})
            main_df = main_df.merge(diff_df)

        for measure in measures:

            cols = [x for x in list(main_df) if measure in x and measure != x]

            fig, ax_objs = plt.subplots(nrows=len(cols), ncols=1, sharex=True, sharey=True)
            fig.suptitle(measure)

            for col, marker, ax1, c in zip(cols, ["1", "2"], ax_objs, ["red", "blue"]):
                ax1.scatter(main_df[measure], main_df[col], color=c, marker=marker, label=col.split("|")[1], alpha=0.1)
                ax1.set_xlabel("value with base settings")
                ax1.set_ylabel("difference to base settings")
                ax1.set_ylim(MEASURE_PLOT_PARAMS[measure]["diff_lims"])
                ax1.set_xlim(MEASURE_PLOT_PARAMS[measure]["lims"])

            fig.legend(loc='upper left')

            plt.savefig(os.path.join(SENSITIVITY_DIRECTORY, measure + "_" + key + "_" + city + "_" + str(target) + ".png"),
                        format="png", dpi=300)


            #diff_df.reset_index(inplace=True)

            #ra_pipeline.plot_multiple_on_map(target, diff_df, "", figs_directory=SENSITIVITY_DIRECTORY)


def sensitivity_analysis_of_transfer_penalty(city, target):
    transfer_penalties = [0, 180, 360, 540]

    feed = CITIES[city][0]
    base_df = None
    for i, tp in enumerate(transfer_penalties):
        print(feed, "target:", target)
        feed_dict = get_feed_dict(feed)
        feed_dict["transfer_penalty_seconds"] = tp
        ra_pipeline = JourneyAnalyzePlots(**feed_dict)
        if i == 0:
            base_df = ra_pipeline.calculate_diversity_for_target(target)
            base_df.set_index("stop_I", inplace=True)
        else:
            df = ra_pipeline.calculate_diversity_for_target(target)
            df.drop(columns=["lat", "lon"], inplace=True)
            df.set_index("stop_I", inplace=True)
            diff_df = subtract_dataframes(df, base_df)

            diff_df.reset_index(inplace=True)
            ra_pipeline.plot_multiple_measures_on_maps(diff_df, target, "", figs_directory=SENSITIVITY_DIRECTORY,
                                                       suffix="_tp-" + str(tp))


def subtract_dataframes(df1, df2, **kwargs):
    cols1 = list(df1)
    cols2 = list(df2)
    common_cols = [col for col in cols1 if col in cols2]
    kwargs["suffixes"] = ["_x", "_y"]
    kwargs["right_index"] = True
    kwargs["left_index"] = True
    prefix = kwargs.pop("prefix", "diff_")
    df = df1.merge(df2, **kwargs)
    for col in common_cols:
        df[prefix+col] = df[col+"_x"] - df[col+"_y"]
        df.drop([col+"_x", col+"_y"], inplace=True, axis=1)
    return df
