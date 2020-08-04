import os

from seaborn import pairplot
import matplotlib.pyplot as plt
import pandas as pd

from gtfspy.smopy_plot_helper import *
from gtfspy.route_types import route_type_to_color_iterable, route_type_to_zorder, ROUTE_TYPE_TO_COLOR
from gtfspy.routing.journey_path_analyzer import NodeJourneyPathAnalyzer
from gtfspy.util import makedirs
from research.route_diversity.measures_and_colormaps import get_colormap
from research.route_diversity.journey_analyze_pipeline import JourneyAnalyzePipeline
from research.route_diversity.diversity_settings import *
from research.route_diversity.rd_utils import save_fig, set_bbox, set_bbox_using_ratio


class JourneyAnalyzePlots(JourneyAnalyzePipeline):
    def __init__(self, *args, **kwargs):
        super(JourneyAnalyzePlots, self).__init__(*args, **kwargs)

    def plot_sample_stops(self, stop_ids):
        fig = plt.figure()

        ax = plt.subplot(projection="smopy_axes")
        lats = []
        lons = []
        stops = []
        for id in stop_ids:
            lat, lon = self.gtfs.get_stop_coordinates(id)
            lats.append(lat)
            lons.append(lon)
            stops.append(str(id))

        ax.set_map_bounds(**self.bbox)
        ax.scatter(lons, lats, s=3)
        for lat, lon, stop in zip(lats, lons, stops):
            ax.text(lon, lat, stop, color="red", fontsize=12)
        ax.set_plot_bounds(**self.bbox)
        fig.savefig(os.path.join(FIGS_DIRECTORY, "sample_map_" + self.feed + ".png"), format="png", dpi=300)

    def plot_multiple_measures_on_maps(self, df=None, target=None, figs_directory=None, **kwargs):
        assert not (df is None and target is None)
        if df is None:
            if target not in self.performance_measure_dict.keys():
                df = self.calculate_diversity_for_target(target)
            else:
                df = self.performance_measure_dict[target]
        print(list(df))
        for measure in [n for n in list(df.columns) if n not in ["from_stop_I", 'to_stop_I', 'from_lat', 'from_lon']]:
            self.plot_measure_on_map(df["from_lon"].tolist(), df["from_lat"].tolist(), df[measure].tolist(), target=target,
                                     measure=measure, figs_directory=makedirs(figs_directory), **kwargs)

    def plot_measure_on_map(self, lons, lats, values, target="", measure="",
                            figs_directory=None, save_fig=True, fname_simple=False, separate_colorbar=False, title=True,
                            **kwargs):
        fname_prefix = kwargs.get("fname_prefix", "")
        fname_suffix = kwargs.get("fname_suffix", "_" + self.routing_id(target))

        cmap_mode = kwargs.get("cmap_using_data", False)

        fig = plt.figure()
        if cmap_mode:
            cmap, norm = get_colormap(data=values, **kwargs)
        else:
            cmap, norm = get_colormap(fname_prefix + measure)
        ax = plt.subplot(projection="smopy_axes")
        im = ax.scatter(lons, lats, c=values,
                        cmap=cmap, norm=norm, s=3)
        if target:
            lat, lon = self.gtfs.get_stop_coordinates(target)
            ax.scatter(lon, lat, s=30, c="green", marker="X", zorder=1)
        ax.set_plot_bounds(**self.bbox)
        if title:
            ax.set_title(measure)
        ax.add_scalebar()

        if not separate_colorbar:
            plt.colorbar(im)

        if save_fig:
            figs_directory = figs_directory if figs_directory else FIGS_DIRECTORY
            if fname_simple:
                save_fname = fname_prefix + measure + "_" + self.feed[:-5] + "__.png"
            else:
                save_fname = fname_prefix + measure + "_" + self.feed[:-5] + fname_suffix + ".png"
            print("saving:", save_fname)
            fig.savefig(os.path.join(figs_directory, save_fname), format="png", dpi=300,
                        bbox_inches='tight')
            if separate_colorbar:
                fig, ax2 = plt.subplots()
                plt.colorbar(im, ax=ax2)
                ax2.remove()
                plt.savefig(os.path.join(makedirs(os.path.join(figs_directory, "cbars")), fname_prefix + measure + "_" +
                                         "cbar" + fname_suffix + ".png"),
                            bbox_inches='tight')
        else:
            return fig

    @save_fig
    def plot_temporal_distance(self, njpa, fastest_path_only, ax=None, **kwargs):
        """

        :return: matplotlib.axes
        """
        if not ax:
            ax = plt.gca()
        tz = njpa.gtfs.get_timezone_pytz()
        if fastest_path_only:
            f = njpa.plot_fastest_temporal_distance_profile
            journey_path_letters = njpa.get_fp_path_letters()
        else:
            f = njpa.plot_temporal_distance_profiles
            journey_path_letters = njpa.get_all_path_letters()

        return f(ax=ax,
                 timezone=tz,
                 plot_journeys=True,
                 journey_letters=journey_path_letters,
                 format_string="%H:%M:%S")

    @save_fig
    def plot_trajectory_variants_map(self, njpa, target_id, origin_id, fastest_path_only, ax=None, **kwargs):
        origin_marker = {"label": "Origin", "color": "red", "marker": "o", "type": "text"}
        destination_marker = {"label": "Destination", "color": "green", "marker": "X", "type": "text"}

        if not ax:
            ax = plt.gca()

        stop_letter_dict = njpa.path_letters_for_stops(fp_only=fastest_path_only)
        t_lat, t_lon = self.gtfs.get_stop_coordinates(target_id)
        o_lat, o_lon = self.gtfs.get_stop_coordinates(origin_id)
        if True or kwargs.get("zoom_factor", None):
            bbox = {"lon_min": min(t_lon, o_lon),
                    "lon_max": max(t_lon, o_lon),
                    "lat_min": min(t_lat, o_lat),
                    "lat_max": max(t_lat, o_lat)}

            bbox = set_bbox_using_ratio(ratio=0.6, bbox=bbox)
        else:
            bbox = self.bbox
        ax.set_map_bounds(**bbox)
        ax.set_plot_bounds(**bbox)

        for lats, lons, leg_type in njpa.get_journey_trajectories(fp_only=fastest_path_only):
            ax.plot(lons, lats, c=ROUTE_TYPE_TO_COLOR[leg_type], zorder=1, label=ROUTE_TYPE_TO_SHORT_DESCRIPTION[leg_type])

        ax.scatter(t_lon, t_lat, s=100, c="green", marker="X", zorder=2, label="Destination")
        ax.scatter(o_lon, o_lat, s=100, c="red", marker="o", zorder=2, label="Origin")
        for stop_id, letters in stop_letter_dict.items():
            lat, lon = self.gtfs.get_stop_coordinates(stop_id)
            ax.scatter(lon, lat, s=20, c="grey", marker="o", zorder=2, label="Boarding stop")
            # ax.annotate(",".join(letters), xy=(lon, lat), color="m", fontsize=10, va="top", ha="left", zorder=10)
            ax.text(lon, lat, ",".join(letters), color="m", fontsize=10, va="top", ha="left", zorder=10, label="Journey label")
            #if text:
                #text.set_path_effects([path_effects.Stroke(linewidth=19, foreground='white'), path_effects.Normal()])

        #custom_legend(ax,)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        ax.add_scalebar()
        return ax

    @save_fig
    def plot_diversity_table(self, njpa, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()
        diversity_dict = njpa.get_simple_diversities()
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=[[str(round(x, 3)) if x else x] for x in diversity_dict.values()],
                             rowLabels=list(diversity_dict.keys()),
                             colWidths=[0.2, 0.2],
                             loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        return the_table

    def stop_to_stop_routes(self, origin_id=None, target_id=None, origin_name=None, target_name=None,
                            plot_separately=True,
                            **kwargs):
        # TODO: map: fix offset of route labels,
        #  Add journey labels to legend
        # TODO: profile: change y-label
        #  Add journey labels to legend
        G = self.gtfs
        if origin_name:
            origin_id = G.get_stop_I_from_name(origin_name)

        if target_name:
            target_id = G.get_stop_I_from_name(target_name)

        store_dict = self.find_pickle(target_id)
        labels = store_dict["labels"]
        walk_time = store_dict["walk_time"]

        if not plot_separately:
            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222, projection="smopy_axes")
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

        else:
            ax1 = ax2 = ax3 = ax4 = None

        fig_format = kwargs.get("fig_format", "png")
        folder = kwargs.get("folder", "")
        fname = kwargs.get("fname",
                           "s2s" + "_o-" + str(origin_id) + "_t-" + str(target_id) + self.feed)
        kwargs['fname'] = fname
        kwargs['plot_separately'] = plot_separately

        njpa = NodeJourneyPathAnalyzer(labels[origin_id], walk_time[origin_id], self.analysis_start_time,
                                       self.analysis_end_time, origin_id, gtfs=G,
                                       transfer_penalty_seconds=self.transfer_penalty_seconds)
        fastest_path_only = False

        self.plot_temporal_distance(njpa, fastest_path_only, ax=ax1, plotname='_temporal_distance', **kwargs)

        self.plot_trajectory_variants_map(njpa, target_id, origin_id, fastest_path_only, ax=ax2,
                                          projection="smopy_axes",
                                          plotname='_trajectory_map', **kwargs)

        self.plot_diversity_table(njpa, ax=ax3, plotname='_diversity_table', **kwargs)
        njpa.plot_journey_graph(ax=ax4, plotname='_journey_graph', **kwargs)

        if not plot_separately:
            plt.tight_layout()
            #plt.show()

            plt.savefig(os.path.join(makedirs(folder), fname + "." + fig_format), format=fig_format, dpi=300)

    def njpa_generator(self, target=None):
        if target is None:
            files = [file for file in self.pickle_generator()]
        else:
            files = [self.find_pickle(target)]
        for file in files:
            store_dict = file
            labels = store_dict["labels"]
            walk_time = store_dict["walk_time"]
            for origin, profile in labels.items():
                if profile:
                    yield NodeJourneyPathAnalyzer(profile, walk_time[origin], self.analysis_start_time,
                                                  self.analysis_end_time, origin, gtfs=self.gtfs,
                                                  transfer_penalty_seconds=self.transfer_penalty_seconds)

    def plot_journeys_to_target(self, target=None):
        leg_list = []
        for njpa in self.njpa_generator(target):
            for leg in njpa.leg_generator(use_leg_stops=True):
                leg_list.append(leg)
        df = pd.DataFrame(leg_list)
        df["trip_type"] = df.apply(lambda row: -1
                                   if row.trip_id == -1
                                   else self.gtfs.get_route_name_and_type_of_tripI(row.trip_id)[1], axis=1)
        df = df.groupby(by=["arr_stop", "dep_stop", "trip_type"]).count()
        df = df.reset_index()
        df = df.rename(index=str, columns={'trip_id': 'count'})

        df = self.gtfs.add_coordinates_to_df(df, stop_id_column="arr_stop", lat_name="to_lat", lon_name="to_lon",
                                             drop_stop_I=False)
        df = self.gtfs.add_coordinates_to_df(df, stop_id_column="dep_stop", lat_name="from_lat", lon_name="from_lon",
                                             drop_stop_I=False)
        df["coord_seq"] = df.apply(lambda row: [(row.from_lon, row.from_lat), (row.to_lon, row.to_lat)], axis=1)

        df = df.assign(zorder=lambda x: route_type_to_zorder(x.trip_type))
        df = df.assign(color=lambda x: route_type_to_color_iterable(x.trip_type))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="smopy_axes")
        ax.set_map_bounds(**self.bbox)
        ax.set_plot_bounds(**self.bbox)
        ax.plot_line_segments(coords=df["coord_seq"].tolist(),
                              linewidths=[x * 0.0002 for x in df["count"].tolist()],
                              colors=df["color"].tolist(), zorders=df["zorder"].tolist(), update=False, alpha=0.4)

        lat, lon = self.gtfs.get_stop_coordinates(target)
        ax.scatter(lon, lat, s=100, c="green", marker="X", zorder=2)
        folder = FIGS_DIRECTORY
        fname = self.feed + "_trips_to_" + str(target) + ".png"
        plt.savefig(os.path.join(folder, fname), format="png", dpi=300)

    def pair_plots(self):
        """
        Correlate n_routes (per stop), avg. headway, speed, number of stops and / or trip departures vs.
        diversity index of the stop.
        GTFS:
        n_routes (per stop
        avg. headway/trip departures
        number of stops within walking distance
        seaborn pair plot
        :return:
        """
        files = self.pickle_generator()
        for store_dict in files:
            labels = store_dict["labels"]
            walk_time = store_dict["walk_time"]
            row_list = []
            for origin, profile in labels.items():
                if profile:
                    njpa = NodeJourneyPathAnalyzer(profile, walk_time[origin], self.analysis_start_time,
                                                   self.analysis_end_time, origin, gtfs=self.gtfs,
                                                   transfer_penalty_seconds=self.transfer_penalty_seconds)
                    row_dict = njpa.get_simple_diversities()
                    row_dict["stop_I"] = origin
                    row_list.append(row_dict)
            df = pd.DataFrame(row_list)
            df = df.drop(['stop_I'], axis=1)
            fig = plt.figure()

            g = pairplot(df.dropna(axis=0), plot_kws={"s": 6})
            plt.show()
