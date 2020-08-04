from collections import Iterable

from pandas import DataFrame
from shapely.geometry import LineString, MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.multipoint import MultiPoint
from shapely.wkt import loads
from geopandas import GeoDataFrame, sjoin
import pandas as pd
from jenks_natural_breaks import classify

import numpy as np

from gtfspy.util import df_to_utm_gdf, utm_to_wgs
from gtfspy.smopy_plot_helper import custom_legend
from research.route_diversity.rd_utils import split_data_frame_list, check_day_start, set_bbox
from research.route_diversity.measures_and_colormaps import *
from research.route_diversity.diversity_settings import *
from research.route_diversity.rd_utils import get_custom_spatial_bounds

# cluster (with jenks natural breaks optimization) to get service level zones
# this enables classification of routes based on relation to urban core(s):
# connects sub center(s) to main center
# connects low density area to sub center/main center
# Other classification parameters could include:
# frequency/capacity, route/corridor
# relation to other routes/corridors:
# shape/circuity


def split_in_parts(coords, to_remove):
    """splits the trajecotories by the stops that are within the centres (indicated in the to_remove vector)"""
    parts = []
    part = []
    prev_truevalue = True
    for coord, truevalue in zip(coords, to_remove):
        if not truevalue:
            part.append(coord)
        elif truevalue and not prev_truevalue:
            parts.append(part)
        prev_truevalue = truevalue
    parts.append(part)
    return 0 if parts == [[]] else parts


def calculate_angles_to_centre(centroid, points):
    def azimuth(point1, point2):
        """azimuth between 2 shapely points (interval 0 - 360)"""
        angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
        return np.degrees(angle) if angle > 0 else np.degrees(angle) + 360

    azimuths = []
    for point in points:
        azimuths.append(azimuth(point, centroid))

    prev_i = sorted(azimuths)[0]
    max_gap = 360 - max(azimuths) + min(azimuths)
    for i in sorted(azimuths)[1:]:
        max_gap = max(i - prev_i, max_gap)
        prev_i = i
    return 360 - max_gap


def get_most_intense_centroid(feed, **kwargs):
    rta = get_route_type_analyzer(feed, **kwargs)
    centroid_gdf = rta.get_max_hub()
    centroid_gdf["geometry"] = centroid_gdf.centroid
    centroid_gdf = centroid_gdf[["geometry"]]
    return centroid_gdf


def get_route_type_analyzer(feed=None, fname_or_conn=None, **kwargs):
    if fname_or_conn is None:
        fname_or_conn = os.path.join(CITIES_DIR, feed, "week" + SQLITE_SUFFIX)
    gtfs = GTFS(fname_or_conn)
    day_start = check_day_start(gtfs, 2)
    start_time = kwargs.pop("start_time", ANALYSIS_START_TIME_DS)
    end_time = kwargs.pop("end_time", ANALYSIS_END_TIME_DS)
    hub_merge_distance_m = kwargs.pop("hub_merge_distance_m", 1000)
    return RouteTypeAnalyzer(day_start, start_time=start_time, end_time=end_time, fname_or_conn=fname_or_conn,
                             hub_merge_distance_m=hub_merge_distance_m, name=feed, use_largest_hub_only=True)


# TODO: RouteTypeAnalyzer class should handle the selection of routes (trips) filling specific criteria, and be a
#  subclass of StaticGTFSStats
#  StaticGTFSStats would perform calculations on said routes (trajectories and segments)
#  RouteTypeAnalyzer would merely help in selecting subsets based on specific criteria
#  RouteTrajectory, subclass of GeoDataFrame, could handle simpler selection of data


class StaticGTFSStats(GTFS):
    def __init__(self, day_start, start_time=0 * 3600, end_time=24 * 3600, name=None,
                 *args,
                 **kwargs):
        super(StaticGTFSStats, self).__init__(*args, **kwargs)
        self.day_start = day_start
        self.start_time = start_time
        self.end_time = end_time
        self.name = name
        self.save_dir = MAPS_DIRECTORY
        self._trajectories = GeoDataFrame()
        self._segments = GeoDataFrame()
        self.trajectory_params = ""
        self.segment_params = ""

    @staticmethod
    def get_param_string(day_start, start_time, end_time, trips=None):
        if trips is not None:
            trips = sorted(trips)
            trips = "_".join(trips)
        else:
            trips = ""
        param_string = "ds-{day_start}_st-{start_time}_et-{end_time}_t-{trips}".format(day_start=day_start,
                                                                                       start_time=start_time,
                                                                                       end_time=end_time,
                                                                                       trips=trips)
        return param_string

    @staticmethod
    def _add_auxiliary_columns(gdf):
        if segment_length_m not in gdf.columns:
            gdf[segment_length_m] = gdf.apply(lambda row: row.geometry.length, axis=1)
        if segment_kilometrage not in gdf.columns:
            gdf[segment_kilometrage] = gdf.apply(lambda row: row.geometry.length * row.n_trips / 1000, axis=1)
        return gdf

    @staticmethod
    def _round_dict_items(d, decimals=3):
        d = {k: round(v, decimals) for k, v in d.items()}
        return d

    def get_segments(self, trip_ids=None, frequency_threshold=0, day_start=None, start_time=None, end_time=None):
        if isinstance(trip_ids, GeoDataFrame) or isinstance(trip_ids, DataFrame):
            trip_ids = trip_ids.tolist()
        day_start = day_start or self.day_start
        start_time = start_time or self.start_time
        end_time = end_time or self.end_time

        param_string = self.get_param_string(day_start, start_time, end_time, trips=trip_ids)
        if param_string != self.segment_params:
            self._segments = self.calculate_trajectory_segments(day_start, start_time, end_time,
                                                                trips=trip_ids,
                                                                ignore_order_of_stop_ids=False)
            self._segments = self._add_auxiliary_columns(self._segments)
            self.segment_params = param_string

        return self._segments.loc[self._segments.n_trips >= frequency_threshold]

    def get_trajectories(self, trip_ids=None, frequency_threshold=0, day_start=None, start_time=None, end_time=None,
                         **kwargs):
        day_start = day_start or self.day_start
        start_time = start_time or self.start_time
        end_time = end_time or self.end_time
        param_string = self.get_param_string(day_start, start_time, end_time)
        if param_string != self.trajectory_params:
            #print("recalculating: fq", frequency_threshold, "start_time", start_time)
            self._trajectories = self.calculate_route_trajectories(day_start, start_time, end_time)
            self._trajectories = self._add_auxiliary_columns(self._trajectories)
            self.trajectory_params = param_string

        if trip_ids is None:
            return self._trajectories.loc[self._trajectories.n_trips >= frequency_threshold]
        else:
            return self._trajectories.loc[self._trajectories.n_trips >= frequency_threshold &
                                          self._trajectories.trip_I.isin(trip_ids)]

    def get_route_length(self, **kwargs):
        return self.get_trajectories(**kwargs)[segment_length_m].sum()

    def get_route_section_length(self, **kwargs):
        return self.get_segments(**kwargs)[segment_length_m].sum()

    def get_route_kilometrage(self, **kwargs):
        return self.get_trajectories(**kwargs)[segment_kilometrage].sum()

    def get_route_overlap(self, **kwargs):
        return self.get_route_length(**kwargs)/self.get_route_section_length(**kwargs)

    def avg_segment_frequency(self, **kwargs):
        return self.get_route_kilometrage(**kwargs) * 1000 / self.get_route_section_length(**kwargs)

    def calculate_prop_stats(self, subset_kwargs, **superset_kwargs):
        route_stats = {prop_length: self.get_route_length(**subset_kwargs)/self.get_route_length(**superset_kwargs),
                       prop_section_length: self.get_route_section_length(
                           **subset_kwargs) / self.get_route_section_length(**superset_kwargs),
                       prop_kilometrage: self.get_route_kilometrage(
                           **subset_kwargs)/self.get_route_kilometrage(**superset_kwargs)}
        return self._round_dict_items(route_stats)

    def calculate_network_stats(self, **kwargs):
        route_stats = {route_length: self.get_route_length(**kwargs),
                       route_section_length: self.get_route_section_length(**kwargs),
                       route_kilometrage: self.get_route_kilometrage(**kwargs)}
        return self._round_dict_items(route_stats)

    def calculate_hour_stats(self, **kwargs):
        hour_stats = {avg_segment_frequency: self.avg_segment_frequency(**kwargs),
                      route_overlap: self.get_route_overlap(**kwargs)}
        return hour_stats


class RouteTypeAnalyzer(StaticGTFSStats):
    def __init__(self, day_start, start_time=0 * 3600, end_time=24 * 3600, hub_merge_distance_m=200, name=None,
                 coords=None, use_largest_hub_only=False, *args, **kwargs):
        super(RouteTypeAnalyzer, self).__init__(day_start, start_time, end_time, name,
                                                *args, **kwargs)

        self.use_largest_hub_only = use_largest_hub_only
        self.day_start = day_start
        self.start_time = start_time
        self.end_time = end_time
        self.hub_merge_distance = hub_merge_distance_m
        self.hub_zones = GeoDataFrame()
        self.crs_wgs = {'init': 'epsg:4326'}
        _, self.crs_utm = df_to_utm_gdf(self.stops())
        self._hubs = None
        self._hubs = self.get_hubs()
        self.name = name
        self.linewidth_multiplier = .05
        self.save_dir = MAPS_DIRECTORY
        if coords:
            self.map_spatial_bounds = set_bbox(**coords)
        else:
            self.map_spatial_bounds = set_bbox(bbox=self.get_bounding_box_by_stops())

    def get_trajectory_segments_wgs(self, **kwargs):
        gdf = self.get_segments(**kwargs)
        return utm_to_wgs(gdf)

    def get_hubs(self):
        """
        get stop list,
        get list of stops with trip_I's
        get trip_I's within buffer
        group by trip_I's
        calculate n_trip_I's
        :param distance:
        :param return_utm:
        :return:
        """

        if self._hubs is None:
            self._hubs = self.calculate_hubs(self.hub_merge_distance, self.day_start, self.start_time,
                                             self.end_time)
        return self._hubs

    def get_percentile_hub(self, percentile):
        """returns the stop_I of the stop closest to the percentile when considering trips_in_area"""
        q_value = self.get_hubs()['trips_in_area'].quantile(percentile)
        targets = self.get_hubs().iloc[(self.get_hubs()['trips_in_area'] - q_value).abs().argsort()[:2]]
        return targets.iloc[0]['stop_I']

    def cluster_using_jenks(self, n_groups=4):
        groups = classify(self._hubs['trips_in_area'].to_numpy(), n_groups)
        print("groups:", groups)
        group_gen = [v for v in groups[1:]]

        self.get_hubs()['jenks'] = np.nan
        self.get_hubs()['jenks_id'] = np.nan

        for i, v in enumerate(reversed(group_gen)):
            self._hubs['jenks'] = self._hubs.apply(lambda row: v if row.trips_in_area <= v else row.jenks,
                                                   axis=1)
            self._hubs['jenks_id'] = self._hubs.apply(lambda row: i if row.trips_in_area <= v else row.jenks_id,
                                                      axis=1)
        return self._hubs

    def create_hub_zones(self, buffer_distance=None):
        if self.hub_zones.empty:
            if not buffer_distance:
                buffer_distance = self.hub_merge_distance
            hubs = self.cluster_using_jenks()
            hubs["buffer"] = hubs["geometry"].buffer(buffer_distance)
            hubs = hubs.set_geometry(hubs["buffer"])
            hubs = hubs.dissolve(by='jenks')
            hubs = hubs.reset_index()
            self.hub_zones = hubs
        return self.hub_zones.copy()

    @staticmethod
    def get_largest_polygon(polygon):
        if isinstance(polygon, MultiPolygon):
            return max(polygon, key=lambda a: a.area)
        else:
            return polygon

    def get_max_hub_zone(self):
        hub_zones = self.create_hub_zones()
        hub_zones["geometry"] = hub_zones.apply(lambda x: self.get_largest_polygon(x.geometry), axis=1)
        return hub_zones.loc[[hub_zones['trips_in_area'].idxmax()]]

    def get_max_hub(self, buffer_distance=None):
        if not buffer_distance:
            buffer_distance = self.hub_merge_distance
        hubs = self.get_hubs()
        hubs = hubs[hubs.index == hubs["trips_in_area"].idxmax()].copy()
        hubs["jenks"] = hubs["trips_in_area"]
        hubs["buffer"] = hubs["geometry"].buffer(buffer_distance)
        hubs = hubs.set_geometry(hubs['buffer'])
        return hubs

    def get_center_zones(self):
        """returns the highest tier hub_zone"""

        if self.use_largest_hub_only:
            return self.get_max_hub()
        else:
            hub_zones = self.create_hub_zones()

            return hub_zones.loc[[hub_zones['trips_in_area'].idxmax()]]

    def get_center_zones_wgs(self):
        center_zones = self.get_center_zones()
        return center_zones.to_crs(self.crs_wgs)

    def get_frequent_corridors(self):
        """Returns the corridors that:
        - are frequent, 10min? or less headway
        - operates whole day
        """

    def get_orbitals(self):
        """Returns the route trajectories that:
         - do not enter the main hub
         - connects to two ore more high capacity routes"""

    def define_feeders_and_orbitals(self, buffer_distance=None, *args, **kwargs):
        """Returns the route trajectories that:
         - do not enter the main hub
         - connects to one and only one high frequency corridor
         - also returns the corridor to which it is connected"""

        if not buffer_distance:
            buffer_distance = self.hub_merge_distance
        route_trajectories = self._check_if_goes_to_hub(*args, **kwargs)

        radials = route_trajectories.loc[route_trajectories.goes_to_hub].copy()
        others = route_trajectories.loc[~route_trajectories.goes_to_hub].copy()
        if others.empty:
            print("All routes are radials!")
            self._trajectories[orbital] = False
            self._trajectories[feeder] = False
            return
        others['lines'] = others['geometry']
        others['points'] = others['geometry'].apply(lambda x: list(x.coords))
        others = split_data_frame_list(others, 'points')
        others['points'] = others['points'].apply(lambda x: Point(x))
        others = others.set_geometry(others['points'])

        radials['points'] = radials['geometry'].apply(lambda x: list(x.coords))
        radials = split_data_frame_list(radials, 'points')
        radials['points'] = radials['points'].apply(lambda x: Point(x))
        radials = radials.set_geometry(radials['points'], crs=self.crs_utm)
        center_zone = self.get_center_zones()

        radials = self._sjoin_and_add_col('point_to_remove', radials, center_zone)

        cols = list(radials)
        agg_dict = {i: lambda x: x.iloc[0] for i in cols}
        agg_dict['points'] = lambda x: list(x)
        agg_dict['point_to_remove'] = lambda x: list(x)
        radials = radials.groupby('trip_Is').agg(agg_dict, axis=1)
        radials = radials.reset_index(drop=True)

        radials['points'] = radials.apply(lambda row: split_in_parts(row.points, row.point_to_remove), axis=1)
        radials = radials.loc[~(radials['points'] == 0)]

        radials = split_data_frame_list(radials, 'points')
        radials = radials.reset_index()
        radials['row_value'] = radials.index
        radials['trip_Is'] = radials.apply(lambda row: str(row.trip_Is) + '_' + str(row.row_value), axis=1)
        radials = radials.drop(['index', 'row_value', 'point_to_remove'], axis=1)

        radials['points'] = radials['points'].apply(lambda x: MultiPoint(x))

        radials = radials.set_geometry(radials['points'])
        radials["buffer"] = radials["geometry"].buffer(buffer_distance)
        radials = radials.set_geometry(radials['buffer'])

        others = self._sjoin_and_add_col('transfer_point', others, radials)
        # TODO: refactor this into smaller methods

        others = others.loc[others.transfer_point]  # TODO: what to do with routes that do not have transfer points
        others['points'] = others['points'].apply(lambda x: x.wkt)

        cols = list(others)

        agg_dict = {i: lambda x: x.iloc[0] for i in cols}

        others = others.groupby(['trip_Is', 'points']).agg(agg_dict, axis=1)
        others = others.reset_index(drop=True)

        others['points'] = others['points'].apply(lambda x: loads(x))
        agg_dict['points'] = lambda x: list(x)
        others = others.groupby(['trip_Is']).agg(agg_dict, axis=1)
        others = others.reset_index(drop=True)
        others['azimuth'] = others.apply(lambda row: calculate_angles_to_centre(center_zone.centroid.iloc[0],
                                                                                row.points), axis=1)
        others[orbital] = False
        others[feeder] = False
        others.loc[others.azimuth < 20, feeder] = True
        others.loc[others.azimuth >= 20, orbital] = True

        self._trajectories = self._trajectories.merge(others[['trip_Is', orbital, feeder]],
                                                      left_on='trip_Is',
                                                      right_on='trip_Is',
                                                      how='left')
        self._trajectories = self._trajectories.fillna(value={orbital: False, feeder: False})
        return self._trajectories

    def define_radials(self, *args, **kwargs):
        """Returns the route trajectories that terminates at the main hub"""
        self._check_if_goes_to_hub(*args, **kwargs)
        self._check_if_endpoints_in_hub(*args, **kwargs)
        self._trajectories[radial_route] = False
        self._trajectories.reset_index(drop=True, inplace=True)
        self._trajectories.loc[self._trajectories.goes_to_hub &
                               (self._trajectories.start_in_hub |
                                self._trajectories.end_in_hub), radial_route] = True
        return self._trajectories

    def define_cross_routes(self, *args, **kwargs):
        """Returns the route trajectories that passes trough the main hub without terminating"""
        self._check_if_endpoints_in_hub(*args, **kwargs)
        self._check_if_goes_to_hub(*args, **kwargs)
        self._trajectories[cross_route] = False
        self._trajectories.loc[self._trajectories.goes_to_hub &
                               ~self._trajectories.start_in_hub & ~self._trajectories.end_in_hub,
                               cross_route] = True
        return self._trajectories

    @staticmethod
    def get_route_type(df, r_type):
        return df.loc[df[r_type]]

    def get_all_route_types(self, *args, **kwargs):
        if self.get_trajectories(*args, **kwargs).empty:
            print("No routes filled criteria")
            return
        self.define_feeders_and_orbitals(*args, **kwargs)

        self.define_cross_routes(*args, **kwargs)

        self.define_radials(*args, **kwargs)
        return self._trajectories

    def get_peak_routes(self):
        """Returns the routes only operated in peak hours"""

    def get_infill_routes(self):
        """Returns routes"""

    def _check_if_endpoints_in_hub(self, *args, **kwargs):
        """Checks if the end points of the trajectories are in the hub"""
        route_trajectories = self.get_trajectories(*args, **kwargs)
        route_columns = list(route_trajectories)
        start_col = 'start_in_hub'
        end_col = 'end_in_hub'
        if start_col not in route_columns and end_col not in route_columns:
            route_columns.append(start_col)
            route_columns.append(end_col)
            centre_zones = self.get_center_zones()
            route_trajectories['first_point'] = route_trajectories["geometry"].apply(lambda x: Point(x.coords[0]))
            route_trajectories['last_point'] = route_trajectories["geometry"].apply(lambda x: Point(x.coords[-1]))
            route_trajectories['old_geom'] = route_trajectories["geometry"]
            route_trajectories = route_trajectories.set_geometry(route_trajectories['first_point'])

            route_trajectories = self._sjoin_and_add_col(start_col, route_trajectories, centre_zones)
            route_trajectories = route_trajectories.set_geometry(route_trajectories['last_point'])
            route_trajectories = self._sjoin_and_add_col(end_col, route_trajectories, centre_zones)
            route_trajectories = route_trajectories.set_geometry(route_trajectories['old_geom'])

            route_trajectories = route_trajectories.drop(['first_point', 'last_point', 'old_geom'], axis=1)
            self._trajectories = route_trajectories
        return self._trajectories

    def _check_if_goes_to_hub(self, *args, **kwargs):
        """Checks if the route intersect the main hub"""
        route_trajectories = self.get_trajectories(*args, **kwargs)

        route_columns = list(route_trajectories)
        new_col = 'goes_to_hub'
        if new_col not in route_columns:
            route_columns.append(new_col)
            centre_zones = self.get_center_zones()

            route_trajectories = sjoin(route_trajectories, centre_zones, how='left')
            route_trajectories[new_col] = True
            route_trajectories.loc[route_trajectories.jenks.isnull(), new_col] = False
            self._trajectories = route_trajectories[route_columns]
        return self._trajectories

    @staticmethod
    def _sjoin_and_add_col(new_col, df_to_retain, join_df):
        join_columns = [i + '_right' if not i == 'geometry' else i for i in list(join_df)]
        join_df.columns = join_columns
        null_col = [i for i in join_columns if not i == 'geometry']
        route_columns = list(df_to_retain)
        route_columns.append(new_col)
        df_to_retain = sjoin(df_to_retain, join_df, how='left')
        df_to_retain[new_col] = True
        df_to_retain.loc[df_to_retain[null_col[0]].isnull(), new_col] = False
        df_to_retain = df_to_retain[route_columns]
        return df_to_retain

    def hub_plot(self, plot_columns='trips_in_area'):
        from mpl_toolkits import axes_grid1

        def add_defined_ax(im, aspect=5, pad_fraction=0.5):
            """Add a vertical color bar to an image plot."""
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            current_ax = plt.gca()
            ax = divider.append_axes("right", size=width, pad=pad)
            plt.sca(current_ax)
            return ax

        cm, norm = get_colormap("trip_density")
        fig, (ax1) = plt.subplots(1, 1, subplot_kw=dict(projection="smopy_axes"))
        ax1.scatter(self._hubs['lon'], self._hubs['lat'], c=self._hubs[plot_columns], s=1, alpha=0.5, cmap=cm,
                    norm=norm)
        ax1.set_plot_bounds(**self.map_spatial_bounds)
        ax1.add_scalebar()

        ax2 = add_defined_ax(ax1)
        plt.axes(ax2)
        n, bins, patches = plt.hist(self._hubs[plot_columns], 25, color='green', range=[norm.vmin, norm.vmax],
                                    orientation="horizontal")
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # scale values to interval [0,1]
        col = bin_centers - min(bin_centers)
        col /= max(col)

        for c, p in zip(col, patches):
            plt.setp(p, 'facecolor', cm(c))
        ax2.tick_params(labelsize=15)

        plt.savefig(self.save_dir+self.name+"_hubplot.png",
                    format="png",
                    bbox_inches='tight',
                    dpi=900)

    def cross_route_plot(self):
        trajectories = self.define_cross_routes()
        bottom = trajectories.loc[trajectories.cross_route & trajectories.n_trips >= 2].plot(color='black')
        trajectories.loc[trajectories.cross_route & trajectories.n_trips >= 2].plot(ax=bottom, color='red')
        plt.show()

    def radial_route_plot(self):
        trajectories = self.define_radials()
        center_zone = self.get_center_zones()
        bottom = center_zone.plot(color='white', edgecolor='blue')

        bottom = trajectories.loc[trajectories.radial_route & (trajectories.n_trips >= 2)].plot(ax=bottom,
                                                                                                color='black')
        trajectories.loc[trajectories.radial_route & (trajectories.n_trips >= 2)].plot(ax=bottom, color='red')

        plt.show()

    def plot_route_category_maps(self, annotate=False, show_map=False, split_maps=False, plot_center=False,
                                 plot_legend=False, *args, **kwargs):
        """"""
        plot_frequency_threshold = kwargs.pop("plot_frequency_threshold")
        trajectories = self.get_all_route_types(*args, **kwargs)
        if trajectories.empty:
            return
        ax = None
        round_to_save = len(ROUTE_TYPE_COLORS.values())
        for i, (r_type, color) in enumerate(ROUTE_TYPE_COLORS.items()):
            if split_maps or i == 0:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw=dict(projection="smopy_axes"))
                ax = self.plot_grey_base(ax=ax, trip_ids=trajectories["trip_Is"].tolist())
                if plot_center:
                    ax = self.plot_center_zone(ax)
                ax.set_plot_bounds(**self.map_spatial_bounds)
                ax.add_scalebar(frameon=False, location="lower right")
                ax.axes.get_xaxis().set_ticks([])
                ax.axes.get_yaxis().set_ticks([])
            if not trajectories.loc[trajectories[r_type]].empty:
                trip_ids = self.get_route_type(self.get_trajectories(
                    frequency_threshold=plot_frequency_threshold), r_type).trip_Is
                if not trip_ids.empty:
                    gdf = self.get_trajectory_segments_wgs(trip_ids=trip_ids.tolist())
                    ax.plot_line_segments(coords=gdf["coord_seq"].tolist(),
                                          linewidths=[x * self.linewidth_multiplier for x in gdf["n_trips"].tolist()],
                                          colors=[color]*len(gdf.index), zorders=[10]*len(gdf.index), update=False)

                    if annotate:
                        tc = self.calculate_terminus_coords(trips=trip_ids)
                        for j in tc.itertuples():
                            plt.annotate(s=j.name, xy=(j.lon, j.lat))
                else:
                    print("No ", r_type)
                    if split_maps:
                        plt.clf()
                        continue

            else:
                print("No ", r_type)
            if split_maps or i == round_to_save-1:
                if plot_legend:
                    fig = plt.gcf()
                    rtc = [{"label": l, "color": c, "type": "line"} for l, c in ROUTE_TYPE_COLORS.items()]
                    custom_legend(fig, rtc,
                                  **{"loc": 8, "ncol": 4, "fontsize": 6, "markersize": 6})
                if show_map:
                    plt.show()
                else:
                    fname = r_type if split_maps else "routes"
                    self.figure_saver(plt, fname, **kwargs)

    def figure_saver(self, plot, basename, **kwargs):
        suffix = kwargs.pop("suffix")

        _format = kwargs.pop("format", "png")
        dpi = kwargs.pop("dpi", 900)
        plot.tight_layout()
        plot.savefig(self.save_dir+self.name+"_"+basename+suffix+"."+_format,
                     format=_format,
                     #bbox_inches='tight',
                     dpi=dpi)

    def plot_center_zone(self, ax):
        # plot central hub
        center_zone = self.get_center_zones_wgs()
        center_zones = center_zone["geometry"].iloc[0].boundary
        if isinstance(center_zones, LineString):
            center_zones = [center_zones]
        for line in center_zones:
            x, y = line.coords.xy
            ax.plot(x, y)
        return ax

    def plot_trips(self, plot_name, gdf=None, trip_Is=None, color="red", return_ax=False, plot_grey_base=True, ax=None):

        assert gdf is not None or trip_Is is not None
        if gdf is None:
            gdf = self.get_trajectory_segments_wgs(trip_ids=trip_Is)

        if plot_grey_base:
            ax = self.plot_grey_base()
        elif not ax:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="smopy_axes"))
            ax.set_map_bounds(**self.map_spatial_bounds)

        ax.plot_line_segments(coords=gdf["coord_seq"].tolist(),
                              linewidths=[x * self.linewidth_multiplier for x in gdf["n_trips"].tolist()],
                              colors=[color]*len(gdf.index), zorders=[1]*len(gdf.index), update=False)

        ax.set_plot_bounds(**self.map_spatial_bounds)
        ax.add_scalebar(frameon=False, location="lower right")
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        if return_ax:
            return ax
        else:
            plt.savefig(self.save_dir + self.name + "_" + plot_name + ".png",
                        format="png",
                        bbox_inches='tight',
                        dpi=900)

    def plot_grey_base(self, ax=None, trip_ids=None):
        """
        Get all stop segments for the time period defined for object and plot on map
        :return:
        """
        gdf = self.get_trajectory_segments_wgs(trip_ids=trip_ids)
        if not ax:
            fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="smopy_axes"))

        ax.set_map_bounds(**self.map_spatial_bounds)
        ax.plot_line_segments(coords=gdf["coord_seq"].tolist(),
                              linewidths=[x * self.linewidth_multiplier for x in gdf["n_trips"].tolist()],
                              colors=['grey'] * len(gdf.index), zorders=[1] * len(gdf.index), update=False)
        return ax

    def route_type_proportions(self):
        """"""
        trajectories = self.get_all_route_types()
        segment_kmts = {}

        for r_type, color in ROUTE_TYPE_COLORS.items():
            # name = self.get_location_name()
            if not trajectories.loc[trajectories[r_type]].empty:
                gdf = self.get_segments(trip_ids=trajectories.loc[trajectories[r_type]].trip_Is)
                segment_kmts[r_type] = sum(gdf["segment_kmt"])

            else:
                print("No ", r_type)
        segment_prop = {k: round(v / sum(segment_kmts.values()), 2) for k, v in segment_kmts.items()}
        print(segment_prop)
