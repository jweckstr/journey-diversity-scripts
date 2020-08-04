import gc
import pickle

import pandas as pd

from gtfspy.networks import walk_transfer_stop_to_stop_network
from gtfspy.routing.connection import Connection
from gtfspy.routing.journey_path_analyzer import NodeJourneyPathAnalyzer
from gtfspy.routing.multi_objective_pseudo_connection_scan_profiler import MultiObjectivePseudoCSAProfiler
from gtfspy.util import makedirs
from gtfspy.util import timeit
from research.route_diversity.diversity_settings import *
from research.route_diversity.rd_utils import get_custom_spatial_bounds

"""
Loops trough a set of target nodes, runs routing for them and stores results in a database
Pipeline
pyfile1
1. Create database, based on the parameters set in settings
2. Divide origin nodes into n parts
3. Run all_to_all.py
4a. pickle
or
4b. direct to db
5. Create indicies once everything is finished
all_to_all.py

srun --mem=1G --time=0:10:00 python3 research/westmetro_paper/scripts/all_to_all.py run_preparations

srun --mem=6G --time=2:00:00 python3 research/westmetro_paper/scripts/all_to_all.py to_db
"""


class JourneyAnalyzePipeline:
    def __init__(self, gtfs, day_start, routing_start_time, routing_end_time, analysis_start_time,
                 analysis_end_time, routing_label_pickle_dir, diversity_pickle_dir, feed, city_coords, **kwargs):

        self.gtfs = gtfs
        self.tz = self.gtfs.get_timezone_name()
        self.day_start = day_start
        self.routing_start_time = routing_start_time
        self.routing_end_time = routing_end_time
        self.analysis_start_time = analysis_start_time
        self.analysis_end_time = analysis_end_time
        self.routing_label_pickle_dir = routing_label_pickle_dir
        self.diversity_pickle_dir = diversity_pickle_dir

        self.feed = feed
        self.city_coords = city_coords
        self.bbox = get_custom_spatial_bounds(**city_coords)
        self.performance_measure_dict = {}
        self.performance_measure_df = pd.DataFrame()

        self.transfer_margin = kwargs.pop("transfer_margin", TRANSFER_MARGIN)
        self.walk_speed = kwargs.pop("walk_speed", WALK_SPEED)
        self.walk_distance = kwargs.pop("walk_distance", CUTOFF_DISTANCE)
        self.track_vehicle_legs = kwargs.pop("track_vehicle_legs", TRACK_VEHICLE_LEGS)
        self.track_time = kwargs.pop("track_time", TRACK_TIME)
        self.track_route = kwargs.pop("track_route", TRACK_ROUTE)
        self.run_if_missing = kwargs.pop("run_if_missing", False)
        self.transfer_penalty_seconds = kwargs.pop("transfer_penalty_seconds", 0)
        self.kwargs = kwargs
        self.diversity_table_fname = os.path.join(CITIES_DIR, self.feed, DIVERSITY_PICKLE_FNAME)

    def routing_id(self, target=""):
        routing_string = "t-{target}_st-{rst}_et-{ret}_tm-{tm}_ws-{ws}_wd-{wd}".format(target=target,
                                                                                       rst=self.routing_start_time,
                                                                                       ret=self.routing_end_time,
                                                                                       tm=self.transfer_margin,
                                                                                       ws=self.walk_speed,
                                                                                       wd=self.walk_distance)
        return routing_string

    def get_pickle_name(self, target):
        return self.routing_id(target) + PICKLE_SUFFIX

    def get_name_year(self):
        (name, year) = self.feed.split("_")
        return name, year

    @staticmethod
    def get_target_from_routing_id(routing_id):
        return int(routing_id.split("_")[0].split("-")[1])

    def get_all_events(self):
        print("Retrieving transit events")
        connections = []
        for e in self.gtfs.generate_routable_transit_events(start_time_ut=self.routing_start_time,
                                                            end_time_ut=self.routing_end_time):
            connections.append(Connection(int(e.from_stop_I),
                                          int(e.to_stop_I),
                                          int(e.dep_time_ut),
                                          int(e.arr_time_ut),
                                          int(e.trip_I),
                                          int(e.seq)))
        assert (len(connections) == len(set(connections)))
        print("scheduled events:", len(connections))
        print("Retrieving walking network")
        net = walk_transfer_stop_to_stop_network(self.gtfs, max_link_distance=self.walk_distance)
        print("net edges: ", len(net.edges()))
        return net, connections

    @timeit
    def loop_trough_targets_and_run_routing_with_route(self, targets, slurm_array_i):
        for profiles, target in self.routing_generator(targets):
            self._pickle_routing_labels(profiles, target, slurm_array_i)

    def routing_generator(self, targets):
        net, connections = self.get_all_events()
        csp = None

        for target in targets:
            print("target: ", target)
            if csp is None:

                csp = MultiObjectivePseudoCSAProfiler(connections,
                                                      target,
                                                      walk_network=net,
                                                      end_time_ut=self.routing_end_time,
                                                      transfer_margin=self.transfer_margin,
                                                      start_time_ut=self.routing_start_time,
                                                      walk_speed=self.walk_speed,
                                                      verbose=True,
                                                      track_vehicle_legs=self.track_vehicle_legs,
                                                      track_time=self.track_time,
                                                      track_route=self.track_route,
                                                      distance_type="d")
            else:
                csp.reset([target])
            csp.run()

            profiles = csp.stop_profiles
            gc.collect()
            yield profiles, target

    def _pickle_diversity_table(self, df, target, pickle_subdir=""):
        pickle_path = makedirs(os.path.join(self.diversity_pickle_dir, str(pickle_subdir)))
        pickle_path = os.path.join(pickle_path, self.get_pickle_name(target))
        df.to_pickle(pickle_path)

    @timeit
    def _pickle_routing_labels(self, profiles, target, pickle_subdir=""):
        pickle_path = makedirs(os.path.join(self.routing_label_pickle_dir, str(pickle_subdir)))
        pickle_path = os.path.join(pickle_path, self.get_pickle_name(target))
        labels = dict((k, v.get_final_optimal_labels()) for (k, v) in profiles.items())
        walk_time = dict((k, v.get_walk_to_target_duration()) for (k, v) in profiles.items())
        store_dict = {"labels": labels, "walk_time": walk_time, "target": target}
        pickle.dump(store_dict, open(pickle_path, 'wb'), -1)
        gc.collect()

    def get_list_of_stops(self, where=''):
        df = self.gtfs.execute_custom_query_pandas("SELECT stop_I FROM stops " + where + " ORDER BY stop_I")
        return df

    def diversity_pickle_generator(self, targets):
        for root, dirs, files in os.walk(self.diversity_pickle_dir):
            for target_file in files:
                target = self.get_target_from_routing_id(target_file)
                if target in targets and self.check_pickle_validity(target_file, target=target):
                    yield pd.read_pickle(os.path.join(root, target_file))

    def pickle_generator(self):
        for root, dirs, files in os.walk(self.routing_label_pickle_dir):
            for target_file in files:
                yield pickle.load(open(os.path.join(root, target_file), 'rb'))

    def check_pickle_validity(self, file, target=None):
        if target:
            return self.get_pickle_name(target) == file
        else:
            return self.routing_id().split("_", maxsplit=1)[1] + PICKLE_SUFFIX in file

    def pickle_generator_with_targets(self, targets):
        for target in targets:
            yield self.find_pickle(target)

    def find_pickle(self, target=None):
        for root, dirs, files in os.walk(self.routing_label_pickle_dir):
            for target_file in files:
                if self.get_pickle_name(target) == target_file:
                    return pickle.load(open(os.path.join(root, target_file), 'rb'))

        print("Data for target", target, "missing in", self.routing_label_pickle_dir)
        if not self.run_if_missing:
            exit()
        elif self.run_if_missing:
            pass
        elif input("Run routing for missing target? y/n")[0] == "n":
            exit()
        print("starting routing...")
        self.loop_trough_targets_and_run_routing_with_route([target], "testing")
        return self.find_pickle(target)

    @timeit
    def calculate_diversity(self, labels, target, walk_time, measures=None):
        row_list = []
        for origin, profile in labels.items():
            if profile:
                njpa = NodeJourneyPathAnalyzer(profile, walk_time[origin], self.analysis_start_time,
                                               self.analysis_end_time, origin, gtfs=self.gtfs,
                                               transfer_penalty_seconds=self.transfer_penalty_seconds)

                row_dict = njpa.get_simple_diversities(measures=measures)
                row_dict["from_stop_I"] = origin
                row_list.append(row_dict)
        df = pd.DataFrame(row_list)
        df["to_stop_I"] = target
        return df

    def calculate_diversity_for_target(self, target, measures=None, **kwargs):
        store_dict = self.find_pickle(target)
        labels = store_dict["labels"]
        walk_time = store_dict["walk_time"]
        df = self.calculate_diversity(labels, target, walk_time, measures)
        self._pickle_diversity_table(df, target)
        return self.gtfs.add_coordinates_to_df(df, drop_stop_I=True, **kwargs)

    @timeit
    def run_everything_to_diversity(self, targets):
        for profiles, target in self.routing_generator(targets):
            self._pickle_routing_labels(profiles, target)
            labels = dict((k, v.get_final_optimal_labels()) for (k, v) in profiles.items())
            walk_time = dict((k, v.get_walk_to_target_duration()) for (k, v) in profiles.items())
            df = self.calculate_diversity(labels, target, walk_time, measures=None)
            self._pickle_diversity_table(df, target)
            gc.collect()

        # CHECKLIST FOR CASES WHERE THERE ARE A LOT OF INF-VALUES:
        # 1: check that routing time range is enough for trips to finnish
        # 2: check that analysis time range is ending long enough before routing time so that the analysis time covers
        # the period where there are complete trips
        # 3: use s2s plots to analyze potential problems

        # TODO: Check which journeys are considered when counting the diversity measures
        # TODO: calculate average measures only on O-D:s where mean temporal distance is finite or below a threshold?
        # TODO: how to handle areas with infrequent service?
        # TODO: maybe a comparison of unavailable trips too->
        #  add a column where the number of unavailable connections are added
        # TODO: test weighted average (weighed by service availability)

        # VISUALIZATIONS:
        # TODO: plot samples
        # TODO: pair plot with averages
        # TODO: heatmap: original value vs. change
        # TODO: divide the O-D connections in four categories:
        # 1) high frequency - low diversity,
        # 2) high frequency - high diversity,
        # 3) low frequency - low diversity,
        # 4) low frequency - high diversity
        # This could be visualized in plots
        # Then propose a global measure where we compare categories 1 & 4
        # With changing thresholds for frequency and diversity this could be further plotted into a distribution
