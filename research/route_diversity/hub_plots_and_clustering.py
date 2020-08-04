import os
import matplotlib.pyplot as plt

from research.route_diversity.static_route_type_analyzer import RouteTypeAnalyzer
from research.tis_paper.settings import ROUTEMAPS_DIRECTORY


def stop_hubs(self, percentile):
    """
    Selects the nth percentile stop based on the stop hubs concept and makes routing to that stop
    :param percentile:
    :return:
    """
    nh = RouteTypeAnalyzer(self.G, self.day_start)

    target = nh.get_percentile_hub(percentile)
    pickle_subdir = ""
    print("target: ", target)
    if not os.path.isfile(os.path.join(self.routing_label_pickle_dir, str(pickle_subdir), str(target) + ".pickle")):
        self.loop_trough_targets_and_run_routing_with_route([target], pickle_subdir)
    self.plot_multiple_measures_on_maps(target)


def hub_plot(self, name):
    # nh.calculate_overlap_rate()
    from gtfspy.mapviz import plot_route_network_from_gtfs, plot_all_stops
    plt.figure()
    ax = plot_route_network_from_gtfs(self.G, use_shapes=True)
    plt.savefig(os.path.join(ROUTEMAPS_DIRECTORY, "routemap_" + name + ".png"),
                format="png", dpi=300)
    plt.figure()
    ax = plot_all_stops(self.G)
    plt.savefig(os.path.join(ROUTEMAPS_DIRECTORY, "stopmap_" + name + ".png"),
                format="png", dpi=300)

    # nh = RouteTypeAnalyzer(self.G, self.day_start, start_time=7 * 3600, end_time=8 * 3600, name=name, coords=self.city_coords)

    # nh.plot_route_maps()
    # nh.hub_plot()

    # nh.get_feeders()
    # nh.radial_route_plot()
    # nh.get_cross_routes()
    # nh.get_center_zones()
    # nh._check_if_endpoints_in_hub()
    # nh.get_radials()
    # nh.get_route_trajectories()
    # nh.create_hub_zones()


def cluster_routes(self):
    """
    Creates a dendrogram of the routes based on stops used
    :return:
    """
    from scipy import sparse
    from scipy.cluster.hierarchy import linkage, dendrogram
    from sklearn.metrics.pairwise import cosine_similarity
    from gtfspy.aggregate_stops import _cluster_stops_multi
    gtfs = self.G
    query = """WITH 
            a AS (SELECT routes.*, stop_I FROM routes, trips, stop_times
            WHERE routes.route_I = trips.route_I AND trips.trip_I = stop_times.trip_I
            GROUP BY routes.route_I, stop_I)

            SELECT * FROM a
            """
    df = gtfs.execute_custom_query_pandas(query)
    stops_df = gtfs.stops()
    stops_df = _cluster_stops_multi(stops_df, 100)
    to_new_stop_id = \
        {stop_id: new_stop_id for stop_id, new_stop_id in zip(stops_df["stop_I"], stops_df["stop_pair_I"])}

    df["stop_I"] = df["stop_I"].apply(lambda x: to_new_stop_id[x])
    df = df.drop_duplicates()
    id_to_route = {i: route for i, route in enumerate(df["route_I"].unique())}
    route_to_id = {route: i for i, route in id_to_route.items()}
    id_to_stop = {i: route for i, route in enumerate(df["stop_I"].unique())}
    stop_to_id = {route: i for i, route in id_to_stop.items()}

    id_to_route_name = [df.loc[df['route_I'] == id_to_route[x], 'name'].iloc[0] for x in id_to_route.keys()]
    print(id_to_route_name)
    row = [route_to_id[x] for x in df["route_I"]]
    col = [stop_to_id[x] for x in df["stop_I"]]

    val = [1] * len(col)
    mat_coo = sparse.coo_matrix((val, (row, col)))
    print(mat_coo.toarray())

    dist_matrix = cosine_similarity(mat_coo, Y=None, dense_output=True)
    # Y = pdist(np.asarray(mat_coo.toarray()).shape, 'cosine')
    clus = linkage(dist_matrix, metric="cosine")
    fig = plt.figure()
    dendrogram(clus, labels=id_to_route_name, leaf_font_size=2.5)
    plt.show()

    """
    def run_plot_hubs(args):
        plots the measures of the stop closest to given percentile of the given city
        city = args[0]
        percentile = float(args[1])
        feed_dict = get_feed_dict(city)
        ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
        ra_pipeline.stop_hubs(percentile)



    def run_get_hubs(args):
        plots the hubs of a city
        city = args[0]
        if city == 'all':
            dirs = [x[1] for x in os.walk(DATA_DIR)]
            print(dirs[0])
            dirs = ALL_FEEDS
        else:
            dirs = [city]
        for city in dirs:
            print(city)
            feed_dict = get_feed_dict(city)
            ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
            ra_pipeline.hub_plot(city)


    def run_clustering(city, **kwargs):

        print(city)
        feed_dict = get_feed_dict(city)
        feed_dict.update(**kwargs)
        ra_pipeline = JourneyAnalyzePipeline(**feed_dict)
        ra_pipeline.cluster_routes()
    
    elif cmd == "plot_hub":
        run_plot_hubs(args)
    elif cmd == "get_hubs":
        run_get_hubs(args)
    elif cmd == "generic":
        run_generic(args[0])
    """