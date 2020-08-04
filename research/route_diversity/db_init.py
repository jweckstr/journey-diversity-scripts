import os
import subprocess
from gtfspy.gtfs import GTFS
from sqlite3 import IntegrityError

from gtfspy.osm_transfers import compute_stop_to_stop_osm_walk_distances_python
from research.route_diversity.diversity_settings import *
from gtfspy.filter import FilterExtract
from gtfspy.import_gtfs import import_gtfs
from gtfspy.import_validator import ImportValidator
from gtfspy.aggregate_stops import merge_stops_tables_multi, remove_unmatching_stops_multi
from shutil import copyfile


def prepare_dbs():
    for city, feeds in CITIES.items():
        print(city, feeds)
        if city in ["indianapolis"]:  # ["houston", "columbus", "baltimore"]: #== "houston":  # not in ["amsterdam", "auckland"]:
            print("copying files")
            for x in feeds:
                if False:
                    subprocess.call(["python3", "extract_pipeline.py", "gtfs_only", os.path.join(CITIES_DIR, x, "gtfs.zip")], cwd=r'gtfs_data_pipeline/extracts/')
                dirs = next(os.walk(os.path.join(CITIES_DIR, x)))[1]
                for dir in dirs:
                    file_path = os.path.join(CITIES_DIR, x, dir, "week.sqlite")
                    if os.path.isfile(file_path):
                        if True:
                            copyfile(file_path, os.path.join(CITIES_DIR, x, "_week.sqlite"))
            d = CUTOFF_DISTANCE
            print(city)
            conns = [GTFS(os.path.join(CITIES_DIR, x, "_week.sqlite")) for x in feeds]
            if True:
                merge_stops_tables_multi(conns, threshold_meters=STOP_MERGE_DISTANCE)

            paths = [os.path.join(CITIES_DIR, x, "week.sqlite") for x in feeds]
            remove_unmatching_stops_multi(conns, paths, d)

            conns = [GTFS(os.path.join(CITIES_DIR, x, "week.sqlite")) for x in feeds]
            sd_table = None
            """
            stops, crs_utm = df_to_utm_gdf(gtfs.stops())
            polygons = create_grid_tesselation(*stops.total_bounds, height=tesselation_distance,
                                               width=tesselation_distance,
                                               random_seed=random_seed)
            """
            # Define extents,
            for conn in conns:
                sd_table = conn.recalculate_stop_distances2(d, remove_old_table=True, return_sd_table=True, sd_table=sd_table)

    """
    lm_stops = G_lm.execute_custom_query_pandas("SELECT * FROM stops")
    old_stops = G_old.execute_custom_query_pandas("SELECT * FROM stops")
    lm_stops_set = set(lm_stops["stop_I"])
    old_stops_set = set(old_stops["stop_I"])
    print(lm_stops_set)
    print(old_stops_set)
    print("stops not in old:", lm_stops_set - old_stops_set)
    print("stops not in old:", old_stops_set - lm_stops_set)

    print("run walk distance routing")
    """
    """
    add_walk_distances_to_db_python(G_lm, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)
    add_walk_distances_to_db_python(G_old, OSM_DIR, cutoff_distance_m=CUTOFF_DISTANCE)
    """
    """
    for fn in [GTFS_DB_LM + SQLITE_SUFFIX, GTFS_DB_OLD + SQLITE_SUFFIX]:
        subprocess.call(["java", "-jar",
                         "gtfspy/java_routing/target/transit_osm_routing-1.0-SNAPSHOT-jar-with-dependencies.jar",
                         "-u",
                         os.path.join(GTFS_DB_WORK_DIR, fn),
                         "-osm",
                         OSM_DIR,
                         "--tempDir", "/tmp/"])
    """


prepare_dbs()
