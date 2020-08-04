import os
import pandas

from gtfspy.gtfs import GTFS
from gtfspy.util import ut_to_utc_datetime

DIRPATH = os.getcwd()

CITIES2 = {#"helsinki": ["helsinki_2014", "helsinki_2018"]#,
          #"baltimore": ["baltimore_2017", "baltimore_2018"],
          "indianapolis": ["indianapolis_2016", "indianapolis_2019"]
          }
CITIES = {"amsterdam": ["amsterdam_2017", "amsterdam_2018"],
          "auckland": ["auckland_2016", "auckland_2018"],
          "austin": ["austin_2017", "austin_2018"],
          "baltimore": ["baltimore_2017", "baltimore_2018"],
          "columbus": ["columbus_2017", "columbus_2018"],
          "helsinki": ["helsinki_2014", "helsinki_2018"],
          "houston": ["houston_2015", "houston_2018"],
          "indianapolis": ["indianapolis_2016", "indianapolis_2019"],
          "wellington": ["wellington_2017", "wellington_2018"]
          }

CITIES3 = {"aalborg": ["aalborg"],
           "aarhus": ["aarhus"],
           "amsterdam": ["amsterdam"],
           "bergen": ["bergen"],
           "berlin": ["berlin"],
           "copenhagen": ["copenhagen"],
           "fredrikstad-sarpsborg": ["fredrikstad-sarpsborg"],
           "gothenburg": ["gothenburg"],
           "helsingborg": ["helsingborg"],
           "helsinki": ["helsinki_2014", "helsinki_2018"],
           "jonkoping": ["jonkoping"],
           "jyvaskyla": ["jyvaskyla"],
           "kuopio": ["kuopio"],
           "lahti": ["lahti"],
           "linkoping": ["linkoping"],
           "malmo": ["malmo"],
           "norrkoping": ["norrkoping"],
           "odense": ["odense"],
           "orebro": ["orebro"],
           "oslo": ["oslo"],
           "oulu": ["oulu"],
           "porsgrunn-skien": ["porsgrunn-skien"],
           "stavanger": ["stavanger"],
           "stockholm": ["stockholm"],
           "tampere": ["tampere"],
           "trondheim": ["trondheim"],
           "turku": ["turku"],
           "umea": ["umea"],
           "uppsala": ["uppsala"],
           "vasteras": ["vasteras"]
          }
# TODO: possible cities to include:
#  Salt Lake city (aug 2019 overhaul, UVX BRT aug 2018) --too big network spatially
#  Anchorage (Oct 2017)

COUNTRIES = {
    "Germany": "Europe",
    "Switzerland": "Europe",
    "Italy": "Europe",
    "Hungary": "Europe",
    "Argentina": "South America",
    "Ireland": "Europe",
    "Spain": "Europe",
    "Czechia": "Europe",
    "Chile": "South America",
    "Austria": "Europe",
    "Belgium": "Europe",
    "Brazil": "South America",
    "France": "Europe",
    "Australia": "Oceania",
    "Canada": "North America",
    "USA": "North America"}

def year_index(feed):
    parts = feed.split("_")
    return CITIES[parts[0]].index(feed)


LINESTYLES = ['-', '--', '-.', ':']


FEED_LIST = []
for key, value in CITIES.items():
    FEED_LIST += value

ALL_FEEDS = FEED_LIST

string_to_add = "" #"intro_" #"hsl_ik_" "plannord_"

SQLITE_SUFFIX = ".sqlite"
PICKLE_SUFFIX = ".pickle"
RESULTS_PREFIX = "results_"
PSEUDO_STOP_FNAME = "pseudo_stops.csv"
TRAVEL_IMPEDANCE_STORE_FNAME = "travel_impedance_store"
BASE_DIR = DIRPATH + "/scratch/diversity_data/"
CITIES_DIR = BASE_DIR + string_to_add + "cities/"

ROUTEMAPS_DIRECTORY = BASE_DIR + string_to_add + "routemaps/"
FIGS_DIRECTORY = BASE_DIR + string_to_add + "multifigs/"
MAPS_DIRECTORY = BASE_DIR + string_to_add + "maps/"
MAPS_CATEGORICAL_DIRECTORY = MAPS_DIRECTORY + "categorical/"

DEBUG_FIGS = BASE_DIR + string_to_add + "debug/"
S2S_DIRECTORY = BASE_DIR + string_to_add + "stop2stop/"
SENSITIVITY_DIRECTORY = BASE_DIR + string_to_add + "sensitivity_analysis/"

DIVERSITY_TO_PUBLISH_CSV = DIRPATH + "/research/route_diversity/"+string_to_add+"diversity_to_publish.csv"

TRAVEL_IMPEDANCE_STORE_PATH = os.path.join(CITIES_DIR, TRAVEL_IMPEDANCE_STORE_FNAME + SQLITE_SUFFIX)


GTFS_DB_FNAME = "week"+SQLITE_SUFFIX
JOURNEY_DB_FNAME = RESULTS_PREFIX+SQLITE_SUFFIX
ROUTING_PICKLE_DIR = "routing_pickles"
DIVERSITY_PICKLE_DIR = "diversity_pickles"
DIVERSITY_PICKLE_FNAME = "diversity_table.pkl"
NETWORK_STATS_PICKLE_FNAME = CITIES_DIR + "stats_table.pickle"


# FEED SETTINGS:
CUTOFF_DISTANCE = 1000
STOP_MERGE_DISTANCE = 20

# ROUTING PARAMETERS:
TRANSFER_MARGIN = 180
TRANSFER_PENALTY_SECONDS = 180
WALK_SPEED = round(70.0 / 60, 2)

PICKLE = True
TRACK_ROUTE = True
TRACK_VEHICLE_LEGS = True
TRACK_TIME = True

ROUTING_START_TIME_DS = 7 * 3600
ROUTING_END_TIME_DS = 12 * 3600
ANALYSIS_START_TIME_DS = 7 * 3600
ANALYSIS_END_TIME_DS = 9 * 3600
day_start_add = 24*3600

PLOT_DIMENSION_DISTANCE = 20000
TESSELATION_DISTANCE = 2000
HUBS_DISTANCE = 500
SAMPLE_SIZE = 5
SAMPLE_FRACTION = 0.001
RANDOM_SEED = 12354

day = "day"
peak = "peak"

time_dict = {day: {"start_time": 12*3600, "end_time": 13*3600},
             peak: {"start_time": 7*3600, "end_time": 8*3600}}


def get_to_publish_row(city):
    """
    Returns
    -------
    pandas.DataFrame
    """

    dtypes = {"publishable": object,
              "license_files": str,
              "lat": float,
              "lon": float,
              "buffer": float,
              "feeds": str,
              "extract_start_date": str,
              "download_date": str}
    to_publish_df = pandas.read_csv(DIVERSITY_TO_PUBLISH_CSV, sep=",", keep_default_na=True, dtype=dtypes)
    to_publish_df.license_files.fillna("")
    return to_publish_df.loc[to_publish_df['id'] == city].to_dict(orient='records')[0]


def get_feed_dict(city):
    print("current directory is : " + DIRPATH)
    foldername = os.path.basename(DIRPATH)
    print("Directory name is : " + foldername)
    G = GTFS(os.path.join(CITIES_DIR, city, GTFS_DB_FNAME))
    day_start, _ = G.get_day_start_ut_span()
    desired_weekday = 2  # wednesday
    tz = G.get_timezone_pytz()
    print(ut_to_utc_datetime(day_start, tz).weekday())
    weekday = ut_to_utc_datetime(day_start, tz).weekday()
    day_start += day_start_add * (9-weekday if weekday > desired_weekday else 2-weekday)
    print("day start:", day_start)
    print("day start weekday:", ut_to_utc_datetime(day_start, tz).weekday())
    to_publish_row = get_to_publish_row(city)
    city_coords = {'distance': PLOT_DIMENSION_DISTANCE, 'lat': to_publish_row['lat'], 'lon': to_publish_row['lon']}
    return {"gtfs": G,
            "routing_label_pickle_dir": os.path.join(CITIES_DIR, city, ROUTING_PICKLE_DIR),
            "diversity_pickle_dir": os.path.join(CITIES_DIR, city, DIVERSITY_PICKLE_DIR),
            "day_start": day_start,
            "routing_start_time": day_start + ROUTING_START_TIME_DS,
            "routing_end_time": day_start + ROUTING_END_TIME_DS,
            "analysis_start_time": day_start + ANALYSIS_START_TIME_DS,
            "analysis_end_time": day_start + ANALYSIS_END_TIME_DS,
            "feed": city,
            "walk_speed": WALK_SPEED,
            "transfer_margin": TRANSFER_MARGIN,
            "track_vehicle_legs": TRACK_VEHICLE_LEGS,
            "city_coords": city_coords,
            "transfer_penalty_seconds": TRANSFER_PENALTY_SECONDS
            }

