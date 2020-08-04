import itertools
import os
import sys

import pandas
from geopandas import GeoDataFrame, sjoin
from pandas import DataFrame
from shapely.geometry import Polygon
import numpy as np
import math
from gtfspy.util import wgs84_width, wgs84_height, df_to_utm_gdf, ut_to_utc_datetime, makedirs, wgs84_distance
import random
import matplotlib.pyplot as plt


def split_data_frame_list(df, target_column, separator=None):
    """ df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.
    """
    row_accumulator = []

    def split_list_to_rows(row, separate_by=None):
        if separate_by:
            split_row = row[target_column].split(separate_by)
        else:
            split_row = row[target_column]
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    df.apply(split_list_to_rows, axis=1, args=(separator, ))
    new_df = pandas.DataFrame(row_accumulator)
    return new_df


def get_custom_spatial_bounds(distance, lat, lon):
    height = wgs84_height(distance)
    width = wgs84_width(distance, lat)
    return {'lon_min': lon-width, 'lon_max': lon+width, 'lat_min': lat-height, 'lat_max': lat+height}


def create_grid_tesselation(xmin, ymin, xmax, ymax, width, height, random_seed=None):
    random.seed(a=random_seed)
    r_width = random.randint(0, width)
    r_height = random.randint(0, height)
    rows = int(np.ceil((ymax - ymin + r_height) / height))
    cols = int(np.ceil((xmax - xmin + r_width) / width))
    x_left_origin = xmin - r_width
    x_right_origin = x_left_origin + width
    y_top_origin = ymax + r_height
    y_bottom_origin = y_top_origin - height
    polygons = []
    for i in range(cols):
        y_top = y_top_origin
        y_bottom = y_bottom_origin
        for j in range(rows):
            polygons.append(Polygon(
                [(x_left_origin, y_top), (x_right_origin, y_top), (x_right_origin, y_bottom),
                 (x_left_origin, y_bottom)]))
            y_top = y_top - height
            y_bottom = y_bottom - height
        x_left_origin = x_left_origin + width
        x_right_origin = x_right_origin + width

    return polygons


def stop_sample(gtfs, sample_size=None, sample_fraction=None, tesselation_distance=1000, random_seed=1, **kwargs):
    stops, crs_utm = df_to_utm_gdf(gtfs.stops())
    total_n_stops = len(stops.index)
    assert sample_size or sample_fraction
    if sample_fraction:
        assert 0 < sample_fraction <= 1
        if sample_size:
            sample_size = max(sample_size, total_n_stops * sample_fraction)
        else:
            sample_size = total_n_stops * sample_fraction
    sample_size = math.ceil(sample_size)
    print("Using sample size:", sample_size)
    polygons = create_grid_tesselation(*stops.total_bounds, height=tesselation_distance, width=tesselation_distance,
                                       random_seed=random_seed)
    grid = GeoDataFrame({'geometry': polygons}, crs=crs_utm)
    grid["id"] = grid.index
    stops = sjoin(stops, grid, how="left", op='within')
    stops_grouped = stops.groupby(["id"])
    stops_grouped = stops_grouped.agg({'stop_I': 'count'}, axis=1)
    stops_grouped = stops_grouped.reset_index()
    sample_sizes = []
    for i in stops_grouped.itertuples():
        (div, mod) = divmod(sample_size * i.stop_I, total_n_stops)
        sample_sizes.append({"id": int(i.id), "div": div, "mod": mod})
    to_allocate = sample_size - sum([x["div"] for x in sample_sizes])
    sample_sizes = sorted(sample_sizes, key=lambda k: k['mod'], reverse=True)

    sample_sizes = [{"id": x["id"], "div": x["div"] + 1, "mod": x['mod']} if i < to_allocate else
                    {"id": x["id"], "div": x["div"], "mod": x['mod']} for i, x in enumerate(sample_sizes)]

    stops = stops.sort_values("stop_I")
    sample = GeoDataFrame()
    for row in sample_sizes:
        if row["div"] > 0:
            sample = sample.append(stops.loc[stops.id == row["id"]].sample(n=row["div"], random_state=random_seed))
    import matplotlib.pyplot as plt

    """
    plt.figure()

    ax = grid.plot(facecolor="none", edgecolor='black', lw=0.7)
    ax = stops.plot(ax=ax, column="id")
    ax = sample.plot(ax=ax, color="red")

    plt.show()
    """
    return sample["stop_I"].tolist()


def split_into_equal_length_parts(array, n_splits):
    # Taken from:
    # http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
    # Pretty nice solution.
    a = array
    n = n_splits
    k, m = divmod(len(a), n)
    lists = [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]
    assert(lists[0][0] == array[0])
    assert(lists[-1][-1] == array[-1])
    return lists


def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs.
    source: http://code.activestate.com/recipes/578114-round-number-to-specified-number-of-significant-di/
    """
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0


def split_by_char_closest_to_middle(text, delimiter=" ", filler="\n"):
    n = len(text)/2
    words = text.split(delimiter)
    candidate_len = 0
    prev_len = 0
    for word in words:
        candidate_len += len(word)
        if candidate_len > n:
            if n - prev_len < candidate_len - n:
                split_len = prev_len
                break
            else:
                split_len = candidate_len
                break
        prev_len = candidate_len
        candidate_len += 1

    char_list = list(text)
    char_list[split_len] = filler
    text = "".join(char_list)
    text = text.replace(delimiter, " ")
    return text


def apply_suffix(d, suffix, filler="_"):
    d = {k+filler+suffix: v for k, v in d.items()}
    return d


def check_day_start(gtfs, desired_weekday):
    """
    Assuming a weekly extract, gets the utc of the start of the desired weekday
    :param gtfs:
    :param day_start:
    :param desired_weekday:
    :return:
    """
    day_start_add = 24 * 3600
    day_start, _ = gtfs.get_day_start_ut_span()
    tz = gtfs.get_timezone_pytz()
    print("original weekday:", ut_to_utc_datetime(day_start, tz).weekday())
    weekday = ut_to_utc_datetime(day_start, tz).weekday()
    day_start += day_start_add * (9-weekday if weekday > desired_weekday else 2-weekday)
    print("day start:", day_start)
    print("day start weekday:", ut_to_utc_datetime(day_start, tz).weekday())
    return day_start


def subtract_dataframes(df1, df2, suffixes=("_x", "_y"), drop_cols=False, **kwargs):
    """
    Merges the dataframes and subtracts the matching columns
    :param df1: pandas DataFrame
    :param df2: pandas DataFrame
    :param suffixes:
    :param drop_cols:
    :param kwargs:
    :return:
    """
    cols1 = list(df1)
    cols2 = list(df2)
    common_cols = [col for col in cols1 if col in cols2]
    kwargs["right_index"] = kwargs.get("right_index", True)
    kwargs["left_index"] = kwargs.get("left_index", True)
    diff_suffix = kwargs.get("diff_suffix", "|diff")
    df = df1.merge(df2, suffixes=suffixes, **kwargs)
    for col in common_cols:
        df[col+diff_suffix] = df[col+suffixes[0]] - df[col+suffixes[1]]
        if drop_cols:
            df.drop([col+suffixes[0], col+suffixes[1]], inplace=True, axis=1)
    return df


def get_differences_between_dataframes(dfs):
    """
    Subtracts the difference of all combinations of dataframes. Dataframes are matched by index.
    Columns with similar name are subtracted.
    :param dfs: dict, {name str: df pandas.DataFrame }
    :return: dfs_to_return list of pandas.DataFrame, names_to_return list of strings
    """
    pairs = itertools.combinations(dfs.items(), 2)
    dfs_to_return = []
    names_to_return = []
    for (name1, df1), (name2, df2) in pairs:
        suffixes = ("|" + name1, "|" + name2)
        df = subtract_dataframes(df1, df2, suffixes=suffixes)
        df = df.reset_index()
        dfs_to_return.append(df)
        names_to_return.append((name1, name2))
    return dfs_to_return, names_to_return


def drop_nans(df):
    init_len = len(df.index)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    len_after_clean = len(df.index)
    print("WARNING! Of {init} rows, {removed} rows with missing data were removed"
          .format(init=init_len, removed=init_len - len_after_clean))
    return df


def flatten_2d_array(array):
    return [item for sublist in array for item in sublist]


def stops_within_buffer(input_geometry, gtfs, buffer_m=0):
    """
    Returns the stops that are within the given buffer of a given geometry
    :param input_geometry: GeoDataFrame or shapely
    :param gtfs: GTFS
    :param buffer_m: int
    :return:
    """
    stops_gdf, crs = df_to_utm_gdf(gtfs.stops())
    len_stops_init = len(stops_gdf.index)
    if isinstance(input_geometry, GeoDataFrame):
        buffer_gdf = input_geometry
    elif isinstance(input_geometry, DataFrame):
        buffer_gdf, crs = df_to_utm_gdf(input_geometry)
    else:
        raise NotImplementedError
    buffer_gdf = buffer_gdf.copy()
    buffer_gdf["geometry"] = buffer_gdf["geometry"].buffer(buffer_m)

    stops_gdf = sjoin(stops_gdf, buffer_gdf, how='inner')
    len_stops_final = len(stops_gdf.index)
    print("filetered from {init} to {final} stops".format(init=len_stops_init, final=len_stops_final))
    return stops_gdf


def filter_stops_spatially(df, gtfs, cols, buffer=None, geometry=None):
    """
    filters a dataframe spatially based on stops
    :param buffer: int or list
    :param df: DataFrame
    :param gtfs: GTFS
    :param cols: name of the stop column or list
    :param geometry: GeoDataFrame or list
    :return: DataFrame
    """
    if not isinstance(cols, list):
        cols = [cols]
    if not isinstance(buffer, list):
        buffer = [buffer]
    if not isinstance(geometry, list):
        geometry = [geometry]
    assert len(cols) == len(buffer) == len(geometry)
    for col_arg, buffer_m, gdf in zip(cols, buffer, geometry):
        if buffer_m and gdf is not None:
            stops = stops_within_buffer(gdf, gtfs, buffer_m=buffer_m)
        else:
            stops = stops_within_buffer(gdf, gtfs)
        df = df.loc[df[col_arg].isin(stops["stop_I"])].copy()
    return df


def tidy_value(v, ft=3):
    if abs(v) <= 10 ** (-1 * ft) or abs(v) >= 10 ** ft:
        return format(v, '.2E')
    else:
        return format(v, ".3f")


def tidy_label(label, capitalize=False):
    if capitalize:
        label = label.capitalize()
    label = label.replace("_", " ")
    return label


def find_df_value(df, key_col, key, value_col):
    print(key)
    return df.loc[df[key_col] == key, value_col].iloc[0]


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def save_fig(func):
    def inner(*args, **kwargs):
        if kwargs.get("plot_separately", False):
            fig = plt.figure(figsize=kwargs.get("figsize", [9, 5]))
            fig.add_subplot(111, projection=kwargs.get("projection", None))
            func(*args)
            #plt.tight_layout()
            # plt.show()
            fig_format = kwargs.get("fig_format", "png")
            folder = kwargs.get("folder", "")
            fname = kwargs.get("fname", "")
            plotname = kwargs.get("plotname", "")
            fname = fname + plotname + "." + fig_format
            plt.tight_layout()
            plt.savefig(os.path.join(makedirs(folder), fname), bbox_inches='tight', format=fig_format, dpi=300)
        else:
            ax = func(*args, **kwargs)
            return ax

    return inner


def set_bbox(distance=20000, lat=None, lon=None, bbox=None):
    if not lat and not lon:
        lat = bbox["lat_min"] + (bbox["lat_max"] - bbox["lat_min"])/2
        lon = bbox["lon_min"] + (bbox["lon_max"] - bbox["lon_min"])/2
    map_spatial_bounds = get_custom_spatial_bounds(distance, lat, lon)
    return map_spatial_bounds


def set_bbox_using_ratio(ratio, bbox=None):
    distance = ratio * wgs84_distance(bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_max"])
    return set_bbox(distance, bbox=bbox)


if __name__ == "__main__":
    print(split_by_char_closest_to_middle("asdasd_asd_asd_asd", "_"))
