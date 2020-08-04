import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits import axes_grid1

# colormaps: "viridis", "plasma_r","seismic"

# ROUTE CATEGORIES:
from research.route_diversity.diversity_settings import day, peak

radial_route = 'radial_route'
cross_route = 'cross_route'
orbital = 'orbital'
feeder = 'feeder'
ROUTE_TYPE_COLORS = {radial_route: 'red', cross_route: 'orange', orbital: 'blue', feeder: 'cyan'}


def ratio_name(name):
    return name + "_ratio"


# INTERMEDIATE STATIC NETWORK MEASURE NAMES:
segment_length_m = "segment_length"
segment_kilometrage = "segment_kilometrage"

# AGGREGATED STATIC NETWORK MEASURE NAMES:
route_length = "route_length"
trunk_length = "trunk_length"
prop_length = "prop_length"
route_section_length = "route_section_length"
trunk_section_length = "trunk_section_length"
prop_section_length = "prop_section_length"
route_kilometrage = "route_kilometrage"
trunk_kilometrage = "trunk_kilometrage"
prop_kilometrage = "prop_kilometrage"
avg_segment_frequency = "avg_segment_frequency"
route_overlap = "route_overlap"
cross_route_ratio = "cross_route_ratio"
long_service_hour_kms = "long_service_hour_kms"
long_service_hour_prop = "long_service_hour_prop"
mean_service_hours = "mean_service_hours"
weighted_mean_service_hours = "weighted_mean_service_hours"
number_of_route_variants = "number_of_route_variants"
mean_jaccard = "mean_jaccard"

STATIC_MEASURE_ALIASES = {
    route_length: "route_length",
    trunk_length: "trunk_length",
    prop_length: "high_frequency_service_prevalence",
    route_section_length: "route_section_length",
    trunk_section_length: "trunk_section_length",
    prop_section_length: "prop_section_length",
    route_kilometrage: "route_kilometrage",
    trunk_kilometrage: "trunk_kilometrage",
    prop_kilometrage: "prop_kilometrage",
    avg_segment_frequency: "average_frequency",
    route_overlap: "average_route_overlap",
    cross_route_ratio: "cross_route_ratio",
    long_service_hour_kms: "long_service_hour_kms",
    long_service_hour_prop: "long_service_hour_prop",
    mean_service_hours: "mean_service_hours",
    weighted_mean_service_hours: "weighted_mean_service_hours",
    number_of_route_variants: "number_of_route_variants",
    mean_jaccard: "mean_jaccard"
}


def apply_static_alias(name):
    if day in name[-3:]:
        string_to_remove = "_" + day
    elif peak in name[-4:]:
        string_to_remove = "_" + peak
    else:
        string_to_remove = ""
    if len(string_to_remove) != 0:
        name = name[:-1*len(string_to_remove)]
        name = STATIC_MEASURE_ALIASES[name]
        return name + "_(" + string_to_remove[1:] + ")"
    else:
        name = STATIC_MEASURE_ALIASES[name]
        return name


# DISTRIBUTIONAL STATIC NETWORK MEASURE NAMES:
route_frequency = "route_frequency"
trips_in_area = "trips_in_area"

measures = {'cross-routes': [cross_route_ratio],
            'service-hours': [long_service_hour_kms, long_service_hour_prop, mean_service_hours,
                              weighted_mean_service_hours],
            'trunk_network': [trunk_kilometrage, trunk_length, trunk_section_length, avg_segment_frequency],
            'network_simplicity': [number_of_route_variants],
            'schedule_simplicity': [mean_jaccard],
            "network": [route_kilometrage, route_length, route_overlap, route_section_length]}

diff_suffix = "|diff"
zero_to_one = [0, 1]
half_to_half = [-0.5, 0.5]
default_cmap = "viridis"
default_diff_cmap = "seismic"

# ROUTING BASED MEASURE NAMES:
journey_variant_weighted_simpson = "journey_variant_weighted_simpson"
most_probable_departure_stop = "most_probable_departure_stop"
most_probable_journey_variant = "most_probable_journey_variant"
number_of_fp_journeys = "number_of_fp_journeys"
number_of_journeys = "number_of_journey_alternatives"
number_of_fp_journey_variants = "number_of_fp_journey_variants"
number_of_journey_variants = "number_of_journey_variants"
time_weighted_simpson = "time_weighted_simpson"
trip_density = "trip_density"
avg_circuity = "avg_circuity"
avg_speed = "avg_speed"
mean_temporal_distance = "mean_temporal_distance"
mean_trip_n_boardings = "mean_trip_n_boardings"
proportion_fp_journeys = "proportion_fp_journeys"
number_of_most_common_journey_variant = "number_of_most_common_journey_variant"
largest_headway_gap = "largest_headway_gap"
expected_pre_journey_waiting_time = "expected_pre_journey_waiting_time"

MEASURE_ALIASES = {journey_variant_weighted_simpson: "diversity_of_journey_alternatives_journey_variant_weighted",
                   most_probable_departure_stop: "most_probable_departure_stop",
                   most_probable_journey_variant: "most_probable_journey_variant",
                   number_of_fp_journeys: "number_of_fp_journeys",
                   number_of_journeys: "number_of_journey_alternatives",
                   number_of_fp_journey_variants: "number_of_fp_journey_variants",
                   number_of_journey_variants: "number_of_journey_variants",
                   time_weighted_simpson: "diversity_of_journey_alternatives",
                   trip_density: "trip_density",
                   avg_circuity: "avg_circuity",
                   avg_speed: "avg_speed",
                   mean_temporal_distance: "mean_temporal_distance",
                   mean_trip_n_boardings: "mean_trip_n_boardings",
                   proportion_fp_journeys: "proportion_fp_journeys",
                   number_of_most_common_journey_variant: "number_of_most_common_journey_variant",
                   largest_headway_gap: "largest_headway_gap",
                   expected_pre_journey_waiting_time: "expected_pre_journey_waiting_time"}

MEASURE_PLOT_PARAMS = {
    journey_variant_weighted_simpson:
        {"lims": zero_to_one, "diff_lims": half_to_half},
    most_probable_departure_stop:
        {"lims": zero_to_one, "diff_lims": half_to_half},
    most_probable_journey_variant:
        {"lims": zero_to_one, "diff_lims": half_to_half},
    number_of_journeys:
        {"lims": [0, 30], "diff_lims": [-10, 10]},
    number_of_fp_journeys:
        {"lims": [0, 30], "diff_lims": [-10, 10]},
    number_of_fp_journey_variants:
        {"lims": [0, 15], "diff_lims": [-5, 5]},
    number_of_journey_variants:
        {"lims": [0, 15], "diff_lims": [-5, 5]},
    time_weighted_simpson:
        {"lims": zero_to_one, "diff_lims": half_to_half},
    trip_density:
        {"lims": [0, 500], "diff_lims": [-100, 100]},
    avg_circuity:
        {"lims": [1, 4], "diff_lims": half_to_half},
    avg_speed:
        {"lims": [5, 50], "diff_lims": [-5, 5]},
    mean_temporal_distance:
        {"lims": [0, 150], "diff_lims": [-15, 15]},
    mean_trip_n_boardings:
        {"lims": [1, 4], "diff_lims": [-1.5, 1.5]},
    proportion_fp_journeys:
        {"lims": zero_to_one, "diff_lims": half_to_half},
    number_of_most_common_journey_variant:
        {"lims": [0, 15], "diff_lims": [-5, 5]},
    largest_headway_gap:
        {"lims": [0, 60], "diff_lims": [-10, 10]},
    expected_pre_journey_waiting_time:
        {"lims": [0, 30], "diff_lims": [-10, 10]},
}


def get_cmap_parameters(data, kwargs):
    percentiles = kwargs.get("percentiles", (9, 91))

    cmap = matplotlib.cm.get_cmap(name=default_cmap, lut=None)
    print(percentiles)
    vmin = numpy.percentile(data, percentiles[0])
    vmax = numpy.percentile(data, percentiles[1])

    if min(data) < 0 < max(data):
        vmax = -1 * vmin
        cmap = matplotlib.cm.get_cmap(name=default_diff_cmap, lut=None)
    elif min(data) == 0:
        vmin = 0
    elif max(data) == 0:
        vmax = 0
    return vmin, vmax, cmap


def get_colormap(observable_name=None, data=None, **kwargs):
    print(observable_name)
    if data is not None:
        vmin, vmax, cmap = get_cmap_parameters(data, kwargs)

    elif "diff" not in observable_name:
        observable_name = observable_name.split("|")[0]
        vmin, vmax = MEASURE_PLOT_PARAMS[observable_name]["lims"]
        cmap = matplotlib.cm.get_cmap(name=default_cmap, lut=None)

    else:
        observable_name = observable_name.split("|")[0]
        vmin, vmax = MEASURE_PLOT_PARAMS[observable_name]["diff_lims"]
        cmap = matplotlib.cm.get_cmap(name=default_diff_cmap, lut=None)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


def create_colorbar(cmap, norm):
    """Create a colorbar with limits of lwr and upr"""
    cax, kw = matplotlib.colorbar.make_axes(matplotlib.pyplot.gca())
    c = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    return c


def add_colorbar(im, aspect=20, pad_fraction=1, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

