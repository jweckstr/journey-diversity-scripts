from collections import OrderedDict

import matplotlib.pyplot as plt
from math import pi
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.cm import get_cmap
import pandas.plotting as pp
import matplotlib as mpl

from research.route_diversity.diversity_settings import CITIES, year_index, LINESTYLES
from research.route_diversity.rd_utils import tidy_value, tidy_label

main_color = 'grey'
text_color = 'dimgrey'


def normalize(df, cols_to_exclude=None):
    new_df = df.copy()
    cols = new_df.columns
    if cols_to_exclude:
        cols = [col for col in cols if col not in cols_to_exclude]

    for col in cols:
        new_df[col] = (new_df[col] - new_df[col].min()) / (new_df[col].max() - new_df[col].min())
    return new_df


def make_spider(value_lists, row, title, colors, measures, n_cols=None, n_rows=None, ylim=40, *args, **kwargs):

    labels = kwargs.get("labels", [None for x in colors])
    # number of variable

    N = len(measures)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(n_cols or 5, n_rows or 5, row + 1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], measures, color=main_color, size=11)
    # [split_by_char_closest_to_middle(col, delimiter="_") for col in cols]

    # Draw ylabels
    ax.set_rlabel_position(0)
    ticks = np.linspace(0, ylim, 4, endpoint=False)
    ticks = ticks[1:]

    plt.yticks(ticks, [str(x) for x in ticks], color=main_color, size=7)
    plt.ylim(0, ylim)

    # Ind1
    for value_list, color, label in zip(value_lists, colors, labels):
        value_list += value_list[:1]
        ax.plot(angles, value_list, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, value_list, label=label, color=color, alpha=0.4)

    # Add a title
    plt.title(title.capitalize(), size=11, color=text_color, y=1.2)
    if any(labels):
        ax.legend(loc=(1.15, 1.1), ncol=1)


def parallel_coordinates(df, group_column):
    n_cities = len(CITIES.keys())
    cmap = get_cmap("tab20", n_cities)

    group_a = []
    group_b = []
    colors_a = []
    colors_b = []

    for i, city in enumerate(CITIES.keys()):
        color = cmap(i)
        for feed in CITIES[city]:
            ix = year_index(feed)
            if ix == 0:
                group_a.append(feed)
                colors_a.append(color)
            else:
                group_b.append(feed)
                colors_b.append(color)

    # min max normalize to scale from 0->1
    new_df = normalize(df, ["city"])
    # Tidy labels:
    new_df["city"] = new_df["city"].apply(lambda x: tidy_label(x, capitalize=True))
    new_df = new_df.rename(columns={x: tidy_label(x) for x in new_df.columns})

    # apply color changes using rc_context
    with mpl.rc_context({'axes.edgecolor': main_color, 'xtick.color': main_color, 'axes.labelcolor': text_color}):

        # plot first batch of lines
        ax = pp.parallel_coordinates(new_df.loc[df["city"].isin(group_a)], group_column, color=colors_a,
                                     linestyle=LINESTYLES[0], axvlines_kwds={"c": main_color, "linewidth": 0.3})
        # plot batch with different linestyle
        ax = pp.parallel_coordinates(new_df.loc[df["city"].isin(group_b)], group_column, ax=ax, color=colors_b,
                                     linestyle=LINESTYLES[1], axvlines=False)
    fig = plt.gcf()
    # Annotate min and max values
    for i, (ix,  val) in enumerate(df.drop(group_column, axis=1).max().items()):
        ax.annotate(tidy_value(val), xy=(i, 1.05), ha='center', va='top', color=text_color)
    for i, (ix, val) in enumerate(df.drop(group_column, axis=1).min().items()):
        ax.annotate(tidy_value(val), xy=(i, -0.05), ha='center', va='bottom', color=text_color)
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([])
    ax.get_legend().remove()

    plt.xticks(rotation=30)
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    fig.legend(handles, labels, loc=9, ncol=len(CITIES.keys()))
    plt.show()

def main():
    # Set data
    column_names = ['A', 'B', 'C', 'D']
    values = [[[38, 1.5, 30, 4]],
              [[29, 10, 9, 34],
               [8, 39, 23, 24]],
              [[7, 31, 33, 14],
               [28, 15, 32, 14]]]

    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(values))

    # Loop to plot
    for row, values in enumerate(values):
        make_spider(value_lists=values, measures=column_names, row=row, title="row" + str(row), colors=my_palette(row))

    plt.show()


if __name__ == "__main__":
    main()
