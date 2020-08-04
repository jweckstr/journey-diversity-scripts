"""
PSEUDOCODE:

Load csv to pandas
csv will be of form: city, event type, event name, year, theme_A, theme_B, theme_C...
City can contain multiple cities, separated by TBD?

Check min and max year

Open figure,

Deal with events in same year, offset a little bit?

For city in cities:tle
for event in events


"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import OrderedDict
from numpy import cos, sin, deg2rad, arange
from matplotlib import gridspec
from pylab import Circle

def clean_years(year):
    if isinstance(year, str):
        if len(year) > 4:
            year = year[:4]
        if year == "?":
            return year
    return int(year)


def split_to_separate_rows(df, column, split_key):
    s = df[column].str.split(split_key, expand=True).stack()
    i = s.index.get_level_values(0)
    df2 = df.loc[i].copy()
    df2[column] = s.values
    return df2


def slot_location(n_slots, which_slot):
    if n_slots == 1:
        return (0, 0)
    else:
        coord_list = []
        for i in range(0, n_slots):
            angle = (360 / n_slots) * i
            coord_list.append((offset * sin(deg2rad(angle)), offset * cos(deg2rad(angle))))
    return coord_list[which_slot]

base_path = "/home/clepe/route_diversity/data/plannord_tables/"
themes_path = base_path + "themes.csv"
events_path = base_path + "events.csv"
year_length = 1
city_height = 1
size = 0.1
theme_length = 0.5
theme_width = 1
offset = 0.15
event_offset = 0.15
start_year = 2000
end_year = 2024

color_dict = {"Land use or infrastructure planning": "#66c2a5",
              "Service level analysis or definitions": "#fc8d62",
              "PTN plan or comparison": "#8da0cb",
              "PT strategy": "#e78ac3",
              "Transport system plan or strategy": "#a6d854",
              'Other': "k"}

type_dict = {"Conference procedings": "Other",
             'PTS whitepaper': "Other",
             'Replies from hearing': "Other",
             'PT authority strategy': "Other",
             'PTS white paper': "Other",
             'PT "product characterization"': "Other",
             'Other': "Other",
             "Infrastructure analysis or plan": "Land use or infrastructure planning",
             "Master planning": "Land use or infrastructure planning",
             "PT service level analysis": "Service level analysis or definitions",
             "PT service level definitions": "Service level analysis or definitions",
             "PTN comparison": "PTN plan or comparison",
             "PTS plan": "PTN plan or comparison",
             "PTS strategy": "PT strategy",
             "Transport system plan": "Transport system plan or strategy",
             "Transport system strategy": "Transport system plan or strategy"}

event_offsets = {"LRT/tram": event_offset,
                 "BHLS or large route overhaul": 0,
                 "BRT/superbus": -1 * event_offset}

event_colors = {"LRT/tram": "g",
                 "BHLS or large route overhaul": "#0042FF",
                 "BRT/superbus": "#001C6E"}


theme_angles = {"through_routes": 0, "network_simplicity": 120, "trunk_network": 240}

themes_df = pd.read_csv(themes_path)
events_df = pd.read_csv(events_path)
themes_df = themes_df[pd.notnull(themes_df['year'])]

events_df = events_df[pd.notnull(events_df['year'])]

themes_df["year"] = themes_df.apply(lambda x: clean_years(x.year), axis=1)
events_df["year"] = events_df.apply(lambda x: clean_years(x.year), axis=1)

themes_df = split_to_separate_rows(themes_df, "city", "/")

themes_df.loc[themes_df['city'] == "Fredrikstad-Sarpsborg", 'city'] = "F:stad-S:borg"
events_df.loc[events_df['city'] == "Fredrikstad-Sarpsborg", 'city'] = "F:stad-S:borg"
themes_df.loc[themes_df['city'] == "Porsgrunn-Skien", 'city'] = "P:grunn-Skien"
events_df.loc[events_df['city'] == "Porsgrunn-Skien", 'city'] = "P:grunn-Skien"

city_year_slots = {}
for i, row in themes_df[["city", "year"]].append(events_df[["city", "year"]]).iterrows():
    if (row.city, row.year) in city_year_slots.keys():
        city_year_slots[(row.city, row.year)] += 1
    else:
        city_year_slots[(row.city, row.year)] = 1
city_year_cur_slot = {key: 0 for key, value in city_year_slots.items()}

cities = [x for x in set(themes_df.city.dropna().tolist()) if "/" not in x]
cities.sort(reverse=True)
themes_df["type"] = themes_df.apply(lambda row: type_dict[row.type], axis=1)
types = [x for x in set(themes_df.type.dropna().tolist())]


fig = plt.figure()
ax1 = plt.subplot(111)
#gs = gridspec.GridSpec(1, 2, width_ratios=[1, 9])
#ax1 = plt.subplot(gs[1])
#ax2 = plt.subplot(gs[0], sharey=ax1)
"""
gs1 = gridspec.GridSpec(3, 3)
gs1.update(right=.7, wspace=0.05)
ax1 = plt.subplot(gs1[:-1, :])
ax2 = plt.subplot(gs1[-1, :-1])
ax3 = plt.subplot(gs1[-1, -1])
"""
groups = themes_df.groupby('type')

for i, row in events_df.iterrows():

    e_offset = event_offsets[row.type]
    c = event_colors[row.type]
    y = city_height * cities.index(row.city) + e_offset
    x = row.year
    ax1.plot([row.year, end_year+1], [y, y], c=c, marker='o', label=row.type, zorder=2, markersize=3)

for name, group in groups:

    for i, row in group.iterrows():
        n_slots = city_year_slots[(row.city, row.year)]
        cur_slot = city_year_cur_slot[(row.city, row.year)]
        city_year_cur_slot[(row.city, row.year)] += 1
        slot_offset = slot_location(n_slots, cur_slot)
        y = city_height * cities.index(row.city) + slot_offset[0]
        x = row.year + slot_offset[1]
        if row.year < start_year:
            continue

        #circle = Circle((x, y), color=color_dict[name], radius=size, label=name, zorder=5)
        ax1.scatter(x, y, color=color_dict[name], s=5, label=name, zorder=5) #add_patch(circle)
        for theme, angle in theme_angles.items():
            if pd.notnull(row[theme]):
                ax1.plot([x, x + theme_length * sin(deg2rad(angle))], [y, y + theme_length * cos(deg2rad(angle))],
                         c=color_dict[name], zorder=10, linewidth=theme_width)


handles, labels = ax1.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))

#ax1.legend(by_label.values(), by_label.keys())

# TODO: add year for GTFS feed as vertical line
#ax2 = fig.add_subplot(121, sharey=ax1)
for city in cities:
    y = city_height * cities.index(city)
    x = end_year
    ax1.text(x, y, city, horizontalalignment='left', verticalalignment='center', fontsize=10) #, bbox=dict(boxstyle="square", facecolor='white', alpha=0.5, edgecolor='white'))
    ax1.plot([start_year-1, end_year+1], [y, y], c="grey", alpha=0.5, linewidth=0.1, zorder=1)

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_yticks([])
ax1.set_yticklabels([])
#ax2.axis('off')
ax1.set_xlim(start_year, end_year)
ax1.set_aspect("equal")
plt.xticks(arange(start_year, end_year, 5))

plt.savefig(base_path+'timeline.pdf', format="pdf", dpi=300, bbox_inches='tight')

fig = plt.figure()
ax2 = plt.subplot(111)
ax2.legend(by_label.values(), by_label.keys(), loc='center', #bbox_to_anchor=(0.5, -0.05),
           fancybox=True, shadow=True, ncol=2)
ax2.axis('off')

plt.savefig(base_path+'legend.pdf', format="pdf", dpi=300, bbox_inches='tight')
#plt.show()

# create legend for themes in a separate figure

fig = plt.figure()
ax3 = plt.subplot(111)
x = 0
y = 0
circle = Circle((x, y), color="black", radius=size, zorder=5)
ax3.add_patch(circle)
for theme, angle in theme_angles.items():
    x1 = x + theme_length * sin(deg2rad(angle))
    y1 = y + theme_length * cos(deg2rad(angle))
    x2 = x + theme_length * sin(deg2rad(angle)) * 1.2
    y2 = y + theme_length * cos(deg2rad(angle)) * 1.2
    ax3.annotate(theme.capitalize().replace("_", " "), (x1, y1), (x2, y2), horizontalalignment='center',
                 verticalalignment='center', color="red", zorder=10, size=15)
    ax3.plot([x, x1], [y, y1], c="black",
             linewidth=10*theme_width, zorder=1)
ax3.set_aspect("equal")
ax3.axis('off')
plt.savefig(base_path+'timeline_themes.pdf', format="pdf", dpi=300, bbox_inches='tight')

