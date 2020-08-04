from research.route_diversity.static_route_type_analyzer import RouteTypeAnalyzer
from pandas import DataFrame
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from research.tis_paper.settings import *


class StaticNavigabilityAnalyzer:
    def __init__(self, gtfs, day_start, name=None):
        self.gtfs = gtfs
        self.day_start = day_start
        self.name = name

    def hourly_route_trajectories(self):
        df = DataFrame()
        time_cols = []
        no_trips_cols = []
        for i in range(0, 24):
            col_name = str(i).zfill(2)
            try:
                temp_df = self.gtfs.calculate_route_trajectories(self.day_start, start_time=i*3600, end_time=(i+1)*3600)

                temp_df.rename(index=str, columns={"n_trips": col_name}, inplace=True)
                df = df.append(temp_df, sort=True, ignore_index=True)
                time_cols.append(col_name)

            except ValueError:
                print("no trips between", i, "and", i+1)
                df[col_name] = 0
                no_trips_cols.append(col_name)
                continue

        df.fillna(0, inplace=True)
        df_gb = df.groupby(['wkt'])
        agg_dict = {k: "sum" for k in list(df) if not k == 'wkt'}
        agg_dict.update({'geometry': lambda x: x.iloc[0], 'name': lambda x: x.iloc[0],
                         'trip_Is': lambda x: ','.join(x)})
        df = df_gb.agg(agg_dict, axis=1)
        df = df.reset_index()
        return df, time_cols

    def time_variation_navigability_measures(self):
        df, time_cols = self.hourly_route_trajectories()
        jaccards = []
        for col1 in time_cols:
            for col2 in time_cols:
                if col1 == col2:
                    continue
                vec1, vec2 = np.array(df[col1]), np.array(df[col2])
                keepers = np.where(np.logical_not((np.vstack((vec1, vec2)) == 0).all(axis=0)))  # removes items that
                # are zero in both lists
                vec1 = np.where(vec1 > 0.0, 1, 0)
                vec2 = np.where(vec2 > 0.0, 1, 0)

                jaccards.append(jaccard_score(vec1[keepers], vec2[keepers], average='binary'))
        df['count'] = df.apply(lambda row: len([row[x] for x in time_cols if row[x] >= 1]), axis=1)
        df['sum'] = df.apply(lambda row: sum([row[x] for x in time_cols if row[x] >= 1]), axis=1)
        df["length"] = df.apply(lambda row: row.geometry.length, axis=1)
        df['service_km'] = df['sum'] * df["length"]/1000
        df_long = df.loc[df['count'] >= 18]
        long_service_hour_kms = df_long['service_km'].sum()
        long_service_hour_prop = df_long['service_km'].sum()/df['service_km'].sum()
        mean_service_hours = df['count'].mean()
        weighted_mean_service_hours = np.average(df['count'], weights=df['service_km'])
        measure_dict = {"mean_jaccard": sum(jaccards)/len(jaccards),
                        "mean_service_hours": mean_service_hours,
                        "weighted_mean_service_hours": weighted_mean_service_hours,
                        "long_service_hour_kms": long_service_hour_kms,
                        "long_service_hour_prop": long_service_hour_prop}
        return measure_dict

    def variants_by_label(self):
        na = RouteTypeAnalyzer(self.gtfs, self.day_start, start_time=0, end_time=24 * 3600,
                               hub_merge_distance_m=200, name=self.name)
        df = na.get_trajectories()
        labels = set(df["name"].tolist())
        for label in labels:
            temp_df = df.loc[df['name'] == label]
            ax = None
            for (i, row), color in zip(temp_df.iterrows(), mcolors.BASE_COLORS):
                try:
                    ax = na.plot_trips("route_"+label, trip_Is=row.trip_Is.split(","), color=color, return_ax=True,
                                       plot_grey_base=False, ax=ax)
                except:
                    print("problem with route:", label, " and trips ", row.trip_Is.split(","))
            try:
                plt.savefig(FIGS_DIRECTORY + self.name + "_" + "route_"+label + ".png",
                            format="png",
                            bbox_inches='tight',
                            dpi=900)
            except:
                plt.show()
            plt.close()
