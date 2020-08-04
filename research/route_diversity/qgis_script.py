import os

from qgis.core import (
    QgsVectorLayer, QgsDataSourceUri, QgsProject
)

to_publish_dir = "/home/clepe/route_diversity/data/cities"

path = os.path.join(to_publish_dir)
layers = []
for root, dirs, files in os.walk(path):
    if dirs:
        feed = os.path.basename(root)
    fname = 'week.sqlite'
    if fname in files:
        year = os.path.basename(root)[:4]
        print(root, dirs, files)
        print(feed, year)

        fname_or_conn = os.path.join(root, fname)
        print(year)
        print(fname_or_conn)
        uri = QgsDataSourceUri()
        uri.setDatabase(fname_or_conn)
        schema = ''
        table = 'stop_intervals'
        geom_column = 'Geometry'
        uri.setDataSource(schema, table, geom_column)

        display_name = feed + "_" + year
        vlayer = QgsVectorLayer(uri.uri(), display_name, 'spatialite')
        #vlayer.loadNamedStyle('/home/clepe/gis/stop_segments_style.qml')
        layers.append(vlayer)
        vlayer = None
QgsProject.instance().addMapLayers(layers)
