#!/usr/bin/env python3


#  raster.SetProjection( layer.GetSpatialRef().ExportToWkt() )


import glob
from osgeo import gdal
from osgeo import ogr

import os



use_dbg_ds = True # use dbg dataset



def get_tif_filenames() :
    src_dir = '/media/manu/TOSHIBA EXT/Ortofotos/Navarra_2017_RGBi/*.tif'
    filenames = glob.glob( src_dir )
    if use_dbg_ds :
        filenames = [
                'data/0141_6-4.tif',
                'data/0141_6-5.tif',
                'data/0141_7-4.tif',
                'data/0141_7-5.tif'
                ]
    return filenames



def get_bboxes( filenames ) :
    bboxes = []
    for fn in filenames :
        raster = gdal.Open( fn )
        gt = raster.GetGeoTransform()
        x0, dx, dxdy, y0, dydx, dy = gt
        x_tl, y_tl = x0, y0
        x_br = x0 + dx * raster.RasterXSize
        y_br = y0 + dy * raster.RasterYSize
        bbox = ( ( x_tl, y_tl ), ( x_br, y_br) )
        bboxes.append( bbox )
    return bboxes



def bbox2geom( bbox ) :
    ( xLeft, yTop ), ( xRight, yBottom ) = bbox
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(xLeft, yTop)
    ring.AddPoint_2D(xLeft, yBottom)
    ring.AddPoint_2D(xRight, yTop)
    ring.AddPoint_2D(xRight, yBottom)
    ring.AddPoint_2D(xLeft, yTop)
    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)
    return geom



def get_rasters_geom( filenames ) :
    bboxes = get_bboxes( filenames )
    return [ bbox2geom( bbox ) for bbox in bboxes ]



def find_intersecting_rasters( rasters_geom, bbox ) :
    geom1 = bbox2geom( bbox )
    result = []
    for fn, bbox in zip( filenames, bboxes ) :
        geom2 = bbox2geom( bbox )
        #  print( 'geom1:', geom1 )
        #  print( 'geom2:', geom2 )
        if geom1.Intersect( geom2 ) :
            #  print( 'intersect :)))))))))))' )
            result.append( fn )
        #  print( 'no intersect..' )
    return result



filenames = get_tif_filenames()
assert( len(filenames) > 0 ), 'folder empty?'
#  filenames = filenames[:10]
bboxes = get_bboxes( filenames )
[ print( fn, bbox ) for fn, bbox in zip(filenames, bboxes) ]
#
rasters_geom = [ bbox2geom( bbox ) for bbox in bboxes ]
#







#  shp_fn = 'data/Edif_Clases.shp'
#  import geopandas as gpd
#  shapes = gpd.read_file( shp_fn )
#  print( shapes.columns.values ) # first features
#  print( shapes.head() )
#  #  #  print( shapes.iloc[39]['geometry'] )
#  #  import numpy as np
#  #  #  print( np.asarray( shapes.iloc[39]['geometry'] ).shape )
#  #  poly = np.asarray( shapes.iloc[103]['geometry'].boundary )
#  #  #  poly = np.asarray( shapes[ shapes.cparcela == 1115 ]['geometry'] )
#  #  x, y = poly[:,0], poly[:,1]






# load .shp 
if use_dbg_ds :
    shp_fn = 'data/Edif_Clases.shp'
else :
    raise 'TODO'
shp = ogr.Open( shp_fn , 0 ) # 0 means read-only. 1 means writeable.
layer = shp.GetLayer()
print( 'len(layer):', len(layer) )


print('processing .shp')
#  for k in range(min(7,len(layer))) :
for k in range(len(layer)) :
    #  feat = layer[k]
    #  print('-', k )
    feat = layer.GetFeature( k )
    envelope = feat.GetGeometryRef().GetEnvelope()
    xmin, xmax, ymin, ymax = envelope
    bbox = (xmin, ymax), (xmax, ymin)
    #  print( 'envelope:', envelope, ' -> bbox:', bbox )
    fns = find_intersecting_rasters( rasters_geom, bbox )
    print( k, fns )
raise "TODO"




#
import matplotlib.pyplot as plt
x = [ *[ bbox[0][0] for bbox in bboxes ], *[ bbox[1][0] for bbox in bboxes ] ]
y = [ *[ bbox[0][1] for bbox in bboxes ], *[ bbox[1][1] for bbox in bboxes ] ]

plt.plot( [ min(x), max(x)], [ min(y), max(y)], '.' )
for fn, bbox in zip( filenames, bboxes) :
    ( ( x_l, y_t ), ( x_r, y_b) ) = bbox
    plt.plot( [x_l, x_r, x_r, x_l, x_l], [y_t, y_t, y_b, y_b, y_t], '-r' )
    plt.text( (x_r+x_l)/2, (y_t+y_b)/2, os.path.basename( fn ) )


print('-----------')
for k in range(len(layer)) :
#  print( k )
    #  feat = layer[k]
    feat = layer.GetFeature( k )
    poly = feat.GetGeometryRef() # Poligon
    #  print( poly )
    #  print( 'poly.GetGeometryType():', poly.GetGeometryType() )
    #  print( 'poly.GetGeometryName():', poly.GetGeometryName() )
    #
    if poly.GetGeometryCount() > 1 :
        print( k, 'contains holes!' )
        print( layer[k].ExportToJson() )
    for k2 in range( poly.GetGeometryCount() ):
        lr =  poly.GetGeometryRef( k2 ) # Linearringaux
        #  print( '  -', lr )
        import numpy as np
        xy = np.asarray( lr.GetPoints() )
        #  print( xy.shape )
        x, y = xy[:,0], xy[:,1]
        if k2 == 0 :
            pass
            plt.plot( x, y, ':c' )
            plt.text( x.mean(), y.mean(), feat.GetField('PARCELA') )
        else :
            plt.plot( x, y, ':r' )
    #  print( 'field count 2 = ', poly.GetGeomFieldCount() )
    #  linestring = poly.GetBoundary() # LineString
    #  print( linestring )
    #  print( 'npoints:', linestring.GetPointCount() )
    #  print( poly.GetEnvelope() )  # = bbox
    #  poly = shapes.iloc[k]['geometry'].boundary
    #  if type(poly) != type(LineString()) :
    #      print( k )
    #  print( k, type(poly), poly )
    #  import numpy as np
    #  poly = np.asarray( linestring )
    #  print( poly )
    #  x, y = poly[:,0], poly[:,1]
    #  plt.plot( x, y, ':c' )

plt.axis('equal')
plt.show()



print('-----------')
import geopandas as gpd
shapes = gpd.read_file( shp_fn )
print( shapes.columns.values ) # first features
print( shapes.head() )




print('-----------')
layer.ResetReading()
layer.SetAttributeFilter('PARCELA = 859.0')
print( 'len(layer):', len(layer) )
for k in range(len(layer)) :
    #  print( k )
    #  feat = layer[k]
    feat = layer.GetFeature( k )
    print( feat.ExportToJson() )
    poly = feat.GetGeometryRef() # Poligon
    #
    if poly.GetGeometryCount() > 1 :
        print( k, 'contains holes!' )
        print( layer[k].ExportToJson() )
    for k2 in range( poly.GetGeometryCount() ):
        lr =  poly.GetGeometryRef( k2 ) # Linearringaux
        #  print( '  -', lr )
        import numpy as np
        xy = np.asarray( lr.GetPoints() )
        x, y = xy[:,0], xy[:,1]
        if k2 == 0 :
            pass
            plt.plot( x, y, ':c' )
            plt.text( x.mean(), y.mean(), feat.GetField('PARCELA') )
        else :
            plt.plot( x, y, ':r' )

plt.axis('equal')
plt.show()








#  if __name__ == '__main__':
#      dbg()
