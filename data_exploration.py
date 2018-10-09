#!/usr/bin/env python3


#  raster.SetProjection( layer.GetSpatialRef().ExportToWkt() )



use_dbg_ds = True # use dbg dataset



def get_tif_filenames() :
    import glob
    import os
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
    from osgeo import gdal
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


filenames = get_tif_filenames()
assert( len(filenames) > 0 ), 'folder empty?'
#  filenames = filenames[:10]
bboxes = get_bboxes( filenames )
[ print( fn, bbox ) for fn, bbox in zip(filenames, bboxes) ]
#  raise 'TODO'



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







if use_dbg_ds :
    shp_fn = 'data/Edif_Clases.shp'
else :
    raise 'TODO'
from osgeo import ogr
shp = ogr.Open( shp_fn , 0 ) # 0 means read-only. 1 means writeable.
layer = shp.GetLayer()
#  layer.SetAttributeFilter("PARCELA = '859.0'") # does not work :/
x_min, x_max, y_min, y_max = layer.GetExtent()
print( layer.GetExtent() )
layer.ResetReading()
print( 'len(layer):', len(layer) )



# attempt to apply spatial filter (does not work)
#
from osgeo import ogr
# Create ring
ring = ogr.Geometry(ogr.wkbLinearRing)
bbox = bboxes[0]
( x_l, y_t ), ( x_r, y_b ) = bbox
ring.AddPoint_2D( x_l, y_t )
ring.AddPoint_2D( x_l, y_b )
ring.AddPoint_2D( x_r, y_b )
ring.AddPoint_2D( x_r, y_t )
ring.AddPoint_2D( x_l, y_t )
# Create polygon
poly = ogr.Geometry(ogr.wkbPolygon)
poly.AddGeometry(ring)
print('poly:', poly)
#
#  layer.SetSpatialFilter( poly )
#  wkt = 'POLYGON ((610730 4733890,614290 4733890,614290 4736370,610730 4736370,610730 4733890))'
#  layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))
print( 'len(layer):', len(layer) )
print('ERROR: Spatial Filter is not working!!!!!!!!!!!!!!')
layer.SetSpatialFilter( poly )
print( 'len(layer):', len(layer) )
#  for k in range(len(layer)) :
#      print( k )
#      aux = layer[k]
#      print( aux.GetGeometryRef() )
#      #  print( layer[k].GetGeometryRef() )





#
import matplotlib.pyplot as plt
x = [ *[ bbox[0][0] for bbox in bboxes ], *[ bbox[1][0] for bbox in bboxes ] ]
y = [ *[ bbox[0][1] for bbox in bboxes ], *[ bbox[1][1] for bbox in bboxes ] ]

plt.plot( [ min(x), max(x)], [ min(y), max(y)], '.' )
for fn, bbox in zip( filenames, bboxes) :
    ( ( x_l, y_t ), ( x_r, y_b) ) = bbox
    plt.plot( [x_l, x_r, x_r, x_l, x_l], [y_t, y_t, y_b, y_b, y_t], '-r' )
    import os
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
