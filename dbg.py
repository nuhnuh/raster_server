#!/usr/bin/env python3





def _get_poligon() :
    # find bbox of object 39 in the .shp
    shp_fn = '/home/manu/Manu/Jobs/UPNA/JdA/projects/BOLETUS/data/tcsa_shp/SEGU_Lin_SenHorizont.shp'
    shp_fn = '/home/manu/tmp/audicana/ciudadela/parcela_urbana.shp'
    import geopandas as gpd
    shapes = gpd.read_file( shp_fn )
    #  print( 'crs:', shapes.crs )
    print( shapes.columns.values ) # first features
    #  print( shapes.iloc[39]['geometry'] )
    import numpy as np
    #  print( np.asarray( shapes.iloc[39]['geometry'] ).shape )
    poly = np.asarray( shapes.iloc[103]['geometry'].boundary )
    #  poly = np.asarray( shapes[ shapes.cparcela == 1115 ]['geometry'] )
    x, y = poly[:,0], poly[:,1]
    #
    #  print( shapes.head(3) )
    return x, y



def _get_bbox( x, y ) :
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    return (xmin, ymin), (xmax, ymax)



def world2pix( gt, x_world, y_world ) : # gt = raster.GetGeoTransform()
    x0, dx, dxdy, y0, dydx, dy = gt
    x = ( x_world - x0 ) // dx
    y = ( y_world - y0 ) // dy
    return x, y



def dbg() :
    import glob
    import os
    files = glob.glob(os.path.expanduser('~/tmp/audicana/*.tif'))
    print( files )

    from osgeo import gdal

    raster = gdal.Open( files[0] )
    print( raster )
    print( 'projection:', raster.GetProjection() )
    print( 'dimensions:', raster.RasterXSize, raster.RasterYSize )
    print( 'nbands:', raster.RasterCount )
    print( 'metadata:', raster.GetMetadata() )

    band = raster.GetRasterBand(1)
    print( 'band.datatype:', gdal.GetDataTypeName(band.DataType) )

    #  if band.GetMinimum() is None or band.GetMaximum()is None:
    #      band.ComputeStatistics(0)
    #      print("Statistics computed.")
    print('[ NO DATA VALUE ] = ', band.GetNoDataValue() ) # none
    print('[ MIN ] = ', band.GetMinimum() )
    print('[ MAX ] = ', band.GetMaximum() )
    print('[ SCALE ] = ', band.GetScale() )




    #  from osgeo import ogr
    #  shp = ogr.Open( shp_fn )
    #  layer = shp.GetLayer()
    #  #  wkt = "POLYGON ((-103.81402655265633 50.253951270672125,-102.94583419409656 51.535568561879401,-100.34125711841725 51.328856095555651,-100.34125711841725 51.328856095555651,-93.437060743203844 50.460663736995883,-93.767800689321859 46.450441890315041,-94.635993047881612 41.613370178339181,-100.75468205106476 41.365315218750681,-106.12920617548238 42.564247523428456,-105.96383620242338 47.277291755610058,-103.81402655265633 50.253951270672125))"
    #  #  layer.SetSpatialFilter(ogr.CreateGeometryFromWkt(wkt))
    #  x_min, x_max, y_min, y_max = layer.GetExtent()
    #  schema = []
    #  ldefn = layer.GetLayerDefn()
    #  print( layer.GetFeatureCount() )
    #  for n in range(ldefn.GetFieldCount()):
    #      fdefn = ldefn.GetFieldDefn(n)
    #      schema.append(fdefn.name)
    #  print( 'schema:', schema )


    # get bbox from the .tif
    x, y = _get_poligon()
    (x_min, y_min), (x_max, y_max) = _get_bbox( x, y )
    #  #
    #  import matplotlib.pyplot as plt
    #  plt.plot( x, y, '-' )
    #  plt.plot( [x_min, x_max], [y_min, y_max], '-r' )
    #  plt.show()


    gt = raster.GetGeoTransform()
    x_min_world, y_min_world = x_min, y_min
    x_min_pix, y_min_pix = world2pix( gt, x_min_world, y_min_world ) # gt = raster.GetGeoTransform()
    print( 'pix min(x,y):', x_min_pix, y_min_pix )
    x_max_world, y_max_world = x_max, y_max
    x_max_pix, y_max_pix = world2pix( gt, x_max_world, y_max_world ) # gt = raster.GetGeoTransform()
    print( 'pix max(x,y):', x_max_pix, y_max_pix )

    xoff = min( x_min_pix, x_max_pix )
    yoff = min( y_min_pix, y_max_pix )
    xcount = abs( x_max_pix - x_min_pix ) + 1
    ycount = abs( y_max_pix - y_min_pix ) + 1
    xoff, yoff, xcount, ycount = int(xoff), int(yoff), int(xcount), int(ycount)


    data = band.ReadAsArray( xoff, yoff, xcount, ycount )
    import matplotlib.pyplot as plt
    plt.imshow( data, cmap='gray' )
    plt.plot( x-x_min, y-y_min, ':r' )
    plt.show()


    #  x0, dx, dxdy, y0, dydx, dy = raster.GetGeoTransform()
    #  print( x0, y0, dx, dy, dxdy, dydx )
    #  #
    #  data = raster.ReadAsArray()
    #  print( type(data) )
    #  print( data )
    #  print( data.shape )



def dbg2():
    from osgeo import ogr
    drv = ogr.GetDriverByName('ESRI Shapefile')



if __name__ == '__main__':
    dbg()
