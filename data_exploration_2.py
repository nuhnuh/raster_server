#!/usr/bin/env python3


#  raster.SetProjection( layer.GetSpatialRef().ExportToWkt() )


import glob
from osgeo import gdal
from osgeo import ogr
import numpy as np
import os
import matplotlib.pyplot as plt



use_dbg_ds = True # use dbg dataset
raise 'TODO: merge subimgs of the tifs intersections'



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
        bbox = x_tl, y_tl, x_br, y_br
        bboxes.append( bbox )
    return bboxes



def bbox2geom( bbox ) :
    xLeft, yTop, xRight, yBottom = bbox
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint_2D(xLeft, yTop)
    ring.AddPoint_2D(xLeft, yBottom)
    ring.AddPoint_2D(xRight, yBottom)
    ring.AddPoint_2D(xRight, yTop)
    ring.AddPoint_2D(xLeft, yTop)
    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)
    return geom



def get_rasters_geom( filenames ) :
    bboxes = get_bboxes( filenames )
    return [ bbox2geom( bbox ) for bbox in bboxes ]



def find_intersecting_rasters( rasters_geom, bbox ) :
    # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    geom1 = bbox2geom( bbox )
    result = []
    for idx, ( fn, bbox ) in enumerate(zip( filenames, bboxes )) :
        geom2 = bbox2geom( bbox )
        #  print( 'geom1:', geom1 )
        #  print( 'geom2:', geom2 )
        if geom1.Intersect( geom2 ) : 
            result.append( idx )
    return result



# load rasters (tifs) geometry
filenames = get_tif_filenames()
assert( len(filenames) > 0 ), 'folder empty?'
#  filenames = filenames[:10]
bboxes = get_bboxes( filenames )
[ print( fn, bbox ) for fn, bbox in zip(filenames, bboxes) ]
#
rasters_geom = [ bbox2geom( bbox ) for bbox in bboxes ]







# find the intersection of the rasters
geom = rasters_geom[0]
print( geom )
for geom_k in rasters_geom[1:] :
    print( geom_k )
    geom = geom.Intersection( geom_k )
#
for geom_k in rasters_geom :
    xy = np.asarray( geom_k.Boundary().GetPoints() )
    x, y = xy[:,0], xy[:,1]
    plt.plot( x, y )
xy = np.asarray( geom.Boundary().GetPoints() )
x, y = xy[:,0], xy[:,1]
plt.plot( x, y, ':k' )
plt.axis('equal')
plt.show()





# load .shp 
if use_dbg_ds :
    shp_fn = 'data/Edif_Clases.shp'
else :
    raise 'TODO'
shp = ogr.Open( shp_fn , 0 ) # 0 means read-only. 1 means writeable.
layer = shp.GetLayer()
print( 'len(layer):', len(layer) )




#
for geom_k in rasters_geom :
    xy = np.asarray( geom_k.Boundary().GetPoints() )
    x, y = xy[:,0], xy[:,1]
    plt.plot( x, y )
xy = np.asarray( geom.Boundary().GetPoints() )
x, y = xy[:,0], xy[:,1]
plt.plot( x, y, ':k' )
#
feature = layer[248]
envelope = feature.GetGeometryRef().GetEnvelope()
xmin, xmax, ymin, ymax = envelope
bbox = xmin, ymax, xmax, ymin
xy = np.asarray( bbox2geom( bbox ).Boundary().GetPoints() )
x, y = xy[:,0], xy[:,1]
plt.plot( x, y, 'c' )
#
boundary = feature.GetGeometryRef().GetBoundary()
xy = np.asarray( boundary.GetPoints() )
x, y = xy[:,0], xy[:,1]
plt.plot( x, y, 'c' )
#
plt.axis('equal')
plt.show()








def raster2img( raster ) :
    nbands = raster.RasterCount
    xcount = raster.RasterXSize
    ycount = raster.RasterYSize
    img = None
    for k in range(min(3,nbands)) :
        band = raster.GetRasterBand( 1+k ) # 1-based index
        data = band.ReadAsArray( 0, 0, xcount, ycount )
        print('TODO (raster2img): use all bands')
        if img is None :
            img = data
        else:
            img = np.dstack(( img, data ))
    return img




def load_roi_from_tif( fn, roi_bbox ) :

    # Open source raster (.tif) and find pixel size, nbands and data type
    raster_in = gdal.Open( fn )
    gt = raster_in.GetGeoTransform()
    x0, dx, dxdy, y0, dydx, dy = gt
    assert( abs(dx) == abs(dy) )
    dy = abs(dy)
    pixel_size = dx
    nbands = raster_in.RasterCount
    data_type = raster_in.GetRasterBand(1).DataType

    # ROI_world (x,y) to ROI_pix (j,i)
    x_min, y_max, x_max, y_min = roi_bbox
    j0 = int(round( ( x_min - x0 ) / pixel_size ))
    j1 = int(round( ( x_max - x0 ) / pixel_size ))
    i0 = int(round( - ( y_min - y0 ) / pixel_size ))
    i1 = int(round( - ( y_max - y0 ) / pixel_size ))
    # origin for ROI
    j_off, i_off = int(j0), int(i1) # upper left
    #
    j_count = j1 - j0 + 1
    i_count = i0 - i1 + 1
    assert( j_count > 0 )
    assert( i_count > 0 )
    #
    x_off = x0 + j0 * pixel_size
    y_off = y0 - i1 * pixel_size


    # Create the destination data source
    raster_out = gdal.GetDriverByName('MEM').Create( '', j_count, i_count, nbands+1, data_type )
    raster_out.SetGeoTransform(( x_off, pixel_size, 0, y_off, 0, -pixel_size ))

    # Setting spatial reference of output raster
    #  wkt = raster_in.GetProjection()
    #  from osgeo import osr
    #  srs = osr.SpatialReference()
    #  srs.ImportFromWkt(wkt)
    #  raster_out.SetProjection( srs.ExportToWkt() )
    raster_out.SetProjection( raster_in.GetProjection() )

    # copy ROI
    for k in range(nbands) :
        band = raster_in.GetRasterBand( 1+k ) # 1-based index
        #  print( '##########', xoff, yoff, xcount, ycount )
        data = band.ReadAsArray( j0, i1, j_count, i_count )
        #  print( '##########', k, data.shape )
        band2 = raster_out.GetRasterBand( 1+k ) # 1-based index
        band2.WriteArray( data )

    return raster_out



def get_roi( filenames, rasters_geom, bbox ) :
    idxs = find_intersecting_rasters( rasters_geom, bbox )
    for idx in idxs :
        subimg = load_roi_from_tif( filenames[idx], bbox )
        subimg = raster2img( subimg )
        #
        import matplotlib.pyplot as plt
        plt.imshow( subimg )
        plt.show()






xmin, xmax, ymin, ymax = feature.GetGeometryRef().GetEnvelope()
roi_bbox = xmin, ymax, xmax, ymin
subimg = get_roi( filenames, rasters_geom, roi_bbox )








# find the rasters that intersect each object
print('processing .shp')
#  for k in range(min(7,len(layer))) :
for k in range(len(layer)) :
    #  feat = layer[k]
    #  print('-', k )
    feat = layer.GetFeature( k )
    envelope = feat.GetGeometryRef().GetEnvelope()
    xmin, xmax, ymin, ymax = envelope
    bbox = xmin, ymax, xmax, ymin
    #  print( 'envelope:', envelope, ' -> bbox:', bbox )
    idxs = find_intersecting_rasters( rasters_geom, bbox )
    print( k, [ filenames[idx] for idx in idxs ] )
    #
    if len(idxs) > 0 :
        roi_bbox = bbox
        subimg = get_roi( filenames, rasters_geom, roi_bbox )











#  if __name__ == '__main__':
#      dbg()
