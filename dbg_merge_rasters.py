#!/usr/bin/env python3
"""
    TODO: merge subimgs of the tifs intersections
"""



import glob
from osgeo import gdal
from osgeo import ogr
import numpy as np
import os
import matplotlib.pyplot as plt



dst_dir = '/tmp'
dst_dir = os.path.expanduser( dst_dir )



def get_tif_filenames() :
    src_dir = 'data2/orto/*.tif'
    filenames = glob.glob( src_dir )
    return filenames



def load_rasters_md( filenames ) :
    rasters_gt = []
    rasters_sz = []
    for fn in filenames :
        raster = gdal.Open( fn )
        raster_gt = raster.GetGeoTransform()
        raster_sz = raster.RasterXSize, raster.RasterYSize
        rasters_gt.append( raster_gt )
        rasters_sz.append( raster_sz )
    return rasters_gt, rasters_sz



def get_bbox( gt, sz ) :
    x0, dx, dxdy, y0, dydx, dy = gt
    x_tl, y_tl = x0, y0
    x_br = x0 + dx * sz[0]
    y_br = y0 + dy * sz[1]
    bbox = x_tl, y_tl, x_br, y_br
    return bbox



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



def geom2bbox( geom ) :
    xmin, xmax, ymin, ymax = geom.GetEnvelope()
    bbox = xmin, ymax, xmax, ymin
    return bbox



def find_intersecting_rasters( rasters_geom, roi_bbox ) :
    # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    roi_geom = bbox2geom( roi_bbox )
    result = []
    for idx, raster_geom in enumerate(rasters_geom) :
        if roi_geom.Intersect( raster_geom ) :
            result.append( idx )
    return result




def raster2img( raster ) :
    nbands = raster.RasterCount
    xcount = raster.RasterXSize
    ycount = raster.RasterYSize
    img = None
    #  for k in range(min(3,nbands)) :
    for k in range(nbands) :
        band = raster.GetRasterBand( 1+k ) # 1-based index
        data = band.ReadAsArray( 0, 0, xcount, ycount )
        print('WARNING: raster2img uses only 3 bands')
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
    j_count = j1 - j0
    i_count = i0 - i1
    assert( j_count > 0 )
    assert( i_count > 0 )
    assert( j_count <= raster_in.RasterXSize )
    assert( i_count <= raster_in.RasterYSize )
    #
    x_off = x0 + j0 * pixel_size
    y_off = y0 - i1 * pixel_size

    # Create the destination data source
    raster_out = gdal.GetDriverByName('MEM').Create( '', j_count, i_count, nbands, data_type )
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
        print('j0,i1,j_count,i_count:', j0, i1, j_count, i_count)
        data = band.ReadAsArray( j0, i1, j_count, i_count )
        print('data.shape:', data.shape)
        #  print( '##########', k, data.shape )
        band2 = raster_out.GetRasterBand( 1+k ) # 1-based index
        band2.WriteArray( data )
        # does not work :/
        #  band2.SetColorInterpretation( band.GetRasterColorInterpretation() )
        #  band2.SetRasterColorInterpretation( gdal.GCI_Undefined )

    return raster_out



def get_roi( filenames, rasters_geom, roi_bbox ) :
    print( 'roi_bbox:', roi_bbox )
    idxs = find_intersecting_rasters( rasters_geom, roi_bbox )
    roi_geom = bbox2geom( roi_bbox )
    for idx in idxs :
        print( 'raster_bbox:', geom2bbox(rasters_geom[idx]) )
        if geom2bbox(roi_geom.Intersection(rasters_geom[idx])) == roi_bbox :
            raster = load_roi_from_tif( filenames[idx], roi_bbox )
            print( 'a tif contains the full roi' )
            return raster
    print( 'no tif contains the full roi' )
    raise 'TODO'
    #
    for idx in idxs :
        # intersect each raster with the roi
        ibbox = geom2bbox(roi_geom.Intersection(rasters_geom[idx]))
        # load subimg of the intersection
        iraster = load_roi_from_tif( filenames[idx], ibbox )

        subimg = raster2img( subimg )
        #
        plt.imshow( subimg )
        plt.show()



def test1() :
    # load rasters (tifs) geometry
    filenames = get_tif_filenames()
    print( len(filenames), ' tifs available' )
    assert( len(filenames) > 0 ), 'folder empty?'
    #  filenames = filenames[:10]
    rasters_gt, rasters_sz = load_rasters_md( filenames )
    [ print( fn, get_bbox(gt,sz) ) for fn, gt, sz in zip(filenames, rasters_gt, rasters_sz) ]
    #
    rasters_geom = [ bbox2geom(get_bbox(gt,sz)) for gt, sz in zip(rasters_gt, rasters_sz) ]
    
    
    
    # find the intersection of the rasters
    print('------')
    print( 'finding the intersection of the 4 tifs:' )
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
    
    
    
    # check that bbox2geom and geom2bbox work OK
    print('------')
    print( 'checking bbox2geom and geom2bbox:' )
    print( geom )
    print( geom2bbox( geom ) )
    print( bbox2geom(geom2bbox( geom )) )
    print( geom2bbox(bbox2geom(geom2bbox( geom ))) )
    
    
    
    # define a 10 m widther roi than the intersection of the 4 tifs
    print('------')
    print( 'defining a ROI' )
    roi_bbox = geom2bbox( geom )
    print( 'roi_bbox:', roi_bbox )
    roi_bbox = (614180.0-10, 4734050.0+10, 614290.0+10, 4733940.0-10)
    print( 'roi_bbox:', roi_bbox )
    #
    for geom_k in rasters_geom :
        xy = np.asarray( geom_k.Boundary().GetPoints() )
        x, y = xy[:,0], xy[:,1]
        plt.plot( x, y )
    xy = np.asarray( bbox2geom(roi_bbox).Boundary().GetPoints() )
    x, y = xy[:,0], xy[:,1]
    plt.plot( x, y, ':k' )
    plt.axis('equal')
    plt.show()
    
    
    
    #  # dbg
    #  roi_bbox = geom2bbox( geom )
    #  for idx in range(len(filenames)) :
    #      print('-- ', idx)
    #      img = load_roi_from_tif( filenames[idx], roi_bbox )
    #      img = raster2img( img )
    #      print(img)
    
    
    
    
    # load pixels in the intersection of images
    roi_raster = get_roi( filenames, rasters_geom, roi_bbox )
    roi_img = raster2img( roi_raster )
    plt.imshow( roi_img )
    plt.show()
    
    
    raise 'what if roi_bbox is not fully included in any of the tifs?'



def draw_mask( subraster, source_layer ) :

    nbands = subraster.RasterCount
    data_type = subraster.GetRasterBand(1).DataType

    subraster_mask = gdal.GetDriverByName('MEM').Create( '', 
            subraster.RasterXSize, subraster.RasterYSize, 1, data_type )
    subraster_mask.SetGeoTransform( subraster.GetGeoTransform() )

    # Rasterize
    band = subraster_mask.GetRasterBand(1)
    band.SetNoDataValue( 128 )
    gdal.RasterizeLayer( subraster_mask, [1], source_layer, burn_values=[255] )

    return subraster_mask



def test2() :

    # load rasters (tifs) geometry
    filenames = get_tif_filenames()
    print( len(filenames), ' tifs available' )
    assert( len(filenames) > 0 ), 'folder empty?'
    #  filenames = filenames[:10]
    rasters_gt, rasters_sz = load_rasters_md( filenames )
    rasters_geom = [ bbox2geom(get_bbox(gt,sz)) for gt, sz in zip(rasters_gt, rasters_sz) ]

    # load shp
    print( 'loading shp..' )
    shp_fn = 'data2/shp/Edif_Clases.shp'
    shp = ogr.Open( shp_fn , 0 ) # 0 means read-only. 1 means writeable.
    layer = shp.GetLayer()
    print( 'len(layer):', len(layer) )
    #
    for idx, feature in enumerate(layer) :
        print( 'processing feature', idx )
        envelope = feature.GetGeometryRef().GetEnvelope()
        xmin, xmax, ymin, ymax = envelope
        #  roi_bbox = xmin, ymax, xmax, ymin
        # extend roi
        margin = 14
        roi_bbox = xmin-margin, ymax+margin, xmax+margin, ymin-margin

        #  # dbg
        #  idxs = find_intersecting_rasters( rasters_geom, roi_bbox )
        #  if len(idxs) < 1 :
        #      assert(False), 'roi does not intersect any raster'

        #
        subraster = get_roi( filenames, rasters_geom, roi_bbox )

        #  # save subraster
        #  fn = '{:06d}.tif'.format( idx )
        #  fn = os.path.join( dst_dir, fn )
        #  print( 'saving ', fn )
        #  gdal.GetDriverByName('GTiff').CreateCopy( fn, subraster, strict=0 ) # strict=1 : report errors

        # subraster_mask
        shp_idx = ogr.GetDriverByName('Memory').CreateDataSource('')
        source_layer = shp_idx.CreateLayer('states_extent')
        source_layer.CreateFeature( feature )
        subraster_mask = draw_mask( subraster, source_layer )

        # dbg
        #  continue
        img = raster2img( subraster )
        mask = raster2img( subraster_mask )
        print( 'mask.shape:', mask.shape )
        plt.subplot(2,3,1)
        plt.imshow( img[:,:,0] )
        plt.title('0')
        plt.subplot(2,3,2)
        plt.imshow( img[:,:,1] )
        plt.title('1')
        plt.subplot(2,3,3)
        plt.imshow( img[:,:,2] )
        plt.title('2')
        plt.subplot(2,3,4)
        plt.imshow( img[:,:,3] )
        plt.title('3')
        plt.subplot(2,3,5)
        plt.imshow( mask )
        plt.title('mask')
        plt.show()



if __name__ == '__main__':
    #  test1()
    test2()
