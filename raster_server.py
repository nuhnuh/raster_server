#!/usr/bin/env python3



from flask import Flask
app = Flask(__name__)



@app.route('/')
def hello():
    return "Hello World!"



@app.route('/image')
def hello_image():
    #  Response(response=response_pickled, status=200, mimetype='image/tiff')
    from flask import send_file
    fn = '/usr/share/nautilus-dropbox/emblems/emblem-dropbox-unsyncable.png'
    with open(fn, 'rb') as file:
        data = file.read()
    print( len(data) )
    import io
    return send_file( io.BytesIO( data ), mimetype='image/png', 
            as_attachment=True, attachment_filename='cambiar.png' )



@app.route('/image1')
def hello_image1():
    #  Response(response=response_pickled, status=200, mimetype='image/tiff')
    # load image
    fn = '/usr/share/themes/AgingGorilla/metacity-1/active-button.png'
    import cv2
    img = cv2.imread( fn )
    # add a mask layer
    import numpy as np
    mask = img[...,0].copy()
    mask[:,:] = 0
    mask[ : , mask.shape[1]//2 : ] = 255
    img = np.dstack( (img, mask) )
    # encode img as tif
    _, data = cv2.imencode( '.png', img )
    # send response
    from flask import send_file
    import io
    return send_file( io.BytesIO( data ), mimetype='image/png' )



@app.route('/image2')
def hello_image2():
    #  Response(response=response_pickled, status=200, mimetype='image/tiff')
    # load image
    # TODO: target_ds = gdal.GetDriverByName('MEM').Create('', x_res, y_res, gdal.GDT_Byte)
    fn = '/usr/share/themes/AgingGorilla/metacity-1/active-button.png'
    import cv2
    img = cv2.imread( fn )
    # add a mask layer
    import numpy as np
    mask = img[...,0].copy()
    mask[:,:] = 0
    mask[ : , mask.shape[1]//2 : ] = 255
    img = np.dstack( (img, mask) )
    # encode img as tif
    _, data = cv2.imencode( '.tif', img )
    # send response
    from flask import send_file
    import io
    return send_file( io.BytesIO( data ), mimetype='image/tif', 
            as_attachment=True, attachment_filename='cambiar.tif' ) # attachment options work, the problem was firefox (client.py works)




@app.route('/image3')
def hello_image3():
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr

    # Filename of the raster Tiff that will be created
    raster_in_fn = '/home/manu/tmp/audicana/mtn25_epsg25830_0141-2.tif'
    # Filename of input OGR file
    vector_fn = 'data/ciudadela.shp'

    # Open source raster (.tif) and find pixel size, nbands and data type
    raster_in = gdal.Open( raster_in_fn )
    gt = raster_in.GetGeoTransform()
    x0, dx, dxdy, y0, dydx, dy = gt
    assert( abs(dx) == abs(dy) )
    dy = abs(dy)
    pixel_size = dx
    nbands = raster_in.RasterCount
    data_type = raster_in.GetRasterBand(1).DataType

    # Open the data source, set filter to select a given parcel and get its extent
    source_ds = ogr.Open( vector_fn , 0 ) # 0 means read-only. 1 means writeable.
    #  drv = ogr.GetDriverByName( 'Memory' )
    source_layer = source_ds.GetLayer()
    source_layer.SetAttributeFilter('cparcela = 1115')
    x_min, x_max, y_min, y_max = source_layer.GetExtent()

    # Create the destination data source
    x_res = int( (x_max - x_min) / pixel_size )
    y_res = int( (y_max - y_min) / pixel_size )
    raster_out_fn = '/tmp/ciudadela.tif'
    raster_out = gdal.GetDriverByName('GTiff').Create( raster_out_fn, x_res, y_res, nbands+1, data_type )
    #  target_ds = gdal.GetDriverByName('MEM').Create( '', x_res, y_res, nbands, data_type )
    raster_out.SetGeoTransform(( x_min, pixel_size, 0, y_max, 0, -pixel_size ))

    # Setting spatial reference of output raster
    #  wkt = raster_in.GetProjection()
    #  srs = osr.SpatialReference()
    #  srs.ImportFromWkt(wkt)
    #  raster_out.SetProjection( srs.ExportToWkt() )
    raster_out.SetProjection( raster_in.GetProjection() )

    # copy ROI
    def world2pix( x_world, y_world ) : # gt = raster.GetGeoTransform()
        x0, dx, dxdy, y0, dydx, dy = gt
        print( gt )
        x = ( x_world - x0 ) // dx
        y = ( y_world - y0 ) // dy
        return x, y
    x_min_pix, y_max_pix = world2pix( x_min, y_max )
    for k in range(nbands) :
        xoff, yoff = int(x_min_pix), int(y_max_pix)
        xcount, ycount = int(x_res), int(y_res)
        band = raster_in.GetRasterBand( 1+k ) # 1-based index
        #  print( '##########', xoff, yoff, xcount, ycount )
        data = band.ReadAsArray( xoff, yoff, xcount, ycount )
        #  print( '##########', k, data.shape )
        band2 = raster_out.GetRasterBand( 1+k ) # 1-based index
        band2.WriteArray( data )

    # Rasterize
    band = raster_out.GetRasterBand(4)
    #  band.SetNoDataValue( 200 )
    gdal.RasterizeLayer( raster_out, [4], source_layer, burn_values=[255] )

    return '/tmp/ciudadela has been generated'



@app.route('/<name>')
def hello_name(name):
    return 'Hello {}!'.format(name)



#  @app.route('/<x0>')
#  def hello_name(name):
#      return 'x0 = {}, y0 = {}, x1 = {}, y1 = {}'.format( x0, y0, x1, y1 )



if __name__ == '__main__':
    app.run()
