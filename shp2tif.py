#!/usr/bin/env python3
"""
Convert an OGR File to a Raster

Tip: open the resulting .tif file with gimp
"""


from osgeo import gdal, ogr

# Define pixel_size and NoData value of new raster
#  pixel_size = 25
pixel_size = 1 
pixel_size = .25 
NoData_value = -9999

# Filename of input OGR file
vector_fn = 'test.shp'
vector_fn = '/home/manu/Manu/Jobs/UPNA/JdA/projects/BOLETUS/data/tcsa_shp/SEGU_Lin_SenHorizont.shp'
vector_fn = '/home/manu/tmp/audicana/ciudadela/parcela_urbana.shp'

# Filename of the raster Tiff that will be created
raster_fn = 'test.tif'

# Open the data source and read in the extent
source_ds = ogr.Open(vector_fn)
source_layer = source_ds.GetLayer()
x_min, x_max, y_min, y_max = source_layer.GetExtent()

# Create the destination data source
x_res = int((x_max - x_min) / pixel_size)
y_res = int((y_max - y_min) / pixel_size)
target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
band = target_ds.GetRasterBand(1)
band.SetNoDataValue(NoData_value)

# Rasterize
#  gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[0])
gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[255])
