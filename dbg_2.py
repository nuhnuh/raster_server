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
    return shapes.iloc[103]



def dbg2( geoms ):
    import rasterio
    from rasterio.tools.mask import mask
    # the polygon GeoJSON geometry
    #  geoms = [{'type': 'Polygon', 'coordinates': [[(250204.0, 141868.0), (250942.0, 141868.0), (250942.0, 141208.0), (250204.0, 141208.0), (250204.0, 141868.0)]]}]
    # load the raster, mask it by the polygon and crop it
    fn = '/home/manu/tmp/audicana/mtn25_epsg25830_0141-2.tif'
    with rasterio.open( fn ) as src:
        out_image, out_transform = mask( src, geoms, crop=True )
    out_meta = src.meta.copy()

    # save the resulting raster  
    out_meta.update({"driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform})

    with rasterio.open("/tmp/ciudadela_rasterio.tif", "w", **out_meta) as dest:
        dest.write(out_image)



# TODO: In [59]: len(aux.geometry.boundary.xy[0])
print( _get_poligon().geometry.bounds )
x, y = _get_poligon().geometry.boundary.xy
print([ (xi, yi) for xi, yi in zip(x, y) ])
# TODO: generate geoms
geoms = [{'type': 'Polygon', 'coordinates': [[ (xi, yi) for xi, yi in zip(x, y) ]]}]
dbg2( geoms )

#  if __name__ == '__main__':
#      dbg()
