# raster_server

Gdal tips: https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html


## prepare python virtual environment

```
    virtualenv -p python3 venv
    source venv/bin/activate
    pip install Flask
    pip install opencv-python
    pip freeze > requirements.txt

    pip install gunicorn

    # pip install gdal
    # sudo apt-get install libgdal-dev libgdal1-dev
    export CFLAGS=$(gdal-config --cflags)
    pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

    pip install geopandas
```


## tip

for production it can be usefull
```
    pip install gunicorn
    gunicorn raster_server:app
```
and also
```
    flask run --host=0.0.0.0
```


## tip

you can also run the app through
```
    FLASK_ENV=development FLASK_APP=raster_server.py flask run
```

this way it is not required to reestart flask after updating the source code



--------------------------------------------

# data exploration

gdalinfo
ogrinfo

## tip: buscar tipos de cubiertas en el .shp

```
    ogrinfo data/Edif_Clases.shp -al | grep TIPO_CUB | sort | uniq
```
