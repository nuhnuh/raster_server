# raster_server


## prepare python virtual environment

```
    virtualenv -p python3 venv
    source venv/bin/activate
    pip instal Flask
    pip instal opencv-python
    pip freeze > requirements.txt

    pip install gunicorn
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

