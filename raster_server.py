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
    return send_file(io.BytesIO( data ),
                     attachment_filename='logo.png',
                     mimetype='image/png')


@app.route('/<name>')
def hello_name(name):
    return "Hello {}!".format(name)


if __name__ == '__main__':
    app.run()
