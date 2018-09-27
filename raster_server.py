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


@app.route('/<name>')
def hello_name(name):
    return 'Hello {}!'.format(name)


#  @app.route('/<x0>')
#  def hello_name(name):
#      return 'x0 = {}, y0 = {}, x1 = {}, y1 = {}'.format( x0, y0, x1, y1 )



if __name__ == '__main__':
    app.run()
