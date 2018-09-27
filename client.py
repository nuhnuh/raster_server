#!/usr/bin/env python3



import requests


url = 'http://127.0.0.1:5000/image2'
req = requests.get( url )

print( req.headers.get('content-type') )
print( req.headers.get('content-length') )
print( req.headers.get('content-disposition') )
print( req.headers )
print( dir(req) )
print( 'links:', req.links )
print( 'json:', req.json )
print( 'encoding:', req.encoding )
print( 'len(content):', len(req.content) )


fn = '/tmp/cambiar.tif'
with open( fn, 'wb' ) as file:
    file.write( req.content )
print('saved as', fn)

#  req.content

#  if __name__ == '__main__':
