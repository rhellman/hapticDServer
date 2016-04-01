#!/usr/local/bin/python
'''
Program runs to longitudinal locaiton of edges('zipper')
in biotac from http server request with image tag in http
body

'''

# Webserver imports
import SimpleHTTPServer
import SocketServer
import logging
import cgi
import os
import shutil
import signal
import sys
import json

from pyHapticTools.dotReward import rewardAssignment
from pyHapticTools.deepNets import directionDNN as dn, regionDNN as rn

dn.init()
rn.init()

def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

class ServerHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):

    def _writeheaders(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.warning("======= GET STARTED =======")
        logging.warning(self.headers)

        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        logging.warning("======= POST STARTED =======")
        logging.warning(self.headers)

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     })
        logging.warning("======= POST VALUES =======")
        #for item in form.list:
            #logging.warning(item)
        #logging.warning("\n")
        if form.has_key("classifier"):
            test = form.getvalue("classifier")
            text = test.split(',')
            inputString = map(float, text) #list of electrode vals
            print "inputString: ", inputString
            self._writeheaders()
            listOutput = np.array([1.0, 2.0, 3.4])
            #rn.predict(inputString)
            #dn.predict(inputString)
            #listOutput needs to be updated to include rn/dn.predict()
            self.wfile.write(json.dumps({'classifier': listOutput.tolist(), 'region': 1}))

        if not form.has_key("image"): return
        fileitem = form["image"]
        if not fileitem.file: return 

        # save file to current folter
        outpath = os.path.join(os.getcwd() + '/jpg', fileitem.filename)
        print 'outpath ', outpath
        with open(outpath, 'wb') as fout:
            print 'fout - ', fout
            shutil.copyfileobj(fileitem.file, fout, 100000)
            print 'Saved: fileitem.filename'

        findOffsetForRewardAssignment(outpath)

        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

signal.signal(signal.SIGINT, signal_handler)

PORT = 8000
I = ""

Handler = ServerHandler
# Allows us to reopen port quickly after program restart
SocketServer.TCPServer.allow_reuse_address = True
httpd = SocketServer.TCPServer(("", PORT), Handler)

print "Serving at: http://%(interface)s:%(port)s" % dict(interface=I or "localhost", port=PORT)
httpd.serve_forever()









