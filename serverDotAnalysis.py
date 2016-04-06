#!/usr/local/bin/python
'''
Program runs to longitudinal locaiton of edges('zipper')
in biotac from http server request with image tag in http
body

'''

# Webserver imports
from __future__ import print_function
import SimpleHTTPServer
import SocketServer
import logging
import cgi
import os
import shutil
import signal
import sys
import json
import numpy as np

from pyHapticTools.dotReward import rewardAssignment

from pyHapticTools.dnnClassifier import directionDNN as dn
from pyHapticTools.dnnClassifier import regionDNN as rn

# Training block
# dn.train(num_steps = 15001)
# rn.train(num_steps = 15001)
# os.system('say "your program has finished"')

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

        if form.has_key("classifier"):
            test = form.getvalue("classifier")
            text = test.split(',')
            inputRow = np.array(map(float, text)).astype(np.float32) #list of electrode vals
            inputRow = np.reshape(inputRow, (1,-1))
            print('inputRow: \n\t{} \n\tinputRow.shape: {}'.format(inputRow, inputRow.shape))
            self._writeheaders()
            listOutput = np.array([1.0, 2.0, 3.4])
            #Prediction
            regionPrediction    = rn.predict(inputRow)
            directionPrediction = dn.predict(inputRow)

            self.wfile.write(json.dumps({'direction': directionPrediction, 'region': regionPrediction}))

        if not form.has_key("image"): return
        fileitem = form["image"]
        if not fileitem.file: return 

        # save base jpg file to current folter 
        outpath = os.path.join(os.getcwd() + '/jpg', fileitem.filename)
        print('outpath ', outpath)
        with open(outpath, 'wb') as fout:
            print('fout - ', fout)
            shutil.copyfileobj(fileitem.file, fout, 100000)
            print('Saved: fileitem.filename')

        findOffsetForRewardAssignment(outpath)
        SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

signal.signal(signal.SIGINT, signal_handler)

PORT = 8000
I = ""

Handler = ServerHandler
# Allows us to reopen port quickly after program restart
SocketServer.TCPServer.allow_reuse_address = True
httpd = SocketServer.TCPServer(("", PORT), Handler)

print("Serving at: http://{interface}:{port}".format(interface=I if 'interface' in locals() else "localhost", port=PORT))
httpd.serve_forever()









