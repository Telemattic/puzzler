import http.server
import io
import os
import socketserver
import argparse
import time
import urllib.parse
import libcamera
import picamera2
import json
import cv2 as cv
import numpy as np

from http import HTTPStatus

def _list_to_tuple(x):

    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, dict):
        return dict((k, _list_to_tuple(v)) for k, v in x.items())
    else:
        return x

def _clean_config(x):

    ret = dict()
    for k, v in x.items():
        if k in ('transform', 'colour_space'):
            continue
        if k in ('NoiseReductionMode',):
            v = int(v)
        elif isinstance(v, dict):
            v = _clean_config(v)
        ret[k] = v
    return ret

class CameraRequestHandler(http.server.BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"
    
    def do_GET(self):
        
        parts = urllib.parse.urlsplit(self.path)
        params = dict(urllib.parse.parse_qsl(parts.query))
        if parts.path.startswith('/image/'):
            self.get_image(parts.path[7:], params)
        elif parts.path == '/config':
            self.get_config(params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):

        parts = urllib.parse.urlsplit(self.path)
        params = dict(urllib.parse.parse_qsl(parts.query))
        if parts.path == '/config':
            self.set_config(params)
        elif parts.path == '/controls':
            self.set_controls(params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def get_config(self, params):
        
        camera = self.server.camera
        config = _clean_config(camera.camera_configuration())
        
        print(f"get_config: {config}")

        data = json.dumps(config)

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

        self.wfile.write(bytes(data, 'utf-8'))

    def set_config(self, params):

        content_length = int(self.headers.get('Content-Length'))
        data = self.rfile.read(content_length)
        config = _list_to_tuple(json.loads(data))

        print(f"set_config: {params=} {config=}")

        main = config.get('main', {})
        lores = config.get('lores', {})
        display = config.get('display', 'lores')
        controls = config.get('controls', {})

        camera = self.server.camera

        cc = camera.create_still_configuration(**config) #main=main, lores=lores, display=display, controls=controls)
        camera.align_configuration(cc)
            
        camera.stop()
        camera.configure(cc)
        camera.start()

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def set_controls(self, params):
        
        content_length = int(self.headers.get('Content-Length'))

        data = self.rfile.read(content_length)
        print(f"{len(data)=}")

        controls = _list_to_tuple(json.loads(data))
        print(f"{controls=}")

        camera = self.server.camera

        camera.set_controls(controls)

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def get_image(self, path, params):

        v = path.split('.')
        if len(v) != 2:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        name = v[0]
        if name not in ('main', 'lores'):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        
        fmt = v[1]
        if fmt not in ('jpeg', 'png'):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        camera = self.server.camera
        request = camera.capture_request()
        try:
            config = request.config[name]
            if config['format'] == 'YUV420':
                image_or_buffer = request.make_buffer(name)
            else:
                image_or_buffer = request.make_image(name)
            metadata = request.get_metadata()
        finally:
            request.release()

        if config['format'] == 'YUV420':
    
            w, h = config['size']
            stride = config['stride']
    
            array = np.array(image_or_buffer, copy=False, dtype=np.uint8)
            array = array.reshape((h * 3 // 2, stride))
    
            image = cv.cvtColor(array, cv.COLOR_YUV420p2RGB)

            del array
    
            image = cv.imencode('.' + fmt, image)[1]
        else:
            fp = io.BytesIO()
            camera.helpers.save(image_or_buffer, metadata, file_output=fp, format=fmt)
            image = fp.getbuffer()

        del image_or_buffer

        self.send_image_and_metadata(image, fmt, metadata)

    def send_image_and_metadata(self, image, fmt, metadata):

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/" + fmt)
        self.send_header("Content-Length", str(len(image)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Bricksort", json.dumps({'image_metadata':metadata}))
        self.end_headers()

        # get a view of the data without copying
        self.wfile.write(image)

    def log_request(self, code='-', size='-'):
        if code == 200:
            return
        super().log_request(code, size)

class CameraServer(http.server.ThreadingHTTPServer):

    allow_reuse_address = True
    allow_reuse_port = True

def initialize_camera(args):

    print("initializing camera")

    camera = picamera2.Picamera2(camera_num=args.device)

    config = {
        'main':{
            'size': (args.width, args.height)
        },
        'display': 'main',
        'buffer_count': 2
    }
    
    if args.preview:
        config['lores'] = {'size':(640, 480)}
        config['display'] = 'lores'
    
    cc = camera.create_still_configuration(**config)
    camera.align_configuration(cc)
    camera.configure(cc)
    
    if args.preview:
        print("starting preview")
        camera.start_preview(picamera2.Preview.QTGL)

    camera.start()
    time.sleep(2)

    print(f"camera started, config={camera.camera_configuration()}")

    return camera

def main():

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=4056)
    parser.add_argument('-h', '--height', type=int, default=3040)
    parser.add_argument('--preview', action='store_true', default=False)
    parser.add_argument('--port', default=8000)

    args = parser.parse_args()

    camera = initialize_camera(args)

    with CameraServer(("", args.port), CameraRequestHandler) as httpd:
        httpd.camera = camera
        print(f"serving at port {args.port}, pid {os.getpid()}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()
