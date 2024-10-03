import argparse
import cv2 as cv
import http.server
import io
import json
import libcamera
import numpy as np
import os
import picamera2
import pprint
import queue
import socket
import sys
import threading
import time
import urllib.parse

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

class KlipperApiClient:

    def __init__(self, sock):
        self.sock = sock
        self.next_id = 1
        self.replies = queue.Queue()
        self.notifications = queue.Queue()

    def send(self, method, params={}, response=False):

        o = {'method':method, 'params':params}

        id = None
        if response:
            o['id'] = id = self.next_id
            self.next_id += 1
        
        msg = json.dumps(o).encode('utf-8') + b'\x03'
        print(f"send: {msg=}")

        i = 0
        while i < len(msg):
            sent = self.sock.send(msg[i:])
            if sent == 0:
                raise RuntimeError("disconnected")
            i += sent

        if id is None:
            return

        repl = self.replies.get()
        while repl['id'] != id:
            repl = self.replies.get()

        return repl

class KlipperApiThread(threading.Thread):

    def __init__(self, klipper):
        super().__init__(daemon=True)
        self.klipper = klipper

    def run(self):

        partial_reply = b''
        while True:
            
            data = self.klipper.sock.recv(4096)
            if len(data) == 0:
                raise RuntimeError("disconnected")

            data = partial_reply + data
            replies = data.split(b'\x03')
            partial_reply = replies.pop()

            for repl in replies:
                o = json.loads(repl.decode('utf-8'))
                if 'id' in o:
                    self.klipper.replies.put(o)
                else:
                    self.klipper.notifications.put(o)
        
class BotRequestHandler(http.server.BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"
    
    def do_GET(self):
        
        parts = urllib.parse.urlsplit(self.path)
        path, params = parts.path, dict(urllib.parse.parse_qsl(parts.query))

        if path.startswith('/camera'):
            self.get_camera(parts.path[7:], params)
        elif path.startswith('/bot'):
            self.get_bot(parts.path[4:], params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        
        parts = urllib.parse.urlsplit(self.path)
        path, params = parts.path, dict(urllib.parse.parse_qsl(parts.query))
        
        if path.startswith('/camera'):
            self.post_camera(parts.path[7:], params)
        elif path == '/bot':
            self.post_bot(parts.path[4:], params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def get_bot(self, path, params):

        if path == '/notifications':
            self.get_bot_notifications(params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def get_bot_notifications(self, params):

        klipper = self.server.klipper

        timeout = dict(params).get('timeout', None)
        if timeout is not None:
            timeout = float(timeout)
            
        block = timeout is not None
        notifications = []
        try:
            x = klipper.notifications.get(block=block, timeout=timeout)
            while True:
                notifications.append(x)
                # after we receive the first response just drain the queue, but don't block again
                x = klipper.notifications.get(block=False)
        except queue.Empty:
            pass
        
        data = json.dumps(notifications)
        self.send_json_response(data)

    def post_bot(self, path, params):

        klipper = self.server.klipper
        content_length = int(self.headers.get('Content-Length'))

        data = self.rfile.read(content_length)
        data = json.loads(data.decode('utf-8'))

        k_method = data['method']
        k_params = data.get('params', {})
        k_response = data.get('response', False)
        repl = klipper.send(k_method, k_params, k_response)

        if repl is not None:
            self.send_json_response(json.dumps(repl))
        else:
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Length', '0')
            self.end_headers()

    def get_camera(self, path, params):

        if path.startswith('/image/'):
            self.get_camera_image(path[7:], params)
        elif path == '/config':
            self.get_camera_config(params)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def get_camera_config(self, params):
        
        camera = self.server.camera
        config = _clean_config(camera.camera_configuration())

        data = json.dumps(config)

        self.send_json_response(data)
        
    def send_json_response(self, data):

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()

        self.wfile.write(bytes(data, 'utf-8'))

    def get_camera_image(self, path, params):

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
        self.send_header("X-Puzzler", json.dumps({'image_metadata':metadata}))
        self.end_headers()

        # get a view of the data without copying
        self.wfile.write(image)

    def log_request(self, code='-', size='-'):
        if code == 200:
            return
        super().log_request(code, size)

class BotServer(http.server.ThreadingHTTPServer):

    allow_reuse_address = True
    allow_reuse_port = True

    def handle_error(self, request, client_address):
        x = sys.exception()
        if isinstance(x, ConnectionResetError):
            print('ConnectionResetError: during processing of request from',
                  client_address, file=sys.stderr)
            return
        return super().handle_error(request, client_address)

def initialize_klipper(args):

    print("initializing klipper")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.klippy)
    
    return KlipperApiClient(sock)
        
def initialize_camera(args):

    print("initializing camera")

    camera = picamera2.Picamera2(camera_num=args.device)

    config = {
        'main': {
            'size': (args.width, args.height),
            'format': 'RGB888'
        },
        'lores': {
            'size': (args.width//4, args.height//4),
            'format': 'YUV420'
        },
        'display': 'main',
        'transform': libcamera.Transform(hflip=1, vflip=1)
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

    camera.set_controls({'AeEnable':True})
    
    print("camera started:")
    o = {'config': _clean_config(camera.camera_configuration()),
         'controls': camera.camera_controls}
    pprint.pp(o, sort_dicts=True)

    return camera

def main():

    parser = argparse.ArgumentParser(conflict_handler='resolve')
    parser.add_argument('-d', '--device', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=4056)
    parser.add_argument('-h', '--height', type=int, default=3040)
    parser.add_argument('--preview', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--klippy', type=str, default='/tmp/klippy_uds')

    args = parser.parse_args()

    camera = initialize_camera(args)
    klipper = initialize_klipper(args)
    thread = KlipperApiThread(klipper)

    with BotServer(("", args.port), BotRequestHandler) as httpd:
        httpd.camera = camera
        httpd.klipper = klipper
        thread.start()
        print(f"serving at port {args.port}, pid {os.getpid()}")
        httpd.serve_forever()

if __name__ == "__main__":
    main()
