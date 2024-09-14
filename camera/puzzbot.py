import socket
import threading

class KlipperApiClient:

    def __init__(self, sock):
        self.sock = sock
        self.next_id = 1
        self.partial_reply = b''
        self.replies = collections.deque()

    def send(self, method, params={}, response=False):

        o = {'method':method, 'params':params}

        id = None
        if response:
            o['id'] = id = self.next_id
            self.next_id += 1
        
        msg = self._format_request(o)
        
        i = 0
        while i < len(msg):
            sent = self.sock.send(msg[i:])
            if sent == 0:
                raise RuntimeError("disconnected")
            i += sent

        return id

    def recv(self):

        while not self.replies:
            
            data = self.sock.recv(4096)
            if len(data) == 0:
                raise RuntimeError("disconnected")

            self.replies = collections.deque(self.partial_reply + data.split(b'\x03'))
            self.partial_reply = replies.pop()
            
        return self._parse_reply(self.replies.popleft())
        
    def _format_request(self, o):
        return json.dumps(o).encode('utf-8') + b'\x03'

    def _parse_reply(self, s):
        return json.loads(s)

def KlipperReceiveThread(threading.Thread):

    def __init__(self, klipper, callback):
        super().__init__(daemon=True)
        self.klipper = klipper
        self.callback = callback

    def run(self):
        while True:
            r = self.klipper.recv()
            self.callback(r)

class BotController:

    def __init__(self, sock):
        self.klipper = KlipperApiClient(sock)
        self.recv_queue = queue.Queue()
        self.recv_thread = KlipperRecvThread(klipper, self.klipper_callback)
        self.recv_thread.start()

    def klipper_callback(self, r):
        if r.get('id'):
            self.recv_queue.put(r)
        else:
            print(r)
    
    def connect(self):
        params = {
            'objects': {
                'motion_report':['live_position'],
                'webhooks':None
            },
            'template': {
                'key': 'objects/subscribe'
            }
        }
        self.klipper.send('objects/subscribe', params)
        
        params = {
            'template': {
                'key': 'gcode/subscribe_output'
            }
        }
        self.klipper.send('gcode/subscribe_output', params)

    def home(self):
        # always home Z first so it is up and off the table
        self.send_gcode('G28 Z')
        self.send_gcode('G28 X Y')
        self.send_gcode('MANUAL_STEPPER STEPPER=stepper_a ENABLE=1 POS=0')

        id = self.klipper.send('objects/query', {'toolhead':None}, True)
        r = self.klipper.await_response(id)
        print(r)

    def move_to(self, *, x=None, y=None, z=None, f=None):
        args = ['G1']
        if x is not None:
            args.append(f"X{x:.2f}")
        if y is not None:
            args.append(f"Y{y:.2f}")
        if z is not None:
            args.append(f"Z{y:.2f}")
        if f is not None:
            args.append(f"F{f:.0f}")

        self.send_gcode(args.join(' '))

    def turn_to(self, *, a=None):

        self.send_gcode(f"MANUAL_STEPPER STEPPER=stepper_a MOVE={a:.1f}")

    def finish_moves(self):
        id = self.send_gcode("M400", response=True)
        self.await_response(id)

    def send_gcode(self, script, response=False):
        return self.klipper.send(
            'gcode/script', {'script':script}, response)

    def await_response(self, id):
        while True:
            r := self.recv_queue.get()
            if r.get('id') == id:
                return r
            print("Unexpected response:", r)
    
def scan_area_iterator(rect, dxdy):

    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    dx, dy = dxdy

    c, r = int(w / dx), int(h / dy)
    x = x0 + 0.5 * (w - c * dx)
    y = y0 + 0.5 * (h - r * dy)

    for j in range(r):
        for i in range(c):
            # even rows are left to right, odds rows are right to left
            if j & 1:
                yield (x + (c-1-i)*dx, y + j*dy)
            else:
                yield (x + i*dx, y + j*dy)
            
def find_pieces(bot, rect):

    pieces = []

    def isdisjoint(r):
        return not any(rect_intersects(r, i) for i in pieces)

    for xy in scan_area_iterator(rect, bot.camera.props.fov_mm * .5):

        bot.move_to(xy)

        image = bot.capture_image()
        proj = bot.camera_to_world(xy)
        for p in find_pieces(image):
            r = Rect(proj(p.ll), proj(p.ur))
            if isdisjoint(r):
                pieces.append(r)

    return pieces

class Rect:

    def __init__(self, ll, ur):
        self.ll = np.array(ll)
        self.ur = np.array(ur)

    def intersects(self, other):
        return np.all(self.ll < other.ur & other.ll < self.ur)
    
class Projection:

    def __init__(self, scale, translate):
        self.scale = scale
        self.translate = translate

    def __call__(self, ij):
        return self.scale * ij + self.translate

    def get_inverse(self):
        return Projection(1/self.scale, -self.translate/self.scale)

class PuzzBot:

    def __init__(self, bot, camera):
        self.bot = bot
        self.camera = camera
        self.props = {}

    def camera_to_world(self, xy):
        return Projection(self.props['camera.scale'], self.props['camera.offset'] + xy)

