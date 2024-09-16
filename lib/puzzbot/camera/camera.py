import cv2 as cv
import json
import numpy as np
import requests
import time

class ICamera:
    pass

class OpenCVCamera(ICamera):

    def __init__(self, camera_id, target_w, target_h):
        print("Opening camera...")
        params = [cv.CAP_PROP_FRAME_WIDTH, target_w, cv.CAP_PROP_FRAME_HEIGHT, target_h]
        cam = cv.VideoCapture(camera_id, cv.CAP_MSMF, params=params)
        assert cam.isOpened()

        def get_frame_size(cam):
            return (int(cam.get(cv.CAP_PROP_FRAME_WIDTH)),
                    int(cam.get(cv.CAP_PROP_FRAME_HEIGHT)))
        
        actual_w, actual_h = get_frame_size(cam)
        print(f"Camera opened: {actual_w}x{actual_h}")

        cam.set(cv.CAP_PROP_FRAME_WIDTH, target_w)
        cam.set(cv.CAP_PROP_FRAME_HEIGHT, target_h)

        actual_w, actual_h = get_frame_size(cam)

        print(f"Capture size set to {actual_w}x{actual_h}")

        self._camera = cam
        self._exposure = self._camera.get(cv.CAP_PROP_EXPOSURE)
        self._frame_size = (actual_w, actual_h)

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def exposure(self):
        return self._exposure

    def set_exposure(self, x):
        self._camera.set(cv.CAP_PROP_EXPOSURE, x)
        self._exposure = x

    def read(self):
        ret, frame = self._camera.read()
        assert ret
        return frame

class RPiCamera(ICamera):

    def __init__(self):
        import libcamera
        import picamera2
        print("Initializing camera...")
        self._camera = picamera2.Picamera2()
        config = self._camera.create_video_configuration(
            main={'size':(4056, 3040), 'format':'RGB888'},
            # lores={'size':(640, 480), 'format':'YUYV'},
            transform=libcamera.Transform(hflip=1, vflip=1),
            display='main')

        # configuration that tries to leverage just the center of the
        # camera for less distortion, although I think this sensor
        # mode also decimates the data (1/2 the resolution is lost in
        # the active area of the sensor)
        config_alternate = self._camera.create_video_configuration(
            sensor={'output_size':(1332,900), 'bit_depth':10},
            main={'size':(1332, 900), 'format':'RGB888'},
            transform=libcamera.Transform(hflip=1, vflip=1))
        
        self._camera.align_configuration(config)
        print(f"{config=}")
        self._camera.configure(config)

        self._camera.start()
        time.sleep(2)

        actual_w, actual_h = config['main']['size']
        self._frame_size = (actual_w, actual_h)
        
        print(f"Capture size set to {actual_w}x{actual_h}")
        self._exposure = 0.

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def exposure(self):
        return self._exposure

    def set_exposure(self, x):
        self._exposure = x
        # oops! not implemented!

    def read(self):
        # see bricksort/camera/camera_focus.py for fancier request processing
        request = self._camera.capture_request()
        image = None
        try:
            image = request.make_array('main')
        finally:
            request.release()

        # image = cv.cvtColor(image, cv.COLOR_YUV2BGR_YUYV)
        return image

class WebCamera(ICamera):

    def __init__(self, host):
        self.host = host
        self.session = requests.Session()
        self.config = self._get_config()
        print(f"{self.host=} {self.config=}")

    @property
    def frame_size(self):
        return tuple(self.config['main']['size'])

    def read(self):
        r = self.session.get(
            self.host + '/image/main.jpeg', params={}, timeout=5)
        # magic to give imdecode data it can parse, this is just a
        # view, not a copy
        buf = np.frombuffer(r.content, dtype=np.uint8)
        # cv.imdecode derives the format from the bytestream and there
        # is no mechanism to force a format, :shrug:
        return cv.imdecode(buf, cv.IMREAD_COLOR)

    def _get_config(self):
        r = self.session.get(self.host + '/config', timeout=5)
        return json.loads(r.content)

