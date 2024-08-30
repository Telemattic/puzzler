import cv2 as cv
import re

def set_size(cam, w, h):
    cam.set(cv.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, h)

def get_size(cam):
    w = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
    return (w,h)

print("Opening camera...")

cam = cv.VideoCapture(2, cv.CAP_MSMF)
# cam = cv.VideoCapture(3, cv.CAP_DSHOW)

for k in dir(cv):
    if not k.startswith('CAP_PROP'):
        continue
    if k.startswith('CAP_PROP_AUDIO_'):
        continue
    if k.startswith('CAP_PROP_DC1394_'):
        continue
    if k.startswith('CAP_PROP_GPHOTO2_'):
        continue
    if k.startswith('CAP_PROP_GIGA_'):
        continue
    if k.startswith('CAP_PROP_INTELPERC_'):
        continue
    if k.startswith('CAP_PROP_IOS_'):
        continue
    if k.startswith('CAP_PROP_OBSENSOR'):
        continue
    if k.startswith('CAP_PROP_OPENNI'):
        continue
    if k.startswith('CAP_PROP_PVAPI_'):
        continue
    if k.startswith('CAP_PROP_XI_'):
        continue
    
    v = cam.get(getattr(cv,k))
    print(f"{k}: {v}")

print("Frame size:", get_size(cam))

w, h = 4656, 3496
set_size(cam, w, h)

w, h = get_size(cam)
print("Frame size:", (w, h))

print("Backend:", cam.getBackendName())

fourcc = int(cam.get(cv.CAP_PROP_FOURCC))
print(f"FourCC: {fourcc:08x}")

cv.namedWindow("overview")
cv.namedWindow("detail")

img_counter = 0
exposure = cam.get(cv.CAP_PROP_EXPOSURE)

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    
    cv.imshow("overview", cv.resize(frame, (w//8, h//8)))
    cv.imshow("detail", frame[7*h//16:9*h//16, 7*w//16:9*w//16,:])

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
    elif k % 256 == ord('<'):
        exposure -= 1
        cam.set(cv.CAP_PROP_EXPOSURE, exposure)
        print(f"{exposure=}")
    elif k % 256 == ord('>'):
        exposure += 1
        cam.set(cv.CAP_PROP_EXPOSURE, exposure)
        print(f"{exposure=}")

cam.release()

cv.destroyAllWindows()
