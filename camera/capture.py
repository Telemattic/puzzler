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

cam = cv.VideoCapture(1, cv.CAP_MSMF)

w, h = 4656, 3496
set_size(cam, w, h)

print("Frame size:", get_size(cam))

print("Backend:", cam.getBackendName())

fourcc = int(cam.get(cv.CAP_PROP_FOURCC))
print(f"FourCC: {fourcc:08x}")

cv.namedWindow("overview")
cv.namedWindow("detail")

img_counter = 0

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

cam.release()

cv.destroyAllWindows()
