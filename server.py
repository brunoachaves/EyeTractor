from flask import Flask, render_template, Response, request, redirect
from utils import VideoCamera

import cv2 as cv
import time

is_rasp = True
use_cloud = True
show_frame = True
app = Flask(__name__)
video_camera = VideoCamera(is_rasp=is_rasp)


def driver_mode(camera):
    while True:
        img, btn_calib, _ = camera.run(is_rasp=is_rasp, use_cloud=use_cloud)
        if show_frame:
            cv.imshow('eye tractor', img)
            key = cv.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("c"):
                cv.destroyAllWindows()
                return 'c'
            elif key == ord('q'):
                cv.destroyAllWindows()
                return 'q'
        if is_rasp and (key == ord("c") or btn_calib):
            cv.destroyAllWindows()
            return 'c'


def gen(camera):
    while True:
        # get camera frame
        frame, _, btn_drive = camera.run(is_rasp=is_rasp, use_cloud=use_cloud)

        # encode OpenCV raw frame to jpg and displaying it
        _, jpeg = cv.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        if is_rasp and btn_drive:
            return redirect('/shutdown')


@app.route('/shutdown')
def shutdown_server():
    print("[INFO] encerrando servidor flask ...")
    shutdown = request.environ.get('werkzeug.server.shutdown')
    if shutdown is None:
        raise RuntimeError('Função indisponível!')
    else:
        shutdown()
    return "Servidor encerrado!"


@app.route('/')
def index():
    print('[INFO] rendering webpage')
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    print('[INFO] video feed')
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    while True:
        time.sleep(2)
        print('[INFO] modo direção')
        cmd = driver_mode(video_camera)
        if cmd == 'q':
            break
        # app.run(host='0.0.0.0', port='5000', debug=True)
        print('[INFO] calibration mode')
        app.run(threaded=True, debug=False, use_reloader=False, host="0.0.0.0", port=5000)
