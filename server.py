from flask import Flask, render_template, Response, request
from utils import VideoCamera

import cv2 as cv

app = Flask(__name__)
video_camera = VideoCamera()
is_rasp = False


def driver_mode(camera):
    while True:
        img = camera.run()
        if not is_rasp:
            cv.imshow('frame', img)
            key = cv.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("c"):
                cv.destroyAllWindows()
                return 'c'
            elif key == ord('q'):
                cv.destroyAllWindows()
                return 'q'
        else:
            pass  # rasp commands to choice the run mode


def gen(camera):
    while True:
        # get camera frame
        frame = camera.run()
        # encode OpenCV raw frame to jpg and displaying it
        _, jpeg = cv.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


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
        print('[INFO] modo direção')
        cmd = driver_mode(video_camera)
        if cmd == 'q':
            break
        # app.run(host='0.0.0.0', port='5000', debug=True)
        print('[INFO] calibration mode')
        app.run(threaded=True, debug=False, use_reloader=False, host="0.0.0.0", port=5000)
