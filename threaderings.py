import sys
from collections import deque

print(sys.path)
from threading import Thread
import numpy as np
import time
import tensorflow as tf
import cv2
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
from flask_socketio import SocketIO, emit
app = Flask(__name__)
socketio = SocketIO(app)
executor = ThreadPoolExecutor(max_workers=4)
processed_frames = deque()
ITEMS = [
    # {"name": "item1", "x": 0, "y": 0, "z": 0, "status": False},
    # {"name": "item2", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item3", "x": 0, "y": 0, "z": 0, "status": False}
    # {"name": "item4", "x": 0, "y": 0, "z": 0, "status": False},
    # {"name": "item5", "x": 0, "y": 0, "z": 0, "status": False}
]
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
hip_history = deque(maxlen=10)  # Stores last 10 frames' hip keypoints
shoulder_history = deque(maxlen=10)  # Stores last 10 frames' shoulder keypoints

class VideoCamera(object):
    def __init__(self, rtsp_url):
        self.video = cv2.VideoCapture(rtsp_url)
        (self.grabbed, self.frame) = self.video.read()
        Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        return self.frame

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def add_text_to_frame(frame, text, font_scale=1.0, color=(255, 0, 0), thickness=2, bottom_margin=10, right_margin=10):
    (h, w) = frame.shape[:2]
    frame_with_text = frame.copy()
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = int(w - text_width - right_margin)
    text_y = int(h - bottom_margin)
    cv2.putText(frame_with_text, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return frame_with_text


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # head ankles  and ((p1 in range(0, 5) or p2 in range(0, 5)) or (p1 in range(15, 16) or p2 in range(15, 16))
            if ((p1 == 5 and p2 == 11) or (p1 == 6 and p2 == 12)):
                if (abs(y2 - y1) <= 10):
                    print("firstcheck")
                    return True


def process_frame(frame, prev_frame):
    found = False
    found2=False
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)
    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Rendering
    found = draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
    draw_keypoints(frame, keypoints_with_scores, 0.4)
    a = {
        0: keypoints_with_scores[0][0][0],  # nose
        1: keypoints_with_scores[0][0][1],  # leye
        2: keypoints_with_scores[0][0][2],  # reye
        3: keypoints_with_scores[0][0][3],  # ears
        4: keypoints_with_scores[0][0][4],
        5: keypoints_with_scores[0][0][5],  # lshoulder
        6: keypoints_with_scores[0][0][6],  # rshoulder
        7: keypoints_with_scores[0][0][11],  # hip
        8: keypoints_with_scores[0][0][12],  # rhip
        9: keypoints_with_scores[0][0][15],
        10: keypoints_with_scores[0][0][16],
        11: keypoints_with_scores[0][0][7],
        12: keypoints_with_scores[0][0][8],
        13: keypoints_with_scores[0][0][9],
        14: keypoints_with_scores[0][0][10],
        15: keypoints_with_scores[0][0][13],
        16: keypoints_with_scores[0][0][14]}
    found = motionchecker(frame, prev_frame, a)
    print("update")
    update_histories(a, hip_history, shoulder_history)
    found2 = detect_fall(hip_history, shoulder_history)
    if found:
        print("sending frames detected")
        frame = add_text_to_frame(frame, "FALL DETECTED", font_scale=1.2, color=(0, 255, 0))
        with app.app_context():
            socketio.emit('foundfall', ITEMS[0])
        found = False
        return frame
    elif found2:
        print("sending frames detected2")
        frame = add_text_to_frame(frame, "FALL DETECTED", font_scale=1.2, color=(0, 255, 0))
        with app.app_context():
            socketio.emit('foundfall', ITEMS[0])
        found2 = False
        return frame
    else:
        return frame


def motionchecker(frame, prev_frame, a):
    if prev_frame is None:
        return False
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0) ## 3 3 for faster?!
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_blurred = cv2.GaussianBlur(prev_frame_gray, (5, 5), 0)

    frame_diff = cv2.absdiff(prev_frame_blurred, frame_blurred)
    _, thresholded_diff = cv2.threshold(frame_diff, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_diff.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 600: #1000
           # print("motion")
            if ((0 in a and a[0][0] > 0.7 and a[0][2] > 0.1) or
                    (1 in a and a[1][0] > 0.7 and a[1][2] > 0.1) or
                    (2 in a and a[2][0] > 0.7 and a[2][2] > 0.1) or
                    (3 in a and a[3][0] > 0.7 and a[3][2] > 0.1) or
                    (4 in a and a[4][0] > 0.7 and a[4][2] > 0.1)):
                print("head")
                if ((a[9][0] > 0.7 and a[9][2] > 0.1)
                        or (a[10][0] > 0.7 and a[10][2] > 0.1)
                        or (a[8][0] > 0.7 and a[8][2] > 0.1) or (a[7][0] > 0.7 and a[7][2] > 0.1)
                        or (a[15][0] > 0.7 and a[15][2] > 0.1)
                        or (a[16][0] > 0.7 and a[16][2] > 0.1)):
                    print("fall")
                    return True


def detect_fall(hip_history, shoulder_history):
    if len(hip_history) < 10 or len(shoulder_history) < 10:
        return False

    #print("fucken analyze it")
    # Analyze the trend of hip movements
    hip_movements = [hips[0][1] + hips[1][1] for hips in hip_history]
    shoulder_movements = [shoulders[0][1] + shoulders[1][1] for shoulders in shoulder_history]

    # Calculate the average position in the last few frames
    avg_hip_position = sum(hip_movements[-5:]) / 5
    avg_shoulder_position = sum(shoulder_movements[-5:]) / 5
    print(f"hip:{avg_hip_position}")
    print(f"shoulder:{avg_shoulder_position}")
    # Check for significant downward movement of hips relative to shoulders
    hip_threshold = 5  # Threshold for hip movement to consider (adjust based on application needs)
    if (hip_movements[-1] - avg_hip_position > hip_threshold) and (
            shoulder_movements[-1] - avg_shoulder_position < hip_threshold / 2):
        print("i am inside the check finally")
        last_hips = hip_history[-1]
        last_shoulders = shoulder_history[-1]
        hip_width = abs(last_hips[0][0] - last_hips[1][0])
        shoulder_width = abs(last_shoulders[0][0] - last_shoulders[1][0])

        # Check if hips are notably wider than shoulders, indicating a horizontal orientation
        if hip_width > 1.5 * shoulder_width:
            return True
    return False

def update_histories(a, hip_history, shoulder_history):
    hip_left = (a[7][1], a[7][0])
    hip_right = (a[8][1], a[8][0])
    shoulder_left = (a[5][1], a[5][0])
    shoulder_right = (a[6][1], a[6][0])
    if a[7][2] > 0.1 and a[8][2] > 0.1:
        hip_history.append((hip_left, hip_right))
    if a[5][2] > 0.1 and a[6][2] > 0.1:
        shoulder_history.append((shoulder_left, shoulder_right))
    if len(hip_history) > 10:
        hip_history.popleft()
    if len(shoulder_history) > 10:
        shoulder_history.popleft()

def gen(camera):
    prev_frame = None
    fps_time = time.time()
    frame_count = 0
    while True:
        try:
            frame = camera.get_frame()
            start_time = time.time()
            if prev_frame is not None:
                future = executor.submit(process_frame, frame, prev_frame)
                processed_frames.append(future)
            if processed_frames:
                processed_future = processed_frames.popleft()
                processed_frame = processed_future.result()
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
                if not ret:
                    raise ValueError("Could not encode frame to JPEG")
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                frame_count += 1
                if time.time() - fps_time >= 1:  # Calculate FPS every second
                    #print(f"FPS: {frame_count}")
                    frame_count = 0
                    fps_time = time.time()

            prev_frame = frame
            process_time = time.time() - start_time
            time.sleep(max(0, 0.03 - process_time))

        except Exception as e:
            print(f"Error: {e}")
            continue


@app.route('/video_feed')
def video_feed():
    try:
        return Response(gen(VideoCamera(0)),  #'rtsp://admin:Test1999!@192.168.1.5:554/Streaming/Channels/101'
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        return str(e)


@app.route('/')
def index():
    return render_template('index.html', items=ITEMS)


if __name__ == '__main__':
    Thread(target=app.run, kwargs={'host': '0.0.0.0', 'debug': True, 'threaded': True}).start()
