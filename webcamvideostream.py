import threaderings
from threaderings import Thread, Lock
import pandas as pd
import numpy as np
import time
import tensorflow as tf
import cv2
from flask import Flask, render_template, Response,jsonify, request,redirect, url_for
from flask_socketio import SocketIO,emit
app = Flask(__name__)
socketio = SocketIO(app)
ITEMS = [
   # {"name": "item1", "x": 0, "y": 0, "z": 0, "status": False},
   # {"name": "item2", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item3", "x": 0, "y": 0, "z": 0, "status": False}
    #{"name": "item4", "x": 0, "y": 0, "z": 0, "status": False},
    #{"name": "item5", "x": 0, "y": 0, "z": 0, "status": False}
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
def save_prediction(data, pose):
  # Extract relevant data from frame (e.g., pixel values)
  print(data[0])
  df = pd.DataFrame(np.array(data[0]), columns=['x', 'y', 'z'])
  df.to_csv('output.csv', index=False)

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
            #head ankles  and ((p1 in range(0, 5) or p2 in range(0, 5)) or (p1 in range(15, 16) or p2 in range(15, 16))
            if ((p1 == 5 and p2 == 11) or (p1 == 6 and p2 == 12)):
                if(abs(y2-y1)<=10):
                    print("firstcheck")
                    return True
def add_text_to_frame(frame, text, font_scale=1.0, color=(255, 0, 0), thickness=2, bottom_margin=10, right_margin=10):
    # Get frame height and width
    (h, w) = frame.shape[:2]

    # Create a copy of the frame to avoid modifying the original
    frame_with_text = frame.copy()

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Calculate text placement coordinates
    text_x = int(w - text_width - right_margin)
    text_y = int(h - bottom_margin)

    # Add text to the frame
    cv2.putText(frame_with_text, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness)

    return frame_with_text
class VideoStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera

        # opening video capture stream
        self.vcap = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        # reading a single frame from vcap stream for initializing
        self.grabbed, self.frame = self.vcap.read()
        if self.grabbed is False:
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream
        self.stopped = True

        # reference to the thread for reading next available frame from input stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True  # daemon threads keep running in the background while the program is executing

    def start(self):
        self.stopped = False
        self.t.start()

        # method for reading next frame

    def update(self):
        while True:
            if self.stopped is True:
                break
            self.grabbed, self.frame = self.vcap.read()
            if self.grabbed is False:
                print('[Exiting] No more frames to read')
                self.stopped = True
                break
        self.vcap.release()

        # method for returning latest read frame

    def read(self):
        return self.frame

        # method called to stop reading frames

    def stop(self):
        self.stopped = True


def handleimages(frame):
    # found = False
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    # Reshape image
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
    input_image = tf.cast(img, dtype=tf.float32)

    # Setup input and output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Make predictions
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
    return found
def framegray():
  #  frame_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
   # frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
   # if prev_frame is not None and prev_prev_frame is not None:
      #  frame_diff = cv2.absdiff(prev_prev_frame, frame_blurred)
      #  _, thresholded_diff = cv2.threshold(frame_diff, 60, 255, cv2.THRESH_BINARY)
      #  contours, _ = cv2.findContours(thresholded_diff.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

     #   for contour in contours:

        #    area = cv2.contourArea(contour)

        #    if area > 1000:
                print("motion")
           #     if ((0 in a and a[0][0] > 0.7 and a[0][2] > 0.1) or
            #            (1 in a and a[1][0] > 0.7 and a[1][2] > 0.1) or
            #            (2 in a and a[2][0] > 0.7 and a[2][2] > 0.1) or
             #           (3 in a and a[3][0] > 0.7 and a[3][2] > 0.1) or
             #           (4 in a and a[4][0] > 0.7 and a[4][2] > 0.1)):
              #      print("head")
               #     if ((a[9][0] > 0.7 and a[9][2] > 0.1)
                #            or (a[10][0] > 0.7 and a[10][2] > 0.1)
                #            or (a[8][0] > 0.7 and a[8][2] > 0.1) or (a[7][0] > 0.7 and a[7][2] > 0.1)
                #            or (a[15][0] > 0.7 and a[15][2] > 0.1)
                 #           or (a[16][0] > 0.7 and a[16][2] > 0.1)):
                 #       found = True
                  #      print("fall")

def updatefound(frame):
    frame_with_text = add_text_to_frame(frame, "FALL DETECTED", font_scale=1.2, color=(0, 255, 0))
    frame_bytes = cv2.imencode('.jpg', frame_with_text)[1].tobytes()
    print("sending frames detected")
    with app.app_context():
        socketio.emit('foundfall', {"name": "item3", "x": 0, "y": 0, "z": 0, "status": False})

def updateimage(jpeg):
    frame_bytes=jpeg.tobytes()
    print("sending frame bytes")
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
           )


# ret1, frame1 = cap1.read()
# Get current date and time
# date_time = str(datetime.datetime.now())
# now = datetime.datetime.now()
# date_time = now.strftime("%H:%M:%S")
# font = cv2.FONT_HERSHEY_SIMPLEX
def videofeed():
    print("videofeed")
    cap0 = VideoStream(stream_id=0)
    cap0.start()
    print("opening read")
    num_frames_processed=0
    start = time.time()
    while True:
        if cap0.stopped is True:
            break
        else:
            frame0 = cap0.read()

        delay = 0.03  # delay value in seconds. so, delay=1 is equivalent to 1 second
        time.sleep(delay)
        num_frames_processed += 1
        #  frame0 = cv2.resize(frame0, (640, 480))
        # frame1 = cv2.resize(frame1, (640, 480))
        # Hori = np.concatenate((frame0, frame1), axis=1)
        # Hori = cv2.putText(Hori, date_time, (10, 100), font, 1, (210, 155, 155), 4, cv2.LINE_4)
        # cv2.imshow('Cam 0&amp;1', Hori)
        handleimages(frame0)
        cv2.imshow("Detection Results", frame0)
        jpeg = cv2.imencode('.jpg', frame0)
        print("updateimage")
        updateimage(jpeg)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    end = time.time()
    cap0.stop()  # stop the webcam stream

    # printing time elapsed and fps
    elapsed = end - start
    fps = num_frames_processed / elapsed
    print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

    # closing all windows
    cv2.destroyAllWindows()
@app.route('/video_feed')
def video_feed():
    return Response(videofeed(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html', items = ITEMS)
if __name__ == '__main__':
    # Create a thread pool with the specified number of worker threads
  #  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit API call tasks to the pool
       # futures = [executor.submit(video_feed(), url) for url in urls]
#
        # Wait for all tasks to complete (optional)
       # for future in concurrent.futures.as_completed(futures):
          #  try:
           #     future.result()  # Process potential exceptions here
         #   except Exception as e:
            #    print(f"Error in API call: {e}")
   # for url in urls:
       # thread = threading.Thread(target=make_api_call, args=(url,))
        #threads.append(thread)
       # thread.start()

        # Wait for all threads to finish
   # for thread in threads:
      #  thread.join()
    socketio.run(app,debug=True,host="0.0.0.0.",port=5000,threaded=True)