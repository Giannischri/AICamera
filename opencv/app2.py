import asyncio
import pandas as pd
import numpy as np
import cv2
import concurrent
import json
import uuid
import tensorflow as tf
import time
import os
from aiortc import RTCPeerConnection, RTCSessionDescription

import threading
import base64
from flask import Flask, render_template, Response,jsonify, request,redirect, url_for
from flask_socketio import SocketIO,emit
app = Flask(__name__)
socketio = SocketIO(app)
ITEMS = [
    {"name": "item1", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item2", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item3", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item4", "x": 0, "y": 0, "z": 0, "status": False},
    {"name": "item5", "x": 0, "y": 0, "z": 0, "status": False}
]

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


   # return shaped[p1],shaped[p2]

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
def add_text_to_frame(frame, text, font_scale=1.0, color=(255, 0, 0), thickness=2, bottom_margin=10, right_margin=10):
  """


  Args:
      frame (numpy.ndarray): The image frame as a NumPy array.
      text (str): The text to be added to the frame.
      font_scale (float, optional): The scale factor for the font size. Defaults to 1.0.
      color (tuple, optional): The RGB color of the text. Defaults to red (255, 0, 0).
      thickness (int, optional): The thickness of the text outline in pixels. Defaults to 2.
      bottom_margin (int, optional): The margin from the bottom of the frame for text placement. Defaults to 10.
      right_margin (int, optional): The margin from the right of the frame for text placement. Defaults to 10.

  Returns:
      numpy.ndarray: The modified image frame with text overlay.
  """

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
def emit_frames():


    found=False
    interpreter = tf.lite.Interpreter(model_path='3.tflite')
    interpreter.allocate_tensors()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    height = int(cap.get(4))
    width = int(cap.get(3))
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_avg_pos = 0
    prev_dete_frame = 0
    last_fall_dete = 0
    threshold = 0.5
    fall_threshold = 0.04 * height
    framerate_threshold = round(fps / 5.0)
    fall_detected_text = (20, round(0.15 * height))
    frame_number = 0
    points = []
    prev_prev_frame = None
    prev_frame=None
    with app.app_context():
        socketio.emit('found', {"name":ITEMS[0]["name"],"x":ITEMS[0]["x"],"y":ITEMS[0]["y"],"z":ITEMS[0]["z"],"status":ITEMS[0]["status"]})
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
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
        found=draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)
        a = {
            0: keypoints_with_scores[0][0][0], #nose
            1: keypoints_with_scores[0][0][1],  #leye
            2: keypoints_with_scores[0][0][2], #reye
            3: keypoints_with_scores[0][0][3],  #ears
            4: keypoints_with_scores[0][0][4],
            5: keypoints_with_scores[0][0][5], #lshoulder
            6: keypoints_with_scores[0][0][6], #rshoulder
            7: keypoints_with_scores[0][0][11], #hip
            8: keypoints_with_scores[0][0][12], # rhip
            9: keypoints_with_scores[0][0][15],
            10: keypoints_with_scores[0][0][16],
            11: keypoints_with_scores[0][0][7],
            12:keypoints_with_scores[0][0][8],
            13:keypoints_with_scores[0][0][9],
            14:keypoints_with_scores[0][0][10],
            15: keypoints_with_scores[0][0][13],
            16: keypoints_with_scores[0][0][14]}

        cv2.imshow("Detection Results", frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        if prev_frame is not None and prev_prev_frame is not None:
            frame_diff = cv2.absdiff(prev_prev_frame, frame_blurred)
            _ , thresholded_diff = cv2.threshold(frame_diff, 60, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_diff.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            for contour in contours:

                area = cv2.contourArea(contour)

                if area > 1000:
                    print("motion")
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
                            found=True
                            print("fall")

        if found:
            frame_with_text = add_text_to_frame(frame, "FALL DETECTED", font_scale=1.2, color=(0, 255, 0))
            frame_bytes = cv2.imencode('.jpg', frame_with_text)[1].tobytes()
            print("sending frames detected")
        #  with app.app_context():
            #   socketio.emit('found', ITEMS[0])
            time.sleep(10)
            found=False
        else:
            frame_bytes=jpeg.tobytes()
        prev_prev_frame=prev_frame
        prev_frame = frame_blurred.copy()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n'
               )


    # if cv2.waitKey(10) & 0xFF == ord('q'):
    # break


# cap.release()
    #cv2.destroyAllWindows()
# Within your prediction loop:
#if predicted_class == your_false_class:  # Replace with your "false" class
 # timestamp = time.time()
  #save_false_prediction(frame, timestamp)
@socketio.on('message_from_server')
def handle_message_from_server(data):
    print('Received data from client:', data)

    # Example: Send a response back to the client
    response_data = "Hello from the server!"
    socketio.emit('message_from_server', response_data)
#@socketio.on('found')
#def handle_data():
   # data = "Data from Flask backend"
   # print("socket path")
    #socketio.emit('found', ITEMS[0])
def trigger_backend_event():
    print('trigger')
    socketio.emit('found', ITEMS[0])
@app.route('/')
def index():
    return render_template('index.html', items = ITEMS)
@app.route('/video_feed')
def video_feed():
    return Response(emit_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@socketio.on('save')
def handle_message_from_server2(cameraid):
    print('Received data from client:', cameraid)

   # save_prediction(nonfall, 'nonfall')
    # Example: Send a response back to the client

async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]

    # Create a data channel named "chat"
    # pc.createDataChannel("chat")

    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    # Prepare the response data with local SDP and type
    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)


# Wrapper function for running the asynchronous offer function
def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()


# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

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
    ENVIRONMENT_DEBUG = os.environ.get("APP_DEBUG", True)
    ENVIRONMENT_PORT = os.environ.get("APP_PORT", 7651)
    app.run(host='0.0.0.0', port=7651, debug=True)
