import mediapipe as mp
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time as t


import numpy as np
from glob import glob
import json

from tqdm import tqdm

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.svm import SVC
import pickle
import pandas as pd
import os

import vgamepad as vg

# import pyautogui as pykb
import pydirectinput as pykb
# from pyautogui import press, typewrite, hotkey, keyUp, keyDown
pykb.FAILSAFE = False
pykb.PAUSE = 0.0001

# import pygetwindow as gw
# from pyautogui import screenshot

# import easyocr
# from time_ocr import process_race_time
UPLOAD_TIME_FRAMES = 0
RESTART_FRAMES = 0
TRIGGER_FRAMES_THRESHOLD = 10



def box_results(results, width, height):
  boxes = []
  if results.multi_hand_landmarks:
    for hand in results.multi_hand_landmarks:
      x = []
      y = []
      hand_lms = hand.landmark
      # print(len(hand_lms))
      for lm in hand_lms:
        x.append(lm.x * width)
        y.append(lm.y * height)
      box = {'xmin': min(x), 'ymin': min(y), 'xmax': max(x), 'ymax': max(y)}
      boxes.append(box)
  return boxes


def yolo_box_results(results, width, height):
    boxes = []
    df = results.pandas().xyxy[0]
    for idx, row in df.iterrows():
        box = {'xmin': row['xmin'], 'ymin': row['ymin'], 'xmax': row['xmax'], 'ymax': row['ymax']}
        boxes.append(box)
    return boxes

def get_direction(boxes, img_height):
    res = None
    if len(boxes) == 2:
        centers = []
        for box in boxes:
            centers.append(((box['xmin'] + box['xmax'])/2, (box['ymin'] + box['ymax'])/2))
        if centers[0][0] < centers[1][0]:
            left_hand = centers[0]
            right_hand = centers[1]
        else:
            left_hand = centers[1]
            right_hand = centers[0]

        # Turn right
        if left_hand[1] > right_hand[1] + img_height/6:
            res = "right"
        elif right_hand[1] > left_hand[1] + img_height/6:
            res = "left"
        
        if sum(centers[i][1] for i in range(2))/2 > img_height*2/3:
            res = "down_" + res if res else "down"
        else:
            res = "up_" + res if res else "up"
    if res is None:
        res = 'none'
    return res

def get_direction_signs(boxes, hand_signs, img_height, img_width):
    res = None
    if len(boxes) == 2:
        centers = []
        for box in boxes:
            centers.append(((box['xmin'] + box['xmax'])/2, (box['ymin'] + box['ymax'])/2))
        if centers[0][0] < centers[1][0]:
            left_hand = centers[0]
            right_hand = centers[1]
        else:
            left_hand = centers[1]
            right_hand = centers[0]

        # Turn
        slope = (right_hand[1] - left_hand[1]) / (right_hand[0] - left_hand[0])*1.5
        if slope > 0.2:
            res = slope - 0.19
            res = str(min(1, res))
        elif slope < -0.2:
            res = slope + 0.19
            res = str(max(-1, res))
        else:
            res = None

        
        if sum(hand_signs) == 2:
            res = "up_" + res if res else "up"
        elif 0 in hand_signs:
            res = "down_" + res if res else "down"
        # print(res)
    if res is None:
        res = 'none'
    return res
    
def restart_run():
    pykb.keyDown('space')
    pykb.keyUp('space')

def get_buttons(boxes, features, img_height, img_width):
    global UPLOAD_TIME_FRAMES
    global RESTART_FRAMES
    hand_signs = [int(rps_model.predict([feats])[0]) for feats in features]

    # Check if the user wants to upload is time
    if hand_signs == [2, 2]:
        if UPLOAD_TIME_FRAMES < TRIGGER_FRAMES_THRESHOLD:
            UPLOAD_TIME_FRAMES += 1
        else: 
            #process_race_time()
            UPLOAD_TIME_FRAMES = 0
    else:
        UPLOAD_TIME_FRAMES = 0
    
    # Check if the user wants to restart the current run
    if hand_signs == [2]:
        if RESTART_FRAMES < TRIGGER_FRAMES_THRESHOLD:
            RESTART_FRAMES += 1
        else: 
            restart_run()
            RESTART_FRAMES = 0
    else:
        RESTART_FRAMES = 0

    direction = get_direction_signs(boxes, hand_signs, img_height, img_width)
    return direction




def get_features(results):

    try:
        landmarks = results.multi_hand_landmarks[0].landmark
    except TypeError:  # If no hand detected
        return []
    hands = []
    for hand in results.multi_hand_landmarks:
        landmarks = hand.landmark
        x = []
        y = []
        for lm in landmarks:
            x.append(lm.x)
            y.append(lm.y)
        delta_x, delta_y = (max(x) - min(x)) / 2, (max(y) - min(y)) / 2  # Half width of the box
        x_0, y_0 = (max(x) + min(x)) / 2, (max(y) + min(y)) / 2  # Center of the box
        # list normalized coordinates (from -1 to +1)
        res = []
        for i in range(len(x)):
            res.append((x[i] - x_0)/delta_x)
            res.append((y[i] - y_0)/delta_y)
        hands.append(res)
    return hands

def plot_bboxes(image, bboxes):
    for bbox_dict in bboxes:
      # print(bbox_dict)
      xmin, ymin = int(bbox_dict['xmin']), int(bbox_dict['ymin'])
      xmax, ymax = int(bbox_dict['xmax']), int(bbox_dict['ymax'])
      image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return image

def press_direction(direction):
    # print(direction)
    if direction is not None:
        if '_' in direction:
            directions = direction.split('_')
            #print(directions)
            for direc in directions:
                custom_press(direc)
            # hotkey(directions[0], directions[1])
        else:
            custom_press(direction)
        # if direction == "left":
        #     custom_press('left')
        #     #print('left')
        # elif direction == "right":
        #     custom_press('right')
        #     #print('right')
    if direction is None:
        pass
        # custom_press('none')

def press_direction_2(direction, current_direction):
    # print(direction)
    if direction is None:
        pass
        return 'none'
    else:
        direction = custom_press_pad(direction, current_direction)
        # custom_press('none')
        return direction
    
def custom_press(direction, current_direction):
    if 'left' in direction:
        pykb.keyUp('right')
        pykb.keyDown('left')
    elif 'right' in direction:
        pykb.keyUp('left')
        pykb.keyDown('right')
    else:
        pykb.keyUp('left')
        pykb.keyUp('right')
    if 'up' in direction:
        pykb.keyUp('down')
        pykb.keyDown('up')
    elif 'down' in direction:
        pykb.keyUp('up')
        pykb.keyDown('down')
    if 'none' in direction:
        pykb.keyUp('up')
        pykb.keyUp('down')
        pykb.keyUp('left')
        pykb.keyUp('Right')

opposites = {
    'none': None,
    'left': 'right',
    'right': 'left',
    'up': 'down',
    'down':'up',
}

def custom_press_2(direction, current_direction):
    if current_direction != direction:
        new_dirs = direction.split('_')
        # print('curr_d:', current_direction)
        old_dirs = current_direction.split('_')
        for dir in old_dirs:
            if dir not in new_dirs:
                pykb.keyUp(dir)
        for dir in new_dirs:
            if dir not in old_dirs:
                pykb.keyDown(dir)
    return direction

def move_stick(dir):
    if dir != 'none':
        dir = float(dir)
        gamepad.left_joystick_float(x_value_float=-dir, y_value_float=0.0)
        gamepad.update()
    else:
        gamepad.left_joystick_float(x_value_float=0, y_value_float=0.0)
        gamepad.update()        


def release_btn(dir):
    if dir == 'up':
        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
    elif dir == 'down':
        gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        gamepad.update()

def press_btn(dir):
    if dir == 'up':
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        gamepad.update()
    elif dir == 'down':
        gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        gamepad.update()

def custom_press_pad(direction, current_direction):
    if current_direction != direction:
        new_dirs = direction.split('_')
        old_dirs = current_direction.split('_')
        for dir in old_dirs:
            if dir not in new_dirs:
                if dir in ['up', 'down']:
                    release_btn(dir)
        for dir in new_dirs:
            if dir not in old_dirs:
                if dir in ['up', 'down']:
                    press_btn(dir)
                else:
                    move_stick(dir)
        # for dir in new_dirs:
        #     if dir in opposites.keys():
        #         release_btn(opposites[dir])
        #         press_btn(dir)
        #     else:
        #         move_stick(dir)
        # if 'up' not in dir and 'down' not in dir:
        #     release_btn('up')
        #     release_btn('down')
    return direction

    


# import torch
gamepad = vg.VX360Gamepad()
rps_model = pickle.load(open('rps_model.pkl', 'rb'))
def main():
    mp_hands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands = 2)
    connections = mp.solutions.hands.HAND_CONNECTIONS
    mp_drawing = mp.solutions.drawing_utils
    
    # model_path = "runs/train/exp4/weights/best.pt"
    # model_path = "best_pruned.pt"
    # model = torch.hub.load("ultralytics/yolov5", 'custom', path=model_path, force_reload=True)
    # STREAMING MODEL TEST
    cap = cv2.VideoCapture(0)  # Put 1 for external webcam
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()


    # Video recording
    # ret, frame = cap.read()
    # cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, 0)
    # cv2.imshow('frame', frame)
    # window_game_title = "Trackmania"
    # window_game = gw.getWindowsWithTitle(window_game_title)[0]
    # left_game, top_game, width_game, height_game = window_game.left, window_game.top, window_game.width, window_game.height
    # window_cam_title = "frame"
    # window_cam = gw.getWindowsWithTitle(window_cam_title)[0]
    # left_cam, top_cam, width_cam, height_cam = window_cam.left, window_cam.top, window_cam.width, window_cam.height
    # fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # out_game = cv2.VideoWriter("recorded_game.avi", fourcc, 20.0, (width_game, height_game))
    # out_cam = cv2.VideoWriter("recorded_cam.avi", fourcc, 20.0, (width_cam, height_cam))

    t_0 = t.time()
    print("START")
    # while t.time() - t_0 < 300:
    t_0 = t.time()
    # for i in range(200):
    current_direction = 'none'
    # while t.time() - t_0 < 300:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # frame.flags.writeable = False
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        # Mediapipe
        results = mp_hands.process(img)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, landmarks, connections)
        boxes = box_results(results, width, height)
        features = get_features(results)
        direction = get_buttons(boxes, features, height, width)
            # print("inputs:", direction)
        # YOLOv5n
        # results = model(img)  # YOLOv5n
        # boxes = yolo_box_results(results, width, height)

        # res_img = cv2.flip(plot_bboxes(img, boxes), 1)
        res_img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA), 1)

        # direction = get_direction(boxes, img_height=height)
        current_direction = press_direction_2(direction, current_direction)
        # cv2.imshow('frame', img)
        # cv2.namedWindow('frame')        # Create a named window
        # cv2.moveWindow('frame', 50, 50)
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, 0)
        cv2.imshow('frame', res_img)
        if cv2.waitKey(1) == ord('q'):
            break
        
        # Record frame in a video file
        # game_frame = screenshot(region=(left_game, top_game, width_game, height_game))
        # video_frame = cv2.cvtColor(np.array(game_frame), cv2.COLOR_RGB2BGR)
        # out_game.write(video_frame)
        # out_cam.write(res_img)

    t_f = t.time()
    print("Total time:", t_f-t_0,"seconds")
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("FINISHED!")


if __name__ == "__main__":
    main()