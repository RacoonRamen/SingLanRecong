import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tensorflow import keras
from utils import *
# install pre-train transformer
from happytransformer import HappyTextToText, TTSettings
happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
args = TTSettings(num_beams=5, min_length=1)
# loading model
model = keras.model.load_model('model')
actions = os.listdir('MP_Data')

colors = [(100,117,162)]
sequence ,sentence = [],[]
good_sentence = []
threshold = 0.9
target_sentence = []
result = []
Target = []

text_img = np.zeros((256,1024,3),)
text_img.fill(200)

prob = np.zeros((800,500,3),)
prob.fill(0)

cap = cv2.VideoCapture(1)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        target = []
        flag = 0
        #text_img = np.zeros((256,1024,3),)
        text_img.fill(200)
        
        #prob = np.zeros((800,500,3))
        prob.fill(0)
        # Read feed
        res, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        #print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extract_keypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(res[np.argmax(res)])
            clear_output(3)
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]: 
                        sentence.append(actions[np.argmax(res)])
                        result ,Target = grammar_correction(sentence)
                        #print(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])  

            prob = prob_viz(res, actions, prob)
        # Show to screen
        #Text showing

        cv2.putText(text_img, str(sentence), (3,30) ,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)     
        cv2.putText(text_img, "Transformer said: " +str(result), (3,150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(text_img, "Original sentence: "+''.join(Target), (3,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2, cv2.LINE_AA)
                             
        cv2.imshow('Text', text_img)                     
        cv2.imshow('OpenCV Feed', image)
        cv2.imshow('Probability',prob)
        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()