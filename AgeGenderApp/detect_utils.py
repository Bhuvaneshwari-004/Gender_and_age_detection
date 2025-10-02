import cv2
import os
import math
from mtcnn import MTCNN
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ageProto = os.path.join(BASE_DIR, "models/age_deploy.prototxt")
ageModel = os.path.join(BASE_DIR, "models/age_net.caffemodel")
genderProto = os.path.join(BASE_DIR, "models/gender_deploy.prototxt")
genderModel = os.path.join(BASE_DIR, "models/gender_net.caffemodel")

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

detector = MTCNN()

def detect_age_gender_frame(frame):
    frameOpencvDnn = frame.copy()
    h, w = frame.shape[:2]
    padding = 20
    results = []

    faces = detector.detect_faces(frame)
    for face in faces:
        x, y, width, height = face['box']
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w - 1, x + width + padding)
        y2 = min(h - 1, y + height + padding)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        results.append(label)

        # Draw bounding box and label with background box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y_label = max(y1, label_size[1] + 10)
        cv2.rectangle(frameOpencvDnn, (x1, y_label - label_size[1] - 10),
                      (x1 + label_size[0], y_label + base_line - 10), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frameOpencvDnn, label, (x1, y_label - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return results, frameOpencvDnn

def detect_age_gender_image(image_path, output_path):
    frame = cv2.imread(image_path)
    results, processed = detect_age_gender_frame(frame)
    cv2.imwrite(output_path, processed)
    return results, output_path

def detect_age_gender_video(video_path, output_path, skip_frames=2):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    results = []
    out = None
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % skip_frames == 0:
            frame_results, processed_frame = detect_age_gender_frame(frame)
        else:
            processed_frame = frame.copy()
            frame_results = []
        results.append(frame_results)
        if out is None:
            h, w = processed_frame.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS) / skip_frames, (w, h))
        out.write(processed_frame)
        frame_id += 1
    cap.release()
    if out:
        out.release()
    return results, output_path
