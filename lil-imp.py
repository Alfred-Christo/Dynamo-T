import cv2
import math
import numpy as np
from ultralytics import YOLO
import cvzone
import json

cap = cv2.VideoCapture("ff.mp4")
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) 

model = YOLO('yolov8n.pt')

# Define two equal-length lines with a small gap in the middle
line_length = 600
gap = 100
shift = 70
line1 = [100 - shift, 500, 100 + line_length - shift, 500]  # First horizontal line
line2 = [100 + line_length + gap - shift, 500, 100 + 2 * line_length + gap - shift, 500]  # Second horizontal line

# Calculate the midpoint for the vertical line
mid_x = (line1[2] + line2[0]) // 2
vertical_line = [mid_x, 0, mid_x, 720]  # Vertical line coordinates, starting from the bottom

counter_left = []
counter_right = []
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def point_line_distance(point, line):
    x1, y1, x2, y2 = line
    x0, y0 = point
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5

while True:
    ret, frame = cap.read()
    cv2.resizeWindow("Frame", 1280, 720)
    if not ret:
        cap = cv2.VideoCapture("usecasecam/Edge_Crossing_2_Cam1_1.avi")
        continue

    results = model(frame, stream=True)
    detections = []

    for info in results:
        boxes = info.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100)
            classindex = int(box.cls[0])
            objectdetect = classnames[classindex]

            # Detect vehicles instead of persons
            if objectdetect in ['car', 'truck', 'bus', 'motorbike'] and conf > 50:
                detections.append((x1, y1, x2, y2, conf, classindex))

    # Draw the horizontal lines
    cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 255, 255), 7)
    cv2.line(frame, (line2[0], line2[1]), (line2[2], line2[3]), (0, 255, 255), 7)

    # Draw the vertical line
    cv2.line(frame, (vertical_line[0], vertical_line[1]), (vertical_line[2], vertical_line[3]), (255, 0, 150), 20)

    for detection in detections:
        x1, y1, x2, y2, conf, classindex = detection
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Add label to the bounding box
        label = f"{classnames[classindex]} {conf}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Calculate the distance from the bottom of the bounding box to the lines
        distance1 = point_line_distance((cx, y2), line1)
        distance2 = point_line_distance((cx, y2), line2)

        # If the distance is less than a certain threshold, consider the line crossed
        if distance1 < 10 or distance2 < 10:  # Adjust this threshold as needed
            if distance1 < 10:
                cv2.line(frame, (line1[0], line1[1]), (line1[2], line1[3]), (0, 0, 255), 15)
            if distance2 < 10:
                cv2.line(frame, (line2[0], line2[1]), (line2[2], line2[3]), (0, 0, 255), 15)
            if (cx, cy) not in counter_left and cx < mid_x:
                counter_left.append((cx, cy))
                path = f"/c:/Users/user/Desktop/lil-proj/left_{len(counter_left)}_crosses.jpg"
                cv2.imwrite(path, frame)
            elif (cx, cy) not in counter_right and cx >= mid_x:
                counter_right.append((cx, cy))
                path = f"/c:/Users/user/Desktop/lil-proj/right_{len(counter_right)}_crosses.jpg"
                cv2.imwrite(path, frame)

    # Display the count of vehicles crossed in both regions
    cvzone.putTextRect(frame, f'Left crossed = {len(counter_left)}', [50, 34], thickness=4, scale=2.3, border=2)
    cvzone.putTextRect(frame, f'Right crossed = {len(counter_right)}', [600, 34], thickness=4, scale=2.3, border=2)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

    # Save the count of vehicles to a JSON file
    data = {"left_region_vehicles_crossed": len(counter_left), "right_region_vehicles_crossed": len(counter_right)}
    with open("output.json", "w") as json_file:
        json.dump(data, json_file)

    # Save the count of vehicles to a text file
    with open("output.txt", "w") as txt_file:
        txt_file.write(f"Left region vehicles crossed: {len(counter_left)}\nRight region vehicles crossed: {len(counter_right)}")
        