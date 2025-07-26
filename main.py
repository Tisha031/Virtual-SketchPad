import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Giving different arrays to handle colour points of different colour
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These indexes will be used to mark the points in particular arrays of specific colour
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# List of shapes
shapes = ["CIRCLE", "SQUARE", "RECTANGLE", "TRIANGLE", "RHOMBUS", "TRAPEZOID", "PENTAGON", "HEXAGON"]
shapeIndex = -1

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# Adding shapes buttons
for i, shape in enumerate(shapes):
    paintWindow = cv2.rectangle(paintWindow, (40, 70 + i * 50), (140, 120 + i * 50), (0, 0, 0), 2)
    cv2.putText(paintWindow, shape, (49, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# Adding exit button
paintWindow = cv2.rectangle(paintWindow, (40, 520), (140, 570), (0, 0, 0), 2)
cv2.putText(paintWindow, "EXIT", (49, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Drawing the rectangles and text for the options on the frame
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
    frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
    frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
    frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)

    # Adding shapes buttons to the frame
    for i, shape in enumerate(shapes):
        frame = cv2.rectangle(frame, (40, 70 + i * 50), (140, 120 + i * 50), (0, 0, 0), 2)
        cv2.putText(frame, shape, (49, 100 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Adding exit button to the frame
    frame = cv2.rectangle(frame, (40, 520), (140, 570), (0, 0, 0), 2)
    cv2.putText(frame, "EXIT", (49, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)
        print(center[1] - thumb[1])
        if (thumb[1] - center[1] < 30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:, :, :] = 255
            elif 160 <= center[0] <= 255:
                colorIndex = 0  # Blue
            elif 275 <= center[0] <= 370:
                colorIndex = 1  # Green
            elif 390 <= center[0] <= 485:
                colorIndex = 2  # Red
            elif 505 <= center[0] <= 600:
                colorIndex = 3  # Yellow
        elif 70 <= center[1] <= 120 + len(shapes) * 50:
            for i, shape in enumerate(shapes):
                if 40 <= center[0] <= 140 and 70 + i * 50 <= center[1] <= 120 + i * 50:
                    shapeIndex = i
                    break
        elif 520 <= center[1] <= 570 and 40 <= center[0] <= 140:  # Exit Button
            break
        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    if shapeIndex != -1:
        center_x, center_y = 320, 240  # Example center point
        if shapes[shapeIndex] == "CIRCLE":
            cv2.circle(frame, (center_x, center_y), 50, colors[colorIndex], 2)
            cv2.circle(paintWindow, (center_x, center_y), 50, colors[colorIndex], 2)
        elif shapes[shapeIndex] == "SQUARE":
            cv2.rectangle(frame, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50), colors[colorIndex], 2)
            cv2.rectangle(paintWindow, (center_x - 50, center_y - 50), (center_x + 50, center_y + 50),
                          colors[colorIndex], 2)
        elif shapes[shapeIndex] == "RECTANGLE":
            cv2.rectangle(frame, (center_x - 80, center_y - 50), (center_x + 80, center_y + 50), colors[colorIndex], 2)
            cv2.rectangle(paintWindow, (center_x - 80, center_y - 50), (center_x + 80, center_y + 50),
                          colors[colorIndex], 2)
        elif shapes[shapeIndex] == "TRIANGLE":
            pts = np.array([[center_x, center_y - 50], [center_x - 50, center_y + 50], [center_x + 50, center_y + 50]],
                           np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, colors[colorIndex], 2)
            cv2.polylines(paintWindow, [pts], True, colors[colorIndex], 2)
        elif shapes[shapeIndex] == "RHOMBUS":
            pts = np.array([[center_x, center_y - 50], [center_x - 50, center_y], [center_x, center_y + 50],
                            [center_x + 50, center_y]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, colors[colorIndex], 2)
            cv2.polylines(paintWindow, [pts], True, colors[colorIndex], 2)
        elif shapes[shapeIndex] == "TRAPEZOID":
            pts = np.array(
                [[center_x - 50, center_y - 50], [center_x + 50, center_y - 50], [center_x + 80, center_y + 50],
                 [center_x - 80, center_y + 50]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, colors[colorIndex], 2)
            cv2.polylines(paintWindow, [pts], True, colors[colorIndex], 2)
        elif shapes[shapeIndex] == "PENTAGON":
            pts = np.array([[center_x, center_y - 50], [center_x - 48, center_y - 15], [center_x - 29, center_y + 40],
                            [center_x + 29, center_y + 40], [center_x + 48, center_y - 15]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, colors[colorIndex], 2)
            cv2.polylines(paintWindow, [pts], True, colors[colorIndex], 2)
        elif shapes[shapeIndex] == "HEXAGON":
            pts = np.array([[center_x, center_y - 50], [center_x - 43, center_y - 25], [center_x - 43, center_y + 25],
                            [center_x, center_y + 50], [center_x + 43, center_y + 25], [center_x + 43, center_y - 25]],
                           np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, colors[colorIndex], 2)
            cv2.polylines(paintWindow, [pts], True, colors[colorIndex], 2)
        shapeIndex = -1

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
