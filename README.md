# Virtual-SketchPad
This project is a virtual sketchpad that uses hand tracking via webcam to let users draw, select colors, and create shapes on a digital canvas without touching the screen. It features gesture-based controls for drawing, clearing, and shape selection, making it an interactive and touchless whiteboard experience

HOW VIRTUAL-SKETCHPAD WORK'S

Virtual sketchpad displays two windows: one with the live webcam feed and UI overlays, and Another showing the virtual sketchpad where your drawings appear in real time.

Four colors available: Blue, Green, Red, Yellow.
Select a color by pointing your finger at the corresponding color button at the top of the screen.

Eight shapes available: Circle, Square, Rectangle, Triangle, Rhombus, Trapezoid, Pentagon, Hexagon.
Select a shape by pointing at its button on the left side; the shape is drawn at the center of the canvas in the selected color.

You can draw with multiple colors by switching between them using gesture-based controls; 
each color’s strokes are managed separately for a smooth, interactive experience.

Point at the "EXIT" button to close the application.
Point at the "CLEAR" button to erase all drawings and reset the canvas.

REQUIREMENTS FOR THIS VIRTUAL-SKETCHPAD

Python 3.7 OR higher
OpenCV (cv2)
NumPy (numpy)
MediaPipe (mediapipe)
Webcam (for hand tracking and drawing)
A system with GUI support (to display OpenCV windows)

You can install the required Python packages with:
pip install opencv-python numpy mediapipe
