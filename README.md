##Driver Drowsiness Detection System##
A real-time Python application that monitors a driver's eyes through a webcam, computes the Eye Aspect Ratio (EAR) from facial landmarks, and triggers an audio alarm when drowsiness is detected — helping prevent road accidents caused by fatigue.
Python 3.8+
Computer Vision
Real-time
Road Safety

What it is
Drowsiness is responsible for thousands of road accidents every year. This system acts as an always-on co-pilot — watching the driver's eyes through a standard webcam and sounding an alarm the moment it detects signs of fatigue, before a serious incident can occur.
Unlike simple timer-based reminders, this system is reactive and personalised: it only alerts when the driver's eyes are actually closing, making it far more accurate and less intrusive than fixed-interval systems.

How it works — 6-step pipeline
Every video frame captured from the webcam goes through this detection pipeline in real time:
01
Capture frame
OpenCV reads the webcam stream at approximately 30 frames per second.
02
Detect face
dlib's HOG-based frontal face detector locates the driver's face in each frame.
03
Get landmarks
A 68-point shape predictor maps facial keypoints — 6 points per eye are extracted.
04
Compute EAR
The Eye Aspect Ratio is calculated from the 6 landmark points for both eyes, then averaged.
05
Threshold check
If EAR drops below 0.25 for 20 or more consecutive frames, drowsiness is confirmed.
06
Sound alarm
pygame plays alarm.wav on a loop until the driver's eyes reopen and EAR recovers.

Eye Aspect Ratio (EAR) formula
The EAR is the mathematical heart of the system. It is a simple geometric ratio that measures how vertically open the eye is relative to its width:
EAR = ( ||p2–p6|| + ||p3–p5|| ) / ( 2 × ||p1–p4|| )
p1–p4 → horizontal eye width (denominator)
p2–p6, p3–p5 → vertical eye opening (numerator)
As the eye closes, the vertical distances shrink → EAR approaches 0
Wide open
~0.35
Blinking
~0.20
Closed
~0.0
0.0 (closed)
0.25 ← alert threshold
0.35 (open)
The threshold of 0.25 is tunable. Lower it in poor lighting, raise it for drivers with naturally smaller eyes.

Tech stack
opencv-python
Webcam capture, frame rendering, drawing eye contours and HUD overlay on screen
dlib
HOG face detector + pre-trained 68-point facial landmark shape predictor model
scipy
euclidean() function computes the distances between eye landmark points for EAR
numpy
Converts dlib landmark objects into arrays for fast mathematical operations
pygame
Loads and loops alarm.wav audio file when drowsiness threshold is exceeded
imutils
shape_to_np() helper converts dlib shape predictor output to NumPy array format

Real-world use
Driver drowsiness accounts for an estimated 20% of road accidents globally. This detection approach is already used in commercial safety systems and can be extended to many scenarios:
🚛
Long-haul trucking
Freight drivers often drive overnight shifts — drowsiness detection integrated into dashcams can alert drivers and fleet managers in real time.
🚌
Public transport
Buses and coaches carrying passengers benefit from continuous monitoring, with the alert system connected to a central dispatch centre.
🏭
Heavy machinery
Crane and forklift operators in warehouses or construction sites can be monitored to prevent fatigue-related workplace accidents.
✈️
Aviation & rail
Pilots and train drivers face high-stakes fatigue risks — a camera-based EAR system can supplement existing regulatory rest limits.
