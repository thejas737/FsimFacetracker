# Webcam-Based 3D Head Tracker for Prepar3D v5.4

A Python webcam head-tracking application that estimates a user's head orientation and relative position, smooths the motion, and sends six-axis tracking data to OpenTrack over UDP. OpenTrack then maps the tracking data to camera movement in Prepar3D (P3D) v5.4.

> \*\*Project status:\*\* This document describes the current `recognize\_face\_pitchboost.py` implementation. It does not claim laboratory-measured accuracy, latency, or frame rate.



## 1\. Project overview

The tracker uses a standard webcam and computer-vision libraries to estimate head pose in real time. Its goal is to provide a low-cost alternative to dedicated head-tracking hardware for use with a flight simulator.

### Data flow

```text
Webcam
  ↓
OpenCV frame capture and image conversion
  ↓
MediaPipe Face Mesh facial landmarks
  ↓
OpenCV solvePnP head-pose estimation
  ↓
Adaptive smoothing and small-motion suppression
  ↓
Six-axis packet: X, Y, Z, Yaw, Pitch, Roll
  ↓ UDP on 127.0.0.1:4242
OpenTrack
  ↓
Axis mapping / response curves / output protocol
  ↓
Prepar3D v5.4 virtual camera
```

The Python program is the custom tracking source. OpenTrack acts as the bridge for receiving, mapping, and forwarding the tracking data to the simulator.





## 2\. Technology stack

|Technology|Role in this project|
|-|-|
|**Python**|Runs the capture, pose-estimation, filtering, and UDP pipeline.|
|**OpenCV (`cv2`)**|Captures webcam frames, converts images to RGB, estimates pose with `solvePnP`, and converts the rotation vector to a rotation matrix.|
|**MediaPipe Face Mesh**|Detects facial landmarks used as image-space reference points.|
|**NumPy**|Stores and processes landmark coordinates, vectors, matrices, and filter state.|
|**Python `socket`**|Sends tracker output to OpenTrack over UDP.|
|**Python `struct`**|Packs the six output values into a binary packet of six little-endian 64-bit floating-point numbers.|
|**Python `time`**|Measures elapsed time so the smoothing filter can adapt to the actual frame interval.|
|**OpenTrack**|Receives the custom tracker data, applies mappings/curves, and sends head-tracking output to the simulator.|
|**Prepar3D v5.4**|Target flight simulator in which the tracked viewpoint is used.|

## 3\. Main algorithm

### 3.1 Webcam capture

The script opens camera index `0` using OpenCV's DirectShow backend on Windows. It requests:

* Resolution: **640 × 480**
* Frame rate: **60 FPS requested**
* Capture buffer: **1 frame requested**, to reduce the chance of processing old frames

These are requested settings, not guarantees. The actual frame rate depends on the webcam, driver, USB connection, lighting, and processing load.

### 3.2 Facial landmark detection

MediaPipe Face Mesh is configured for one face, with detection and tracking confidence thresholds of `0.65`.

The tracker uses six landmarks:

|Landmark|MediaPipe landmark index|Purpose|
|-|-:|-|
|Nose tip|1|Central facial reference|
|Chin|152|Vertical facial reference|
|Left eye corner|33|Eye position reference|
|Right eye corner|263|Eye position reference|
|Left mouth corner|61|Lower-face reference|
|Right mouth corner|291|Lower-face reference|

The detected normalized landmark coordinates are converted into pixel coordinates using the current frame width and height.





### 3.3 3D head-pose estimation

The program defines a compact, simplified 3D face model and matches its six points to the detected 2D image points.

OpenCV's `solvePnP` estimates:

* `rvec`: the estimated head rotation as a Rodrigues rotation vector.
* `tvec`: the estimated translation of the model relative to the camera coordinate system.

The first successful estimate uses `SOLVEPNP\_SQPNP`. Subsequent estimates use `SOLVEPNP\_ITERATIVE`, initialized from the previous raw pose estimate.

The rotation vector is converted into a rotation matrix with `cv2.Rodrigues`. OpenCV's projection-matrix decomposition is then used to obtain Euler-angle values for pitch, yaw, and roll.

**Important limitation:** the 3D face model uses arbitrary model units, the camera focal length is approximated from the image width, and lens distortion is assumed to be zero. Translation and depth are therefore relative estimates, not calibrated measurements in centimetres. Head rotation can also affect the apparent translation.





### 3.4 Adaptive smoothing

The script applies separate filters to rotation and translation. This is intended to reduce small movements caused by landmark noise without making quick head movements feel excessively delayed.

The adaptive filter changes its cutoff based on estimated movement speed:

* When movement is slow, it uses a lower cutoff for stronger smoothing.
* When movement is faster, it raises the cutoff to improve responsiveness.
* The filter calculates its smoothing factor using elapsed time, making its behavior less dependent on a fixed frame rate.

The script also uses a small deadband to suppress tiny changes:

* Rotation deadband: `0.12` degrees
* Position deadband: `0.025` model units

Angle differences are wrapped to the shortest path within the range `\[-180°, 180°)` to avoid artificial large jumps when angles cross the wrap boundary.





### 3.5 Pitch sensitivity and axis conventions

The current pitch output is scaled by `PITCH\_SCALE = 0.85`. This was increased from `0.40` to make head nods more responsive.

Other current settings include:

* `Y\_SCALE = 0.10`
* `INVERT\_YAW = True`
* `INVERT\_Y = True`

The output signs and scales should be checked against the OpenTrack axis display and mapping. OpenTrack response curves and any additional filtering also affect the final in-simulator feel.





### 3.6 UDP output

The tracker sends a packet to:

* IP address: `127.0.0.1`
* UDP port: `4242`
* Packet layout: six little-endian 64-bit floating-point values (`<dddddd`)

The order is:

1. `X`
2. `Y`
3. `Z`
4. `Yaw`
5. `Pitch`
6. `Roll`

Both the Python sender and OpenTrack receiver must use a compatible protocol and agree on the address, port, data format, and axis order. The port `4242` is a project configuration choice, not a universal OpenTrack default.





## 4\. Prerequisites

### Hardware

* A Windows PC capable of running Prepar3D v5.4 and the Python tracker simultaneously.
* A webcam supported by OpenCV.
* Stable webcam placement, preferably near the centre of the monitor and at approximately eye level.
* Reasonably consistent front-facing lighting.
* A clear view of the face; avoid strong backlighting and frequent face occlusion.

A webcam capable of a stable 30 FPS or higher is a practical starting point. The script requests 60 FPS, but the camera may deliver less.

### Software

* Windows 10 or Windows 11.
* Python installed and available from the terminal.
* A compatible version of MediaPipe that provides the legacy `mp.solutions.face\_mesh` API used by this script.
* OpenCV Python package.
* NumPy.
* OpenTrack configured to receive the script's UDP tracking data.
* Prepar3D v5.4.





### Python package installation

Open PowerShell or Command Prompt in the project directory and install the dependencies:

```powershell
py -m pip install opencv-python numpy mediapipe
```

**MediaPipe compatibility note:** this script specifically uses `mp.solutions.face\_mesh`. Not every MediaPipe package/API version exposes that legacy interface. If the script reports that `mediapipe` has no attribute `solutions`, the installed package is not compatible with this code as written. Use a MediaPipe release that includes `solutions.face\_mesh` in the Python environment you intend to run, or update the code to the API supported by your installed version. Avoid installing random versions into multiple Python environments without checking which interpreter runs the script.

Check the interpreter and imports:

```powershell
py -c "import sys, cv2, numpy, mediapipe as mp; print(sys.version); print('OpenCV:', cv2.\_\_version\_\_); print('NumPy:', numpy.\_\_version\_\_); print('MediaPipe:', getattr(mp, '\_\_version\_\_', 'unknown')); print('Has mp.solutions:', hasattr(mp, 'solutions'))"
```

The final line should report `True` for the current script's MediaPipe API requirement.

If several Python versions are installed, use the same interpreter for package installation and execution. For example, if Python 3.11 is the compatible environment on your system:

```powershell
py -3.11 -m pip install opencv-python numpy mediapipe
py -3.11 recognize\_face\_pitchboost.py
```

Replace `3.11` with the version that has the compatible dependencies installed.





## 5\. OpenTrack setup

1. Install and launch OpenTrack.
2. Configure its input to receive the custom UDP tracker data using a protocol/input option compatible with the script's six-double packet.
3. Set the receiving address to `127.0.0.1` and port to `4242`, where applicable.
4. Confirm that the input format expects six 64-bit floating-point values in the order `X, Y, Z, Yaw, Pitch, Roll`. If your selected OpenTrack input expects a different packet format, this script's packet will not be interpreted correctly without adapting one side.
5. Configure the output to the tracking protocol supported by your Prepar3D setup.
6. Check the raw tracker and game-data displays to verify that all six axes respond.
7. Start with OpenTrack's additional smoothing disabled or low, since smoothing is already performed in Python.
8. Enable and tune the X, Y, and Z translation curves if positional movement is desired. Ensure the yaw, pitch, and roll curves are also configured.

OpenTrack builds and configurations can expose different input/output options. Select the receiver and output protocol actually available in your installed version rather than assuming every installation uses the same menu labels.





## 6\. Running the tracker

1. Connect the webcam and close applications that may be using it.
2. Open OpenTrack and confirm its input/UDP receiver is configured.
3. Open PowerShell in the directory containing `recognize\_face\_pitchboost.py`.
4. Run:

```powershell
   py recognize\_face\_pitchboost.py
   ```

   Or specify the compatible Python version:

```powershell
   py -3.11 recognize\_face\_pitchboost.py
   ```

5. Confirm the terminal prints that the tracker is running and sending to `127.0.0.1:4242`.
6. Check OpenTrack's raw tracker data. Move your head gently in each direction and confirm the corresponding axes change.
7. Start Prepar3D v5.4 and verify the simulator camera follows the configured OpenTrack output.
8. Press **Ctrl+C** in the terminal to stop the tracker cleanly.

The current script initializes the filter from the first valid pose. Sit in a comfortable neutral position before beginning movement. If the camera fails to open, check the camera index, permissions, and whether another application is using it.





## 7\. Configuration reference

The main tuning constants are near the top of the script.

|Constant|Current value|Effect|
|-|-:|-|
|`UDP\_IP`|`127.0.0.1`|Local destination for UDP packets|
|`UDP\_PORT`|`4242`|Destination UDP port|
|`ROT\_MIN\_CUTOFF`|`1.8`|Lower cutoff gives smoother but slower rotational response|
|`ROT\_MAX\_CUTOFF`|`8.0`|Upper cutoff used for more responsive rotation|
|`POS\_MIN\_CUTOFF`|`1.2`|Lower cutoff for positional smoothing|
|`POS\_MAX\_CUTOFF`|`5.0`|Upper cutoff for responsive positional tracking|
|`SPEED\_SCALE`|`0.10`|Controls how quickly the filter cutoff rises with movement speed|
|`ROT\_DEADBAND\_DEG`|`0.12`|Suppresses tiny angular changes|
|`POS\_DEADBAND`|`0.025`|Suppresses tiny position changes in model units|
|`PITCH\_SCALE`|`0.85`|Scales pitch response|
|`Y\_SCALE`|`0.10`|Scales the Y translation sent to OpenTrack|
|`INVERT\_YAW`|`True`|Reverses the yaw sign|
|`INVERT\_Y`|`True`|Reverses the Y translation sign|
|`TRACK\_LOST\_RESET\_SECONDS`|`0.60`|Resets filter state after the face has been absent for this duration|
|`MAX\_DT`|`0.10` seconds|Caps the time interval used in the smoothing calculation|

### Practical tuning advice

* **Too much rotational jitter:** lower `ROT\_MIN\_CUTOFF` slightly.
* **Rotations feel delayed:** raise `ROT\_MAX\_CUTOFF` moderately.
* **Position jitters:** lower `POS\_MIN\_CUTOFF` or adjust `POS\_DEADBAND`.
* **Position feels sluggish:** raise `POS\_MAX\_CUTOFF` moderately.
* **Pitch feels too weak:** increase `PITCH\_SCALE` in small steps; check OpenTrack's Pitch mapping as well.
* **Vertical movement feels too weak:** inspect `Y\_SCALE` and OpenTrack's Y mapping curve.
* **Axis moves in the wrong direction:** verify the relevant inversion setting and OpenTrack mapping.
* Change one parameter at a time and test both small movements and larger head turns.

The best values depend on the camera, seating position, lighting, OpenTrack curves, and personal preference.





## 8\. Troubleshooting

### `Could not read from camera`

* Check that the webcam is connected.
* Close video-call, camera, or browser applications that may own the camera.
* Check Windows camera permissions.
* If necessary, test another camera index in `cv2.VideoCapture(0, cv2.CAP\_DSHOW)`.

### `module 'mediapipe' has no attribute 'solutions'`

* The installed MediaPipe package does not expose the API used by this script.
* Check which Python interpreter is running the program.
* Install/use a compatible MediaPipe release that includes `mp.solutions.face\_mesh`, or migrate the script to the installed API.

### OpenTrack receives no data

* Verify the script is running without exceptions.
* Confirm that the receiver is configured for the matching UDP protocol, IP, port, value size, byte order, and axis order.
* Ensure both programs are on the same computer if using `127.0.0.1`.
* Check firewall/security software if local UDP traffic appears to be blocked.
* Confirm OpenTrack is not expecting a different packet format.

### OpenTrack shows data but Prepar3D does not move

* Check OpenTrack's output protocol and whether the output is active.
* Verify that the simulator and its camera-control integration support the selected output.
* Check the game-data display and axis mappings.
* Ensure the response curves are not flat or excessively restricted.
* Confirm that another camera utility is not overriding the viewpoint.

### Movement is jittery or delayed

* Improve lighting and position the webcam securely.
* Reduce unnecessary smoothing in OpenTrack while testing the Python filter.
* Adjust the filter constants gradually.
* Check actual webcam frame rate and CPU load.
* Avoid changing several parameters simultaneously.

### Head movement is interpreted as translation or rotation incorrectly

* Head rotation can change the apparent position of facial landmarks.
* The simplified 3D face model and approximate camera calibration limit pose accuracy.
* Check OpenTrack's six-axis display to distinguish a pose-estimation issue from a mapping issue.
* Revisit the camera position and OpenTrack response curves before increasing sensitivity dramatically.





## 9\. Known limitations

* The face model is simplified and uses arbitrary units rather than a personalized, physically calibrated 3D model.
* Camera focal length is approximated as the frame width; lens distortion is set to zero.
* Translation, particularly depth, is an estimate and may be affected by head rotation and facial landmark noise.
* The program requests 640 × 480 at 60 FPS but does not guarantee those settings are achieved.
* Only one face is tracked.
* Performance depends on lighting, camera quality, face visibility, and computer load.
* The script sends no new tracking packet when a face is missing; after a sufficiently long loss, it resets the filter state. The simulator's behavior during this interval depends on OpenTrack's handling of missing input.
* There is no built-in graphical configuration panel or automatic camera calibration routine.





## 10\. Privacy and network behavior

The current script processes webcam frames locally. It does not contain code to upload frames to a cloud service or save video to disk. It sends only the six tracking values to the local machine's loopback address (`127.0.0.1`) over UDP.

This describes the behavior of this script; other software running on the computer may have its own camera or network behavior.







## 11\. Suggested future improvements

* Calibrate camera intrinsics and lens distortion.
* Use a personalized or better-scaled 3D face model.
* Add a neutral-pose calibration and recenter command.
* Add a configurable camera index and resolution.
* Add a small diagnostic window for FPS, landmarks, pose, and UDP status.
* Improve handling of brief tracking loss and recovery.
* Add configurable translation gains for X, Y, and Z.
* Measure end-to-end latency and compare tracking behavior under controlled conditions.
* Package the application with a clear dependency/environment setup.





## 12\. Summary

This project combines MediaPipe facial landmarks, OpenCV pose estimation, NumPy-based adaptive filtering, and UDP communication to provide webcam-based six-axis head tracking for OpenTrack and Prepar3D v5.4. Its main advantage is accessibility and customizability: it can run with a conventional webcam and an existing PC. Its main trade-off is that monocular webcam pose and translation estimates are less physically calibrated and may be less robust than purpose-built tracking systems.

\---

**Project file:** `recognize\_face\_pitchboost.py`  
**Target simulator:** Prepar3D v5.4  
**Tracking bridge:** OpenTrack  
**UDP destination:** `127.0.0.1:4242`

