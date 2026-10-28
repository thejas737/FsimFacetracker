import cv2
import socket
import struct
import time
import numpy as np
import mediapipe as mp

# ============================================================
# OpenTrack UDP
# ============================================================
UDP_IP = "127.0.0.1"
UDP_PORT = 4242
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ============================================================
# MediaPipe
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
)

# Keep your existing compact model so the basic pose/axis feel
# remains familiar. These dimensions are arbitrary model units.
model_points = np.array([
    (0.0, 0.0, 0.0),       # Nose tip
    (0.0, 5.5, -2.0),      # Chin
    (-3.0, -3.0, -2.0),    # Left eye
    (3.0, -3.0, -2.0),     # Right eye
    (-2.0, 2.5, -1.0),     # Left mouth corner
    (2.0, 2.5, -1.0),     # Right mouth corner
], dtype=np.float64)

# ============================================================
# Camera
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow often behaves better on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Prefer newest frame over a delayed queue

ret, frame = cap.read()
if not ret:
    cap.release()
    sock.close()
    raise RuntimeError("Could not read from camera. Check the camera index/permissions.")

h, w = frame.shape[:2]
focal_length = float(w)
center = (w / 2.0, h / 2.0)
camera_matrix = np.array([
    [focal_length, 0.0, center[0]],
    [0.0, focal_length, center[1]],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

# ============================================================
# Filter tuning
# ============================================================
# Adaptive exponential smoothing:
#   - low alpha while stationary = less jitter
#   - higher alpha while moving = less lag
#
# Alpha is calculated from elapsed time, so behavior is less
# dependent on whether the camera actually delivers 30 or 60 FPS.
ROT_MIN_CUTOFF = 1.8       # Lower = smoother, but more lag
ROT_MAX_CUTOFF = 8.0       # Responsiveness during fast turns
POS_MIN_CUTOFF = 1.2       # Lower = smoother positional tracking
POS_MAX_CUTOFF = 5.0
SPEED_SCALE = 0.10         # Higher = filter reacts sooner to movement

ROT_DEADBAND_DEG = 0.12    # Suppress tiny angular tremor
POS_DEADBAND = 0.025       # Suppress tiny translation tremor in model units
MAX_DT = 0.10
TRACK_LOST_RESET_SECONDS = 0.60

# Keep the existing axis mapping/scaling initially; tune in OpenTrack
# after confirming the filter is stable.
PITCH_SCALE = 0.85       # Increased pitch response; previous 0.40 heavily reduced head nods
Y_SCALE = 0.10
INVERT_YAW = True
INVERT_Y = True

prev_rvec = None
prev_tvec = None
filtered_angles = None       # [pitch, yaw, roll] in degrees, before output scaling
filtered_pos = None          # [x, y, z] in pose/model units
last_frame_time = time.perf_counter()
last_seen_time = last_frame_time
last_send_time = last_frame_time
last_raw_angles = None
last_raw_pos = None

def wrap_angle_delta(delta):
    """Wrap angular difference to [-180, 180) degrees."""
    return (delta + 180.0) % 360.0 - 180.0

def adaptive_alpha(dt, speed, min_cutoff, max_cutoff):
    """Time-aware EMA alpha with higher responsiveness at higher speed."""
    cutoff = min_cutoff + (max_cutoff - min_cutoff) * (
        1.0 - np.exp(-max(0.0, speed) * SPEED_SCALE)
    )
    return float(1.0 - np.exp(-2.0 * np.pi * cutoff * dt))

def suppress_deadband(current, previous, threshold, angular=False):
    """Ignore small changes, without permanently preventing gradual motion."""
    delta = current - previous
    if angular:
        delta = wrap_angle_delta(delta)
    delta[np.abs(delta) < threshold] = 0.0
    return previous + delta

def reset_filter():
    global prev_rvec, prev_tvec, filtered_angles, filtered_pos
    global last_raw_angles, last_raw_pos
    prev_rvec = None
    prev_tvec = None
    filtered_angles = None
    filtered_pos = None
    last_raw_angles = None
    last_raw_pos = None

print(f"Adaptive head tracker running. Sending to {UDP_IP}:{UDP_PORT}")
print("For best results, disable or reduce OpenTrack's extra smoothing initially.")
print("Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        now = time.perf_counter()
        dt = max(1.0 / 120.0, min(now - last_frame_time, MAX_DT))
        last_frame_time = now

        if not ret:
            time.sleep(0.005)
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Don't send noisy or stale pose data when the face disappears.
            if now - last_seen_time > TRACK_LOST_RESET_SECONDS:
                reset_filter()
            continue

        last_seen_time = now
        face_landmarks = results.multi_face_landmarks[0]
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h),
        ], dtype=np.float64)

        # SQPNP initializes the pose; ITERATIVE refines subsequent frames
        # from the previous pose estimate. This avoids treating the
        # previous *smoothed* pose as the filter's only state.
        if prev_rvec is None or prev_tvec is None:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )

        if not success or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue

        # Store the raw PnP solution for the next solver iteration.
        prev_rvec = rvec.copy()
        prev_tvec = tvec.copy()

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        projection_matrix = np.hstack((rotation_matrix, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(projection_matrix)

        # Preserve the original axis convention. Smooth final angles instead
        # of averaging Rodrigues rotation vectors (which are not Euler angles).
        raw_angles = np.array([
            float(euler[0, 0]),   # pitch
            float(-euler[1, 0]) if INVERT_YAW else float(euler[1, 0]),
            float(euler[2, 0]),   # roll
        ], dtype=np.float64)
        raw_pos = np.array([
            float(tvec[0, 0]),
            float(-tvec[1, 0]) if INVERT_Y else float(tvec[1, 0]),
            float(tvec[2, 0]),
        ], dtype=np.float64)

        if filtered_angles is None or filtered_pos is None:
            filtered_angles = raw_angles.copy()
            filtered_pos = raw_pos.copy()
            last_raw_angles = raw_angles.copy()
            last_raw_pos = raw_pos.copy()
        else:
            # Velocity estimates use shortest angular distance, avoiding
            # false huge speeds when an angle crosses +/-180 degrees.
            angle_delta = wrap_angle_delta(raw_angles - last_raw_angles)
            angle_speed = float(np.linalg.norm(angle_delta) / dt)
            pos_speed = float(np.linalg.norm(raw_pos - last_raw_pos) / dt)

            # Remove tiny landmark fluctuations before the main filter.
            stable_angles = suppress_deadband(
                raw_angles, filtered_angles, ROT_DEADBAND_DEG, angular=True
            )
            stable_pos = suppress_deadband(
                raw_pos, filtered_pos, POS_DEADBAND, angular=False
            )

            rot_alpha = adaptive_alpha(dt, angle_speed, ROT_MIN_CUTOFF, ROT_MAX_CUTOFF)
            pos_alpha = adaptive_alpha(dt, pos_speed, POS_MIN_CUTOFF, POS_MAX_CUTOFF)

            angle_step = wrap_angle_delta(stable_angles - filtered_angles)
            filtered_angles = filtered_angles + rot_alpha * angle_step
            filtered_pos = filtered_pos + pos_alpha * (stable_pos - filtered_pos)

            last_raw_angles = raw_angles.copy()
            last_raw_pos = raw_pos.copy()

        # Output in the same six-double order expected by your OpenTrack
        # UDP interface: x, y, z, yaw, pitch, roll.
        x = filtered_pos[0]
        y = filtered_pos[1] * Y_SCALE
        z = filtered_pos[2]
        pitch = filtered_angles[0] * PITCH_SCALE  # Increase/decrease PITCH_SCALE to tune nod sensitivity
        yaw = filtered_angles[1]
        roll = filtered_angles[2]

        try:
            packet = struct.pack("<dddddd", x, y, z, yaw, pitch, roll)
            sock.sendto(packet, (UDP_IP, UDP_PORT))
            last_send_time = now
        except (socket.error, struct.error) as exc:
            print(f"UDP send failed: {exc}")

except KeyboardInterrupt:
    print("\nTracker stopped by user.")
finally:
    cap.release()
    face_mesh.close()
    sock.close()
