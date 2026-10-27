"""
CYBORG AR HUD - Computer Vision Final Year Project
---------------------------------------------------
Webcam-based sci-fi HUD built with OpenCV + MediaPipe.

Features
  * Face detection with a "facial recognition" style overlay
  * Real-time hand tracking (21 landmarks) drawn as a glowing skeleton
  * Gesture control:
        Open Palm -> circuit overlays around the hand
        Fist      -> "cyborg evolution" progress bar fills up
        Pinch     -> face scanning animation
  * System status panel, neural-network graphic, FPS counter
  * Press 'q' to quit

Run:  python main.py
"""

import math
import random
import time

import cv2
import mediapipe as mp
import numpy as np

# ----------------------------- colours (BGR) -----------------------------
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)

FONT = cv2.FONT_HERSHEY_SIMPLEX

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# ----------------------------- gesture logic -----------------------------


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_points(hand_landmarks, w, h):
    """Convert normalized landmarks to pixel coordinates."""
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


def fingers_extended(pts):
    """
    Return list of 5 booleans [thumb, index, middle, ring, pinky].
    Orientation independent: a finger is 'extended' when its tip is farther
    from the wrist than its middle joint.
    """
    wrist = pts[0]
    tips = [4, 8, 12, 16, 20]
    joints = [3, 6, 10, 14, 18]
    result = []
    for tip, joint in zip(tips, joints):
        result.append(dist(pts[tip], wrist) > dist(pts[joint], wrist) * 1.05)
    return result


def classify_gesture(pts):
    """Return one of: OPEN_PALM, FIST, PINCH, NONE."""
    ext = fingers_extended(pts)
    hand_size = dist(pts[0], pts[9]) + 1e-6  # wrist -> middle knuckle
    pinch_dist = dist(pts[4], pts[8]) / hand_size

    # Pinch: thumb and index tips close together
    if pinch_dist < 0.35 and not all(ext[2:]):
        return "PINCH"
    if sum(ext[1:]) >= 4:
        return "OPEN_PALM"
    if sum(ext[1:]) == 0:
        return "FIST"
    return "NONE"


# ----------------------------- drawing helpers -----------------------------


def glow_line(img, p1, p2, color=WHITE, thickness=2):
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_hand_skeleton(img, pts):
    for a, b in HAND_CONNECTIONS:
        glow_line(img, pts[a], pts[b], WHITE, 2)
    for i, p in enumerate(pts):
        cv2.circle(img, p, 6, CYAN, -1, cv2.LINE_AA)
        cv2.circle(img, p, 9, WHITE, 1, cv2.LINE_AA)
    # crosshair on fingertip of index finger
    x, y = pts[8]
    cv2.line(img, (x - 18, y), (x + 18, y), WHITE, 1, cv2.LINE_AA)
    cv2.line(img, (x, y - 18), (x, y + 18), WHITE, 1, cv2.LINE_AA)


def draw_circuit_overlay(img, pts, t):
    """Random-looking but stable circuit traces that originate at the hand."""
    rng = random.Random(int(t * 4))  # changes 4x per second -> flicker effect
    h, w = img.shape[:2]
    for i in (4, 8, 12, 16, 20):
        x, y = pts[i]
        for _ in range(3):
            cx, cy = x, y
            for _ in range(rng.randint(3, 6)):
                if rng.random() < 0.5:
                    nx, ny = cx + rng.choice([-1, 1]) * rng.randint(20, 60), cy
                else:
                    nx, ny = cx, cy + rng.choice([-1, 1]) * rng.randint(20, 60)
                nx = int(np.clip(nx, 0, w - 1))
                ny = int(np.clip(ny, 0, h - 1))
                cv2.line(img, (cx, cy), (nx, ny), CYAN, 1, cv2.LINE_AA)
                cv2.circle(img, (nx, ny), 3, CYAN, -1, cv2.LINE_AA)
                cx, cy = nx, ny
    # pulsing ring on the palm
    r = int(40 + 10 * math.sin(t * 6))
    cv2.circle(img, pts[9], r, CYAN, 2, cv2.LINE_AA)


def draw_face_box(img, box, scanning, t):
    x, y, bw, bh = box
    col = RED
    cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 2)
    # corner brackets
    L = 20
    for (cx, cy, dx, dy) in [(x, y, 1, 1), (x + bw, y, -1, 1),
                             (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * L, cy), CYAN, 3)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), CYAN, 3)
    cv2.putText(img, "FACIAL RECOGNITION ACTIVE", (x, y - 10), FONT, 0.5, RED, 1,
                cv2.LINE_AA)
    # centre reticle
    cx, cy = x + bw // 2, y + bh // 2
    cv2.line(img, (cx - 12, cy), (cx + 12, cy), WHITE, 1)
    cv2.line(img, (cx, cy - 12), (cx, cy + 12), WHITE, 1)

    if scanning:
        # moving horizontal scan line
        sy = y + int((math.sin(t * 5) * 0.5 + 0.5) * bh)
        cv2.line(img, (x, sy), (x + bw, sy), GREEN, 2)
        cv2.putText(img, "SCANNING...", (x, y + bh + 20), FONT, 0.5, GREEN, 1,
                    cv2.LINE_AA)


def draw_neural_net(img, origin=(10, 160), layers=(3, 5, 4, 2), spacing=45, vgap=40):
    """Small decorative neural network on the left side of the screen."""
    ox, oy = origin
    nodes = []
    for li, n in enumerate(layers):
        col = []
        for ni in range(n):
            col.append((ox + li * spacing, oy + ni * vgap - (n * vgap) // 2))
        nodes.append(col)
    for l1, l2 in zip(nodes[:-1], nodes[1:]):
        for a in l1:
            for b in l2:
                cv2.line(img, a, b, (180, 180, 0), 1, cv2.LINE_AA)
    for layer in nodes:
        for p in layer:
            cv2.circle(img, p, 6, CYAN, -1, cv2.LINE_AA)


def draw_status_panel(img, fps, gesture, face_found, evolution):
    h, w = img.shape[:2]
    pw, ph = 230, 120
    x0, y0 = w - pw - 10, 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), DARK, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), CYAN, 1)
    cv2.putText(img, "SYSTEM STATUS", (x0 + 8, y0 + 20), FONT, 0.5, CYAN, 1, cv2.LINE_AA)
    lines = [
        "NEURAL LINK: ACTIVE",
        f"BIOMETRIC SCAN: {'OK' if face_found else 'NO FACE'}",
        f"GESTURE: {gesture}",
        f"EVOLUTION: {int(evolution)}%",
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (x0 + 8, y0 + 45 + i * 18), FONT, 0.4, GREEN, 1,
                    cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}", (w - 110, y0 + ph + 22), FONT, 0.55, GREEN, 2,
                cv2.LINE_AA)


def draw_evolution_bar(img, evolution):
    h, w = img.shape[:2]
    bw, bh = 300, 18
    x0, y0 = w - bw - 15, h - 60
    cv2.putText(img, f"CYBORG EVOLUTION: {evolution:.0f}%", (x0, y0 - 8), FONT, 0.5,
                ORANGE, 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), ORANGE, 1)
    fill = int(bw * evolution / 100)
    cv2.rectangle(img, (x0, y0), (x0 + fill, y0 + bh), ORANGE, -1)


def draw_footer(img, gesture, face_found):
    h, _ = img.shape[:2]
    cv2.putText(img, f"GESTURE: {gesture}", (15, h - 40), FONT, 0.55, CYAN, 1,
                cv2.LINE_AA)
    cv2.putText(img, f"BIOMETRIC: {'FACE DETECTED' if face_found else 'SEARCHING'}",
                (15, h - 18), FONT, 0.55, CYAN, 1, cv2.LINE_AA)


def draw_help(img):
    lines = ["Show your hand gestures:",
             "  Open Palm -> Circuit overlays",
             "  Fist      -> Evolution progress",
             "  Pinch     -> Face scanning",
             "  Press 'q' to quit"]
    for i, t in enumerate(lines):
        cv2.putText(img, t, (15, 25 + i * 18), FONT, 0.45, WHITE, 1, cv2.LINE_AA)


# ----------------------------- main loop -----------------------------


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check that it is connected and not in use.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    evolution = 0.0
    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.6) as hands, \
            mp_face.FaceDetection(model_selection=0,
                                  min_detection_confidence=0.6) as faces:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror view
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_res = hands.process(rgb)
            face_res = faces.process(rgb)

            gesture = "NONE"
            hand_pts = None
            if hand_res.multi_hand_landmarks:
                hand_pts = hand_points(hand_res.multi_hand_landmarks[0], w, h)
                gesture = classify_gesture(hand_pts)

            t = time.time()

            # --- face ---
            face_found = False
            if face_res.detections:
                face_found = True
                bb = face_res.detections[0].location_data.relative_bounding_box
                box = (max(int(bb.xmin * w), 0), max(int(bb.ymin * h), 0),
                       int(bb.width * w), int(bb.height * h))
                draw_face_box(frame, box, scanning=(gesture == "PINCH"), t=t)

            # --- hand + gesture effects ---
            if hand_pts:
                draw_hand_skeleton(frame, hand_pts)
                if gesture == "OPEN_PALM":
                    draw_circuit_overlay(frame, hand_pts, t)
                elif gesture == "FIST":
                    evolution = min(100.0, evolution + 0.6)
            if gesture != "FIST":
                evolution = max(0.0, evolution - 0.1)  # slow decay

            # --- HUD ---
            draw_neural_net(frame)
            draw_status_panel(frame, fps, gesture, face_found, evolution)
            if evolution > 0:
                draw_evolution_bar(frame, evolution)
            draw_footer(frame, gesture, face_found)
            draw_help(frame)

            # FPS
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            cv2.imshow("Cyborg AR HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
