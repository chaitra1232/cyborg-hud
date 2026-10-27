"""
CYBORG AR HUD - ANIMATED EDITION
---------------------------------
Webcam sci-fi HUD built with OpenCV + MediaPipe, fully animated.

Animations
  * Boot sequence on start-up (typing text + progress bar)
  * Neon glow (additive blending + blur), moving grid, scanlines
  * Neural network with signal pulses travelling along the connections
  * Radar sweep, scrolling data ticker, live waveform, status panel
  * Face: pulsing corner brackets, rotating dashed rings, landmark points,
    flickering match %, side callouts
  * Hand: glowing skeleton, fingertip light trail, pulsing joints

Gestures
  Open Palm -> rotating rings + circuit traces that grow + particle burst
  Fist      -> energy rings converge, evolution bar charges (shockwave at 100%)
  Pinch     -> face scan: sweeping scan line, mesh, progress, IDENTITY VERIFIED

Keys:  q = quit   r = start/stop recording (hud_recording.mp4)
       s = snapshot   h = show/hide help
"""

import math
import random
import time
from collections import Counter, deque

import cv2
import numpy as np

# ----------------------------- colours (BGR) -----------------------------
CYAN = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
TIPS = (4, 8, 12, 16, 20)

# ----------------------------- small helpers -----------------------------


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def ip(p):
    return (int(p[0]), int(p[1]))


def dim(color, k):
    k = max(0.0, min(1.0, k))
    return tuple(int(v * k) for v in color)


def txt(img, s, org, color, scale=0.5, th=1):
    cv2.putText(img, s, ip(org), FONT, scale, color, th, AA)


def txt_center(img, s, cx, y, color, scale=0.8, th=2):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, th)
    cv2.putText(img, s, (int(cx - tw / 2), int(y)), FONT, scale, color, th, AA)


def dashed_circle(img, c, r, offset, color, th, segs, frac):
    if r < 2:
        return
    step = 360.0 / segs
    for i in range(segs):
        a0 = offset + i * step
        cv2.ellipse(img, ip(c), (int(r), int(r)), 0, a0, a0 + step * frac,
                    color, th, AA)


def hand_points(hand_landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]


# ----------------------------- gesture logic -----------------------------


def fingers_extended(pts):
    wrist = pts[0]
    joints = [3, 6, 10, 14, 18]
    return [dist(pts[t], wrist) > dist(pts[j], wrist) * 1.05
            for t, j in zip(TIPS, joints)]


def classify_gesture(pts):
    ext = fingers_extended(pts)
    hand_size = dist(pts[0], pts[9]) + 1e-6
    pinch_dist = dist(pts[4], pts[8]) / hand_size
    # In a real pinch the index tip stays far from the wrist; in a fist it curls in.
    index_out = dist(pts[8], pts[0]) / hand_size > 1.15
    if pinch_dist < 0.35 and index_out and not all(ext[2:]):
        return "PINCH"
    if sum(ext[1:]) >= 4:
        return "OPEN_PALM"
    if sum(ext[1:]) == 0:
        return "FIST"
    return "NONE"


# ----------------------------- animation state -----------------------------


class State:
    def __init__(self):
        self.t0 = time.time()
        self.evolution = 0.0
        self.scan = 0.0
        self.trail = deque(maxlen=24)
        self.particles = []     # [x, y, vx, vy, life, maxlife]
        self.pulses = []        # [edge_index, progress]
        self.hist = deque(maxlen=6)
        self.gesture = "NONE"
        self.glitch_until = 0.0
        self.wave = deque([0.0] * 60, maxlen=60)
        self.fps = 30.0
        self.complete_at = None
        self.hexid = "0x00000000"
        self.hex_t = 0.0
        self.nn = None
        self.nn_size = None
        self.recording = False
        self.writer = None
        self.show_help = True


# ----------------------------- background elements -----------------------------


def draw_grid(fx, w, h, t, s):
    sp = max(30, int(60 * s))
    for y in range(-sp + int((t * 25) % sp), h, sp):
        cv2.line(fx, (0, y), (w, y), (28, 28, 0), 1)
    for x in range(0, w, sp):
        cv2.line(fx, (x, 0), (x, h), (20, 20, 0), 1)


def build_nn(w, h, s):
    layers = (4, 6, 6, 3)
    spacing = int(52 * s)
    vgap = int(34 * s)
    ox = int(34 * s)
    oy = int(h * 0.42)
    nodes = []
    for li, n in enumerate(layers):
        nodes.append([(ox + li * spacing, oy + int((ni - (n - 1) / 2) * vgap))
                      for ni in range(n)])
    edges = [(a, b) for l1, l2 in zip(nodes[:-1], nodes[1:]) for a in l1 for b in l2]
    return nodes, edges


def draw_nn(fx, st, t, dt, s, fs):
    nodes, edges = st.nn
    for a, b in edges:
        cv2.line(fx, a, b, (70, 70, 0), 1, AA)
    if random.random() < 0.6:
        st.pulses.append([random.randrange(len(edges)), 0.0])
    alive = []
    for e, p in st.pulses:
        p += dt * 1.6
        if p < 1.0:
            a, b = edges[e]
            pos = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
            cv2.circle(fx, ip(pos), max(2, int(3 * s)), WHITE, -1, AA)
            alive.append([e, p])
    st.pulses = alive[-50:]
    i = 0
    for col in nodes:
        for p in col:
            r = max(2, int((5 + 2 * math.sin(t * 3 + i)) * s))
            cv2.circle(fx, p, r, CYAN, -1, AA)
            i += 1
    top = nodes[0][0][1] - int(30 * s)
    txt(fx, "NEURAL NET // ACTIVE", (nodes[0][0][0] - 10, top), dim(CYAN, 0.9), fs * 0.8)


def draw_radar(fx, t, s, fs, c, r):
    for k in (1.0, 0.66, 0.33):
        cv2.circle(fx, c, int(r * k), (90, 140, 0), 1, AA)
    cv2.line(fx, (c[0] - r, c[1]), (c[0] + r, c[1]), (60, 90, 0), 1)
    cv2.line(fx, (c[0], c[1] - r), (c[0], c[1] + r), (60, 90, 0), 1)
    sweep = (t * 120) % 360
    for k in range(0, 48, 4):
        a = math.radians(sweep - k)
        end = (int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a)))
        cv2.line(fx, c, end, dim(GREEN, 1 - k / 48), 1, AA)
    for j in range(6):
        ang = (j * 67 + 20) % 360
        rad = (0.25 + 0.11 * j) * r
        d = (sweep - ang) % 360
        if d < 100:
            b = 1 - d / 100
            pos = (c[0] + rad * math.cos(math.radians(ang)),
                   c[1] + rad * math.sin(math.radians(ang)))
            cv2.circle(fx, ip(pos), max(2, int(3 * s)), dim(GREEN, b), -1, AA)
    txt(fx, "RADAR", (c[0] - r, c[1] - r - 8), dim(GREEN, 0.8), fs * 0.8)


def update_particles(fx, st, dt, s):
    alive = []
    for p in st.particles:
        p[4] -= dt
        if p[4] > 0:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            k = p[4] / p[5]
            cv2.circle(fx, ip(p), max(1, int(3 * s * k)), dim(CYAN, k), -1, AA)
            alive.append(p)
    st.particles = alive[-400:]


# ----------------------------- face -----------------------------


def draw_face(fx, face, st, t, s, fs):
    x, y, bw, bh = face["box"]
    pulse = int(4 * s * math.sin(t * 4))
    L = int(26 * s) + pulse
    cv2.rectangle(fx, (x, y), (x + bw, y + bh), (0, 0, 170), 1)
    for cx, cy, dx, dy in ((x, y, 1, 1), (x + bw, y, -1, 1),
                           (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)):
        cv2.line(fx, (cx, cy), (cx + dx * L, cy), CYAN, 3, AA)
        cv2.line(fx, (cx, cy), (cx, cy + dy * L), CYAN, 3, AA)

    cx, cy = x + bw // 2, y + bh // 2
    r = int(10 * s)
    cv2.line(fx, (cx - r, cy), (cx + r, cy), WHITE, 1, AA)
    cv2.line(fx, (cx, cy - r), (cx, cy + r), WHITE, 1, AA)

    big = max(bw, bh)
    dashed_circle(fx, (cx, cy), int(big * 0.72), t * 40, CYAN, 1, 28, 0.55)
    dashed_circle(fx, (cx, cy), int(big * 0.82), -t * 60, (0, 0, 200), 1, 40, 0.3)

    kps = face["kps"]
    for p in kps:
        cv2.circle(fx, p, max(2, int(4 * s)), GREEN, -1, AA)
        cv2.circle(fx, p, int(8 * s + 2 * math.sin(t * 5)), GREEN, 1, AA)
    if len(kps) >= 4:
        cv2.line(fx, kps[0], kps[1], GREEN, 1, AA)
        cv2.line(fx, kps[0], kps[2], GREEN, 1, AA)
        cv2.line(fx, kps[1], kps[2], GREEN, 1, AA)
        cv2.line(fx, kps[2], kps[3], GREEN, 1, AA)

    # side callouts (like the video)
    sx = x - int(80 * s)
    for k, off in enumerate((-34, 4)):
        ry = cy + int(off * s)
        rw, rh = int(52 * s), int(18 * s)
        cv2.rectangle(fx, (sx, ry), (sx + rw, ry + rh), CYAN, 1)
        fill = int(rw * (0.3 + 0.7 * abs(math.sin(t * 2 + k * 1.7))))
        cv2.rectangle(fx, (sx + 2, ry + 2), (sx + fill, ry + rh - 2), dim(CYAN, 0.6), -1)
        cv2.line(fx, (sx + rw, ry + rh // 2), (x, ry + rh // 2), dim(CYAN, 0.7), 1, AA)

    txt(fx, "FACIAL RECOGNITION ACTIVE", (x, y - int(12 * s)), RED, fs * 0.9)
    match = 97.0 + 2.9 * abs(math.sin(t * 1.7)) + random.uniform(0, 0.08)
    txt(fx, f"ID {st.hexid}", (x, y + bh + int(20 * s)), CYAN, fs * 0.85)
    txt(fx, f"MATCH {min(match, 99.9):.1f}%", (x, y + bh + int(40 * s)), GREEN, fs * 0.85)

    if st.scan > 0:
        phase = (t * 1.2) % 2
        ph = phase if phase < 1 else 2 - phase
        sy = y + int(ph * bh)
        gstep = max(12, int(20 * s))
        for gy in range(y, y + bh, gstep):
            cv2.line(fx, (x, gy), (x + bw, gy), (0, 60, 0), 1)
        for gx in range(x, x + bw, gstep):
            cv2.line(fx, (gx, y), (gx, y + bh), (0, 60, 0), 1)
        for k in range(12):
            yy = sy - k * 3
            if y <= yy <= y + bh:
                cv2.line(fx, (x, yy), (x + bw, yy), dim(GREEN, 1 - k / 12), 1, AA)
        label = "IDENTITY VERIFIED" if st.scan >= 100 else f"SCANNING... {int(st.scan)}%"
        txt(fx, label, (x, y + bh + int(60 * s)), GREEN, fs * 0.9, 2)
        bar_w = bw
        cv2.rectangle(fx, (x, y + bh + int(68 * s)),
                      (x + bar_w, y + bh + int(76 * s)), GREEN, 1)
        cv2.rectangle(fx, (x, y + bh + int(68 * s)),
                      (x + int(bar_w * st.scan / 100), y + bh + int(76 * s)), GREEN, -1)


# ----------------------------- hand -----------------------------


def circuit_path(seed, start, w, h, s):
    rng = random.Random(seed)
    cx, cy = start
    pts = [ip((cx, cy))]
    for _ in range(rng.randint(4, 7)):
        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        length = rng.randint(25, 70) * s
        cx = int(np.clip(cx + dx * length, 0, w - 1))
        cy = int(np.clip(cy + dy * length, 0, h - 1))
        pts.append((cx, cy))
    return pts


def draw_path(fx, path, p, color):
    n = len(path) - 1
    if n < 1:
        return
    total = p * n
    k = int(total)
    frac = total - k
    for i in range(min(k, n)):
        cv2.line(fx, path[i], path[i + 1], color, 1, AA)
        cv2.circle(fx, path[i + 1], 3, color, -1, AA)
    if k < n:
        a, b = path[k], path[k + 1]
        head = (int(a[0] + (b[0] - a[0]) * frac), int(a[1] + (b[1] - a[1]) * frac))
        cv2.line(fx, a, head, color, 1, AA)
        cv2.circle(fx, head, 4, WHITE, -1, AA)


def draw_hand(fx, st, pts, g, t, s, w, h, fs):
    ps = dist(pts[0], pts[9]) + 1
    palm = pts[9]

    # fingertip light trail
    st.trail.append(pts[8])
    tr = list(st.trail)
    for i in range(1, len(tr)):
        k = i / len(tr)
        cv2.line(fx, tr[i - 1], tr[i], dim(CYAN, k), max(1, int(5 * k * s)), AA)

    # skeleton
    for a, b in HAND_CONNECTIONS:
        cv2.line(fx, pts[a], pts[b], WHITE, 2, AA)
    for i, p in enumerate(pts):
        cv2.circle(fx, p, max(3, int(5 * s)), CYAN, -1, AA)
    for tip in TIPS:
        r = int((10 + 3 * math.sin(t * 6 + tip)) * s)
        cv2.circle(fx, pts[tip], r, WHITE, 1, AA)
    x, y = pts[8]
    c = int(20 * s)
    cv2.line(fx, (x - c, y), (x + c, y), WHITE, 1, AA)
    cv2.line(fx, (x, y - c), (x, y + c), WHITE, 1, AA)

    if g == "OPEN_PALM":
        dashed_circle(fx, palm, int(ps * 1.5), t * 90, CYAN, 2, 16, 0.6)
        dashed_circle(fx, palm, int(ps * 1.9), -t * 60, WHITE, 1, 24, 0.45)
        dashed_circle(fx, palm, int(ps * 2.3), t * 30, CYAN, 1, 36, 0.3)
        period = 2.5
        epoch = int(t / period)
        prog = min(1.0, ((t % period) / period) * 1.6)
        for i, tip in enumerate(TIPS):
            for b in range(3):
                path = circuit_path(epoch * 100 + i * 10 + b, pts[tip], w, h, s)
                draw_path(fx, path, prog, CYAN)
        for tip in TIPS:
            for _ in range(2):
                ang = random.uniform(0, 2 * math.pi)
                sp = random.uniform(30, 140) * s
                life = random.uniform(0.6, 1.2)
                st.particles.append([pts[tip][0], pts[tip][1],
                                     math.cos(ang) * sp, math.sin(ang) * sp, life, life])
        txt(fx, "CIRCUIT OVERLAY", (palm[0] - 60 * s, palm[1] + ps * 2.6), CYAN, fs)

    elif g == "FIST":
        for i in range(3):
            ph = ((t * 1.5) + i / 3) % 1
            r = int(ps * (2.4 - 1.9 * ph))
            cv2.circle(fx, palm, max(2, r), dim(ORANGE, ph), 2, AA)
        core = int(ps * 0.45 * (0.5 + st.evolution / 100))
        cv2.circle(fx, palm, max(2, core), dim(ORANGE, 0.8), -1, AA)
        txt(fx, "CHARGING", (palm[0] - 40 * s, palm[1] + ps * 2.6), ORANGE, fs)

    elif g == "PINCH":
        a, b = pts[4], pts[8]
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.line(fx, a, b, GREEN, 2, AA)
        cv2.circle(fx, mid, int((12 + 4 * math.sin(t * 8)) * s), GREEN, 2, AA)
        txt(fx, f"PINCH {int(dist(a, b))}px", (mid[0] + 14, mid[1] - 10), GREEN, fs)


# ----------------------------- HUD panels -----------------------------


def draw_evolution_bar(fx, evo, t, s, fs, w, h):
    bw, bh = int(300 * s), int(20 * s)
    x0, y0 = w - bw - int(18 * s), h - int(70 * s)
    txt(fx, f"CYBORG EVOLUTION: {evo:.0f}%", (x0, y0 - 8), ORANGE, fs)
    cv2.rectangle(fx, (x0, y0), (x0 + bw, y0 + bh), ORANGE, 1)
    fill = int(bw * evo / 100)
    if fill > 0:
        tmp = np.zeros((bh, bw, 3), np.uint8)
        cv2.rectangle(tmp, (0, 0), (fill, bh), dim(ORANGE, 0.85), -1)
        shift = int((t * 50) % 18)
        for xs in range(-bh, fill, 18):
            cv2.line(tmp, (xs + shift, bh), (xs + shift + bh, 0), (0, 70, 140), 3)
        tmp[:, fill:] = 0
        fx[y0:y0 + bh, x0:x0 + bw] = cv2.add(fx[y0:y0 + bh, x0:x0 + bw], tmp)


def draw_panel(fx, st, t, s, fs, w, h, face_found, g):
    pw, ph = int(270 * s), int(185 * s)
    x0, y0 = w - pw - int(12 * s), int(12 * s)
    cv2.rectangle(fx, (x0, y0), (x0 + pw, y0 + ph), CYAN, 1)
    L = int(12 * s)
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x0 + pw, y0, -1, 1),
                           (x0, y0 + ph, 1, -1), (x0 + pw, y0 + ph, -1, -1)):
        cv2.line(fx, (cx, cy), (cx + dx * L, cy), WHITE, 2, AA)
        cv2.line(fx, (cx, cy), (cx, cy + dy * L), WHITE, 2, AA)
    cursor = "_" if int(t * 2) % 2 == 0 else " "
    txt(fx, "SYSTEM STATUS" + cursor, (x0 + 10 * s, y0 + 22 * s), CYAN, fs, 1)
    lines = [
        "NEURAL LINK: ACTIVE",
        f"BIOMETRIC: {'LOCK' if face_found else 'NO FACE'}",
        f"GESTURE: {g}",
        f"EVOLUTION: {int(st.evolution)}%",
        f"SCAN: {int(st.scan)}%",
    ]
    for i, line in enumerate(lines):
        txt(fx, line, (x0 + 10 * s, y0 + (44 + i * 17) * s), GREEN, fs * 0.85)

    # live waveform
    wx0, wx1 = x0 + int(10 * s), x0 + pw - int(10 * s)
    wy = y0 + ph - int(22 * s)
    amp = 14 * s
    v = 0.5 * math.sin(t * 4) + random.uniform(-0.3, 0.3)
    if g != "NONE":
        v += 0.4 * math.sin(t * 17)
    st.wave.append(max(-1.0, min(1.0, v)))
    n = len(st.wave)
    pts = np.array([[wx0 + (wx1 - wx0) * i / (n - 1), wy - val * amp]
                    for i, val in enumerate(st.wave)], np.int32)
    cv2.polylines(fx, [pts], False, GREEN, 1, AA)

    # FPS / REC
    txt(fx, f"FPS: {st.fps:.1f}", (x0 + pw - 100 * s, y0 + ph + 24 * s), GREEN, fs * 1.1, 2)
    if st.recording and int(t * 2) % 2 == 0:
        cv2.circle(fx, (x0 + int(10 * s), y0 + ph + int(18 * s)), int(6 * s), RED, -1, AA)
        txt(fx, "REC", (x0 + 22 * s, y0 + ph + 24 * s), RED, fs)


def draw_footer(fx, st, t, s, fs, w, h, face_found, g):
    txt(fx, f"GESTURE: {g}", (15, h - int(56 * s)), CYAN, fs * 1.1)
    txt(fx, f"BIOMETRIC: {'FACE DETECTED' if face_found else 'SEARCHING'}",
        (15, h - int(36 * s)), CYAN, fs * 1.1)
    msg = ("   //  NEURAL LINK ACTIVE  //  BIOMETRIC SCAN OK  //  TRACKING 21 LANDMARKS"
           "  //  FRAME SYNC LOCKED  //  CYBORG PROTOCOL ONLINE   ")
    sc = fs * 0.75
    tw = cv2.getTextSize(msg, FONT, sc, 1)[0][0]
    off = int((t * 80) % tw)
    x = -off
    while x < w:
        txt(fx, msg, (x, h - int(10 * s)), dim(CYAN, 0.7), sc)
        x += tw


def draw_help(fx, s, fs):
    lines = ["Show your hand gestures:",
             "  Open Palm -> Circuit overlays",
             "  Fist      -> Evolution progress",
             "  Pinch     -> Face scanning",
             "  q quit | r record | s snapshot | h help"]
    for i, line in enumerate(lines):
        txt(fx, line, (15, int((26 + i * 19) * s)), WHITE, fs * 0.9)


def draw_boot(frame, el, s, fs):
    dur = 3.2
    if el >= dur:
        return frame
    k = 1 - el / dur
    dark = (frame * (1 - 0.85 * k)).astype(np.uint8)
    lines = [
        (0.2, "BOOTING CYBORG CORE v2.0"),
        (0.7, "LOADING VISION MODULES ..... OK"),
        (1.2, "CALIBRATING HAND TRACKER ... OK"),
        (1.7, "NEURAL LINK HANDSHAKE ...... OK"),
        (2.2, "BIOMETRIC SENSORS ONLINE ... OK"),
    ]
    h, w = dark.shape[:2]
    y = int(h * 0.35)
    for start, line in lines:
        if el >= start:
            shown = line[: int((el - start) * 45)]
            cv2.putText(dark, shown, (int(w * 0.3), y), FONT, fs * 1.4, GREEN, 2, AA)
        y += int(36 * s)
    bw = int(w * 0.4)
    x0, y0 = int(w * 0.3), y + int(10 * s)
    cv2.rectangle(dark, (x0, y0), (x0 + bw, y0 + int(16 * s)), GREEN, 1)
    cv2.rectangle(dark, (x0, y0), (x0 + int(bw * el / dur), y0 + int(16 * s)), GREEN, -1)
    return dark


def compose(frame, fx):
    h, w = fx.shape[:2]
    small = cv2.resize(fx, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3)
    glow = cv2.resize(small, (w, h))
    out = cv2.add(frame, fx)
    return cv2.add(out, glow)


# ----------------------------- render one frame -----------------------------


def render(frame, st, hand_pts, raw_gesture, face, t, dt):
    h, w = frame.shape[:2]
    s = h / 720.0
    fs = max(0.4, 0.5 * s)

    # gesture smoothing (majority vote) + glitch on change
    st.hist.append(raw_gesture if hand_pts else "NONE")
    stable = Counter(st.hist).most_common(1)[0][0]
    if stable != st.gesture:
        st.gesture = stable
        st.glitch_until = t + 0.3
    g = st.gesture if hand_pts else "NONE"
    if not hand_pts:
        st.trail.clear()

    # state updates
    if g == "FIST":
        st.evolution = min(100.0, st.evolution + 40 * dt)
    else:
        st.evolution = max(0.0, st.evolution - 12 * dt)
    if st.evolution >= 100 and st.complete_at is None:
        st.complete_at = t
    if st.evolution < 100:
        st.complete_at = None
    if g == "PINCH" and face:
        st.scan = min(100.0, st.scan + 45 * dt)
    else:
        st.scan = max(0.0, st.scan - 60 * dt)
    if t - st.hex_t > 0.1:
        st.hexid = "0x%08X" % random.getrandbits(32)
        st.hex_t = t
    if random.random() < 0.004:
        st.glitch_until = max(st.glitch_until, t + 0.15)

    if st.nn is None or st.nn_size != (w, h):
        st.nn = build_nn(w, h, s)
        st.nn_size = (w, h)

    # base image: darker + scanlines
    frame = cv2.convertScaleAbs(frame, alpha=0.8)
    frame[::3] = (frame[::3] * 0.85).astype(np.uint8)
    pw, ph = int(270 * s), int(185 * s)
    px, py = w - pw - int(12 * s), int(12 * s)
    frame[py:py + ph, px:px + pw] = (frame[py:py + ph, px:px + pw] * 0.45).astype(np.uint8)

    # neon layer
    fx = np.zeros_like(frame)
    draw_grid(fx, w, h, t, s)
    draw_nn(fx, st, t, dt, s, fs)
    draw_radar(fx, t, s, fs, (int(90 * s), h - int(190 * s)), int(60 * s))
    if face:
        draw_face(fx, face, st, t, s, fs)
    if hand_pts:
        draw_hand(fx, st, hand_pts, g, t, s, w, h, fs)
    update_particles(fx, st, dt, s)
    if st.evolution > 0:
        draw_evolution_bar(fx, st.evolution, t, s, fs, w, h)

    # evolution complete: shockwaves + flashing text
    if st.complete_at is not None:
        c = ip(hand_pts[9]) if hand_pts else (w // 2, h // 2)
        for i in range(3):
            e = t - st.complete_at - i * 0.25
            if 0 < e < 1.4:
                cv2.circle(fx, c, int(e * 500 * s), dim(ORANGE, 1 - e / 1.4), 3, AA)
        if int(t * 4) % 2 == 0:
            txt_center(fx, "EVOLUTION COMPLETE", w / 2, h * 0.12, ORANGE, 1.2 * s + 0.3, 3)

    draw_panel(fx, st, t, s, fs, w, h, face is not None, g)
    draw_footer(fx, st, t, s, fs, w, h, face is not None, g)
    if st.show_help:
        draw_help(fx, s, fs)

    out = compose(frame, fx)

    if st.complete_at is not None and t - st.complete_at < 0.3:
        v = int(120 * (1 - (t - st.complete_at) / 0.3))
        out = cv2.add(out, np.full_like(out, v))

    # glitch
    if t < st.glitch_until:
        for _ in range(3):
            by = random.randrange(0, max(1, h - 40))
            bh = random.randint(10, 40)
            dx = random.randint(-25, 25)
            out[by:by + bh] = np.roll(out[by:by + bh], dx, axis=1)

    return draw_boot(out, t, s, fs)


# ----------------------------- main loop -----------------------------


def main():
    import mediapipe as mp  # imported here so the drawing code can be tested without it

    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Close other apps using it, "
                         "or change VideoCapture(0) to VideoCapture(1).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    st = State()
    prev = time.time()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                        min_tracking_confidence=0.6) as hands, \
            mp_face.FaceDetection(model_selection=0,
                                  min_detection_confidence=0.6) as faces:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hr = hands.process(rgb)
            fr = faces.process(rgb)

            hand_pts, raw = None, "NONE"
            if hr.multi_hand_landmarks:
                hand_pts = hand_points(hr.multi_hand_landmarks[0], w, h)
                raw = classify_gesture(hand_pts)

            face = None
            if fr.detections:
                d = fr.detections[0]
                bb = d.location_data.relative_bounding_box
                x, y = max(int(bb.xmin * w), 0), max(int(bb.ymin * h), 0)
                bw = min(int(bb.width * w), w - x)
                bh = min(int(bb.height * h), h - y)
                kps = [(int(k.x * w), int(k.y * h))
                       for k in d.location_data.relative_keypoints]
                face = {"box": (x, y, bw, bh), "kps": kps}

            now = time.time()
            dt = min(now - prev, 0.1)
            prev = now
            st.fps = 0.9 * st.fps + 0.1 / max(dt, 1e-6)

            out = render(frame, st, hand_pts, raw, face, now - st.t0, dt)

            if st.recording and st.writer is not None:
                st.writer.write(out)

            cv2.imshow("Cyborg AR HUD", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                st.show_help = not st.show_help
            elif key == ord("s"):
                name = f"hud_snapshot_{int(time.time())}.png"
                cv2.imwrite(name, out)
                print("Saved", name)
            elif key == ord("r"):
                if not st.recording:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    st.writer = cv2.VideoWriter("hud_recording.mp4", fourcc, 20.0,
                                                (out.shape[1], out.shape[0]))
                    st.recording = True
                    print("Recording started -> hud_recording.mp4")
                else:
                    st.recording = False
                    st.writer.release()
                    st.writer = None
                    print("Recording saved -> hud_recording.mp4")

    if st.writer is not None:
        st.writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
