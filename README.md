# Cyborg AR HUD - Gesture + Face Computer Vision Project

A real-time webcam project in Python (OpenCV + MediaPipe) that overlays a sci-fi
heads-up display on your face and hands, and reacts to hand gestures.

## Setup

1. Install Python 3.9 - 3.11 (MediaPipe works best on these).
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run:
   ```
   python main.py
   ```
Press **q** to quit.

## Gestures

| Gesture    | Effect                                   |
|------------|------------------------------------------|
| Open palm  | Glowing circuit traces around your hand  |
| Fist       | "Cyborg evolution" progress bar fills up |
| Pinch      | Scanning line sweeps over your face box  |

## How it works (for your report / viva)

1. **Capture** - OpenCV reads frames from the webcam and mirrors them.
2. **Hand tracking** - MediaPipe Hands returns 21 landmarks per hand. They are
   drawn as a skeleton using the known connection pairs.
3. **Gesture classification** - rule based on landmark geometry:
   - A finger is *extended* if its tip is farther from the wrist than its middle joint.
   - *Open palm*: 4 fingers extended. *Fist*: none extended.
   - *Pinch*: thumb-tip to index-tip distance is small relative to hand size.
4. **Face detection** - MediaPipe Face Detection (BlazeFace) gives a bounding box.
   Note: this is *detection*, not identity recognition. The "facial recognition"
   label is only a visual style. If your teacher expects real recognition, see the
   extensions below.
5. **HUD rendering** - OpenCV drawing primitives (lines, circles, text, alpha blending).

## Ideas to extend it (for extra marks)

- Real face recognition with the `face_recognition` library (register known faces,
  show their name in the box).
- More gestures: peace sign, thumbs up, swipe left/right using landmark motion.
- Train a small neural network on landmark data to classify gestures instead of rules.
- Two-hand support (`max_num_hands=2`).
- Record the HUD output to a video file with `cv2.VideoWriter`.

## Troubleshooting

- **Black window / cannot open webcam**: change `cv2.VideoCapture(0)` to `1`.
- **mediapipe install error**: use Python 3.10 or 3.11.
- **Low FPS**: lower the capture resolution in `main.py`.
