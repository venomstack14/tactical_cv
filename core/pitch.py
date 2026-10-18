"""Pitch geometry shared by calibration, rendering and tests.
COORDINATE CONVENTION (meters): x runs along the length (0..105, left->right as seen by the main camera),
y runs across (0..68) starting at the touchline NEAREST the camera and increasing away from it.
The radar draws y upward so it matches the broadcast view."""
import numpy as np
L, W = 105.0, 68.0
def _arc(cx, cy, r, a0, a1, n=48):
    t = np.radians(np.linspace(a0, a1, n)); return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
def polylines():
    """All pitch markings as polylines in meters."""
    P = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float), np.array([[L / 2, 0], [L / 2, W]], float), _arc(L / 2, W / 2, 9.15, 0, 360, 73)]
    a = np.degrees(np.arccos(5.5 / 9.15))                                     # penalty arc is the part outside the box
    for x0, sg in ((0.0, 1), (L, -1)):
        for half in (40.32 / 2, 18.32 / 2):
            d = 16.5 if half > 15 else 5.5
            P.append(np.array([[x0, W / 2 - half], [x0 + sg * d, W / 2 - half], [x0 + sg * d, W / 2 + half], [x0, W / 2 + half]]))
        P.append(_arc(x0 + sg * 11, W / 2, 9.15, -a, a) if sg == 1 else _arc(x0 + sg * 11, W / 2, 9.15, 180 - a, 180 + a))
    return P
# Named landmarks usable in calibration JSON ("name": ...) instead of raw meters.
LANDMARKS = {"center_spot": (52.5, 34), "halfway_near": (52.5, 0), "halfway_far": (52.5, 68), "circle_near": (52.5, 24.85), "circle_far": (52.5, 43.15),
 "left_box_near": (16.5, 13.84), "left_box_far": (16.5, 54.16), "left_6yd_near": (5.5, 24.84), "left_6yd_far": (5.5, 43.16),
 "right_box_near": (88.5, 13.84), "right_box_far": (88.5, 54.16), "right_6yd_near": (99.5, 24.84), "right_6yd_far": (99.5, 43.16),
 "corner_left_near": (0, 0), "corner_left_far": (0, 68), "corner_right_near": (105, 0), "corner_right_far": (105, 68)}
def template_raster(ppm=4, thickness=2):
    """Float image (W*ppm, L*ppm) of the pitch lines, used for line-based refinement."""
    import cv2
    img = np.zeros((int(W * ppm), int(L * ppm)), np.float32)
    for p in polylines(): cv2.polylines(img, [np.round(p * ppm).astype(np.int32)], False, 1.0, thickness)
    return img
