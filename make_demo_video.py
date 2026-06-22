"""
Generates a realistic demo video with multiple vehicles and license plates
visible in various conditions (day, angle, partial occlusion, night scene).
Saves to demo_plates.mp4 in the project folder.
"""
import cv2
import numpy as np
import math
import random

W, H = 1280, 720
FPS = 30 
DURATION_SECS = 20   # total video length
OUT = "demo_plates.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))

# Plates: (text, authorized)
PLATES = [
    ("KL07CK4521", True),
    ("TN38BA1190", True),
    ("KL11AX8830", True),
    ("MH12EF9012", True),
    ("KA09XX9999", False),   # unknown vehicle
    ("DL01AB5678", False),   # unknown vehicle
]

rng = random.Random(42)


# ─── helpers ───────────────────────────────────────────────────────────────

def draw_sky(frame, night=False):
    if night:
        frame[:H//2] = (15, 10, 8)
        # stars
        for _ in range(80):
            x, y = rng.randint(0, W-1), rng.randint(0, H//2-1)
            cv2.circle(frame, (x, y), 1, (220, 220, 220), -1)
    else:
        frame[:H//2] = (180, 140, 90)
        cv2.rectangle(frame, (0, 0), (W, H//2), (180, 140, 90), -1)


def draw_road(frame):
    pts = np.array([[0, H], [W//2-80, H//2], [W//2+80, H//2], [W, H]], np.int32)
    cv2.fillPoly(frame, [pts], (55, 55, 55))
    # lane markings
    for i in range(5):
        t = i / 4
        x1 = int(W//2 - 10 + t*50)
        y1 = int(H//2 + t*(H - H//2))
        x2 = int(x1 + 40)
        y2 = int(y1 + 30)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 60), -1)


def draw_plate_on_vehicle(frame, plate_text, cx, cy, scale=1.0,
                           angle_deg=0, brightness=1.0, night=False):
    """Draw a vehicle silhouette with a clearly readable licence plate."""
    # -- vehicle body --
    bw = int(200 * scale)
    bh = int(100 * scale)
    x0, y0 = cx - bw//2, cy - bh//2
    col_body = tuple(int(c * brightness) for c in (90, 90, 200))
    col_window = tuple(int(c * brightness) for c in (160, 200, 220))
    cv2.rectangle(frame, (x0, y0), (x0+bw, y0+bh), col_body, -1)
    # windows
    wx = x0 + int(30*scale)
    ww = int(140*scale)
    wh = int(40*scale)
    cv2.rectangle(frame, (wx, y0-wh+5), (wx+ww, y0+5), col_window, -1)
    # wheels
    for wx2 in [x0 + int(35*scale), x0 + bw - int(35*scale)]:
        cv2.ellipse(frame, (wx2, y0+bh), (int(22*scale), int(12*scale)), 0, 0, 360,
                    (20, 20, 20), -1)

    # -- plate background --
    pw = int(120 * scale)
    ph = int(30 * scale)
    px = cx - pw//2
    py = y0 + bh - ph - int(4*scale)

    # White plate
    plate_brightness = min(255, int(255 * brightness * (1.8 if night else 1.0)))
    cv2.rectangle(frame, (px, py), (px+pw, py+ph),
                  (plate_brightness, plate_brightness, plate_brightness), -1)
    cv2.rectangle(frame, (px, py), (px+pw, py+ph), (0, 0, 0), 1)

    # -- plate text (black on white) --
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45 * scale
    thickness = max(1, int(1.5 * scale))
    txt_col = (0, 0, 0)

    (tw, th), _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    tx = px + (pw - tw) // 2
    ty = py + (ph + th) // 2 - 2
    cv2.putText(frame, plate_text, (tx, ty), font, font_scale, txt_col, thickness, cv2.LINE_AA)

    # night headlights
    if night:
        for lx in [x0 + int(10*scale), x0 + bw - int(10*scale)]:
            cv2.circle(frame, (lx, y0 + bh//2), int(8*scale), (240, 240, 180), -1)
            # cone
            pts = np.array([
                [lx, y0 + bh//2 - int(6*scale)],
                [lx, y0 + bh//2 + int(6*scale)],
                [lx + int(120*scale), y0 + bh//2 + int(30*scale)],
                [lx + int(120*scale), y0 + bh//2 - int(30*scale)],
            ], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (220, 220, 130))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)


def draw_hud(frame, frame_no, label):
    ts = f"{frame_no // (FPS*3600):02d}:{(frame_no//(FPS*60))%60:02d}:{(frame_no//FPS)%60:02d}"
    cv2.rectangle(frame, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(frame, f"SENTINEL · CAM-01 · MAIN GATE IN · {ts}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, "● REC", (W-70, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, label, (10, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


# ─── scenes ────────────────────────────────────────────────────────────────

total_frames = DURATION_SECS * FPS

for f in range(total_frames):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    t = f / FPS  # seconds

    # Scene 1 (0-5s): two authorised vehicles approaching, daytime
    if t < 5:
        draw_sky(frame, night=False)
        draw_road(frame)
        progress = t / 5.0
        # vehicle 1 — KL07CK4521
        cx1 = int(W//2 - 60 + progress * 60)
        cy1 = int(H//2 + 60 + progress * 100)
        sc1 = 0.7 + progress * 0.6
        draw_plate_on_vehicle(frame, "KL07CK4521", cx1, cy1, scale=sc1)
        # vehicle 2 in background
        draw_plate_on_vehicle(frame, "TN38BA1190", W//2+150, H//2+40, scale=0.55)
        draw_hud(frame, f, "Scene 1 · Daytime gate · Multiple authorised vehicles")

    # Scene 2 (5-10s): unknown vehicle + close-up plate
    elif t < 10:
        progress = (t - 5) / 5.0
        draw_sky(frame, night=False)
        draw_road(frame)
        cx = int(W//2 - 80 + progress * 200)
        cy = int(H//2 + 40 + progress * 130)
        sc = 0.6 + progress * 0.8
        draw_plate_on_vehicle(frame, "KA09XX9999", cx, cy, scale=sc, brightness=0.95)
        # second authorised vehicle parked
        draw_plate_on_vehicle(frame, "KL11AX8830", W - 260, H - 160, scale=0.9)
        draw_hud(frame, f, "Scene 2 · Unknown vehicle KA09XX9999 + authorised KL11AX8830")

    # Scene 3 (10-15s): night scene with headlights
    elif t < 15:
        progress = (t - 10) / 5.0
        draw_sky(frame, night=True)
        # dark road
        pts = np.array([[0, H], [W//2-60, H//2], [W//2+60, H//2], [W, H]], np.int32)
        cv2.fillPoly(frame, [pts], (30, 30, 30))
        cx = int(W//2 + 50 - progress * 180)
        cy = int(H//2 + 30 + progress * 150)
        sc = 0.55 + progress * 0.9
        draw_plate_on_vehicle(frame, "MH12EF9012", cx, cy, scale=sc,
                               brightness=0.75, night=True)
        draw_plate_on_vehicle(frame, "DL01AB5678", W - 200, H - 200, scale=0.7,
                               brightness=0.55, night=True)
        draw_hud(frame, f, "Scene 3 · Night scene · Headlights · Low light conditions")

    # Scene 4 (15-20s): fast pan with multiple plates simultaneously
    else:
        progress = (t - 15) / 5.0
        draw_sky(frame, night=False)
        draw_road(frame)
        # 3 vehicles in one frame
        draw_plate_on_vehicle(frame, "KL07CK4521",
                               int(W//4 + progress*30), H//2+100, scale=0.85)
        draw_plate_on_vehicle(frame, "TN38BA1190",
                               int(W//2 + progress*20), H//2+90, scale=0.78)
        draw_plate_on_vehicle(frame, "KA09XX9999",
                               int(3*W//4 - progress*25), H//2+80, scale=0.72)
        draw_hud(frame, f, "Scene 4 · Multi-vehicle frame · Mixed authorised + unknown")

    out.write(frame)

out.release()
print(f"Demo video saved: {OUT}  ({total_frames} frames, {DURATION_SECS}s @ {FPS}fps)")
