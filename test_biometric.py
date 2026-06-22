"""
Biometric face auth end-to-end test.
Steps:
  1. Capture two frames from webcam (or fall back to synthetic face)
  2. Register person via POST /api/v1/biometric/register
  3. Verify with the SAME photo  -> expect GRANTED
  4. Verify with a DIFFERENT random image -> expect DENIED
  5. Check biometric log via GET /api/v1/biometric/log
"""
import sys, time, cv2, numpy as np, urllib.request, json

API = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# Helper: multipart POST via urllib (no requests library needed)
# ---------------------------------------------------------------------------

def _multipart_post(url, fields, files):
    boundary = "----TestBoundary7890"
    body_parts = []
    for name, value in fields.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}'.encode()
        )
    for name, (filename, data, mime) in files.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode()
            + data
        )
    body = b"\r\n".join(body_parts) + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(url):
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def _delete(url):
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Step 1: Acquire two face images
# ---------------------------------------------------------------------------

def capture_from_webcam():
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None, None

    # Warm up
    for _ in range(10):
        cap.read()

    ret1, frame1 = cap.read()
    time.sleep(0.5)
    ret2, frame2 = cap.read()
    cap.release()

    if ret1 and ret2:
        print("  Webcam OK - captured 2 frames")
        return frame1, frame2
    return None, None


def make_synthetic_face(seed=0):
    """Create a realistic face-like image that passes Haar cascade detection."""
    np.random.seed(seed)
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Skin-toned background ellipse (face)
    cv2.ellipse(img, (150, 155), (95, 115), 0, 0, 360, (180, 150, 120), -1)

    # Eyes
    for ex in [110, 190]:
        cv2.ellipse(img, (ex, 130), (20, 12), 0, 0, 360, (240, 220, 200), -1)
        cv2.circle(img, (ex, 130), 9, (40, 30, 20), -1)
        cv2.circle(img, (ex, 130), 4, (10, 10, 10), -1)
        cv2.circle(img, (ex - 3, 127), 2, (255, 255, 255), -1)

    # Eyebrows
    cv2.ellipse(img, (110, 110), (22, 7), 0, 0, 180, (80, 55, 30), -1)
    cv2.ellipse(img, (190, 110), (22, 7), 0, 0, 180, (80, 55, 30), -1)

    # Nose
    pts = np.array([[150, 155], [138, 185], [162, 185]], dtype=np.int32)
    cv2.polylines(img, [pts], False, (140, 110, 90), 2)

    # Mouth
    cv2.ellipse(img, (150, 210), (28, 12), 0, 0, 180, (110, 70, 70), -1)
    cv2.ellipse(img, (150, 200), (28, 8), 0, 0, 180, (200, 130, 120), 2)

    # Ears
    cv2.ellipse(img, (55, 155), (15, 25), 0, 0, 360, (170, 140, 110), -1)
    cv2.ellipse(img, (245, 155), (15, 25), 0, 0, 360, (170, 140, 110), -1)

    # Hair
    cv2.ellipse(img, (150, 90), (100, 70), 0, 180, 360, (50, 35, 20), -1)

    # Add slight noise for realism
    noise = np.random.randint(-8, 8, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


print("=" * 60)
print("  Biometric Face Auth — End-to-End Test")
print("=" * 60)

frame1, frame2 = capture_from_webcam()

if frame1 is None:
    print("  No webcam found — using synthetic face images")
    frame1 = make_synthetic_face(seed=1)
    frame2 = make_synthetic_face(seed=1)   # same seed = same person
    frame_unknown = make_synthetic_face(seed=99)  # different seed = different person
    webcam = False
else:
    frame_unknown = make_synthetic_face(seed=99)
    webcam = True

# Encode to JPEG bytes
def to_jpg(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return bytes(buf)

jpg1 = to_jpg(frame1)
jpg2 = to_jpg(frame2)
jpg_unknown = to_jpg(frame_unknown)

cv2.imwrite("bio_register.jpg", frame1)
cv2.imwrite("bio_verify.jpg", frame2)
cv2.imwrite("bio_unknown.jpg", frame_unknown)
print(f"\n  Source: {'webcam' if webcam else 'synthetic'}")
print(f"  register image : bio_register.jpg  ({len(jpg1)//1024}KB)")
print(f"  verify  image  : bio_verify.jpg    ({len(jpg2)//1024}KB)")
print(f"  unknown image  : bio_unknown.jpg   ({len(jpg_unknown)//1024}KB)")

# ---------------------------------------------------------------------------
# Step 2: Register
# ---------------------------------------------------------------------------

print("\n[1] Registering person: Nihal Riyas (EMP-TEST-01) ...")
try:
    reg = _multipart_post(
        f"{API}/api/v1/biometric/register",
        fields={
            "name": "Nihal Riyas",
            "employee_id": "EMP-TEST-01",
            "department": "Engineering",
            "clearance_level": "L3",
        },
        files={"photo": ("bio_register.jpg", jpg1, "image/jpeg")},
    )
    print(f"  ok={reg['ok']}  face_detected={reg['face_detected']}")
    print(f"  Person record: {reg['person']['name']} / {reg['person']['employee_id']}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3: Verify with same person's photo -> GRANTED
# ---------------------------------------------------------------------------

print("\n[2] Verifying with SAME person's photo ...")
print("    (DeepFace Facenet model may download ~90MB on first run — please wait)")
t0 = time.time()
try:
    v1 = _multipart_post(
        f"{API}/api/v1/biometric/verify",
        fields={},
        files={"photo": ("bio_verify.jpg", jpg2, "image/jpeg")},
    )
    elapsed = time.time() - t0
    print(f"  Decision  : {v1['decision']}  ({elapsed:.1f}s)")
    print(f"  Confidence: {v1['confidence']}")
    print(f"  Engine    : {v1['engine']}")
    if v1['person']:
        print(f"  Matched   : {v1['person']['name']} ({v1['person']['department']})")
    result1 = "PASS" if v1['decision'] == "GRANTED" else "FAIL (expected GRANTED)"
    print(f"  Test      : {result1}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# Step 4: Verify with unknown face -> DENIED
# ---------------------------------------------------------------------------

print("\n[3] Verifying with UNKNOWN face ...")
t0 = time.time()
try:
    v2 = _multipart_post(
        f"{API}/api/v1/biometric/verify",
        fields={},
        files={"photo": ("bio_unknown.jpg", jpg_unknown, "image/jpeg")},
    )
    elapsed = time.time() - t0
    print(f"  Decision  : {v2['decision']}  ({elapsed:.1f}s)")
    print(f"  Confidence: {v2['confidence']}")
    result2 = "PASS" if v2['decision'] == "DENIED" else "FAIL (expected DENIED)"
    print(f"  Test      : {result2}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# Step 5: Check log
# ---------------------------------------------------------------------------

print("\n[4] Biometric log (last 5 entries):")
try:
    log = _get(f"{API}/api/v1/biometric/log")
    for entry in log[:5]:
        print(f"  [{entry['id']}] {entry['person_name']:20s}  conf={entry['confidence']}  {entry['decision']}  {entry['logged_at']}")
except Exception as e:
    print(f"  ERROR: {e}")

# ---------------------------------------------------------------------------
# Step 6: List registered persons
# ---------------------------------------------------------------------------

print("\n[5] Registered persons:")
try:
    persons = _get(f"{API}/api/v1/biometric/persons")
    for p in persons:
        print(f"  {p['employee_id']:15s}  {p['name']:20s}  {p['department']:15s}  {p['clearance_level']}  photo={'yes' if p['has_photo'] else 'no'}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("  Test complete.")
print("=" * 60)
