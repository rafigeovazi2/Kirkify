import cv2
import numpy as np
import time
import sys
import os
import urllib.request

# ============================================================
# MediaPipe - supports both old (0.9.x) and new (0.10.x+) API
# ============================================================
try:
    # New Tasks API (mediapipe >= 0.10.0)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    USE_TASKS_API = True
    print("[INFO] Using MediaPipe Tasks API (0.10.x+)")
except (ImportError, AttributeError):
    # Legacy solutions API (mediapipe 0.9.x)
    try:
        import mediapipe as mp
        _face_mesh_module = mp.solutions.face_mesh
        USE_TASKS_API = False
        print("[INFO] Using MediaPipe Solutions API (legacy)")
    except Exception as e:
        print(f"[ERROR] Cannot load MediaPipe: {e}")
        print("  Run: pip install mediapipe==0.10.9")
        sys.exit(1)

# =============
# Configuration
# =============
FADE_DURATION   = 5.0        # seconds for fading transition
KIRK_IMAGE_PATH = "kirk.jpg"
MODEL_PATH      = "face_landmarker.task"
MODEL_URL       = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Landmark indices (valid for both 468 and 478 landmark models)
SELECTED_LANDMARKS = [
    # Face oval / jawline
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    # Eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # Eyes
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # Nose
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327,
    # Mouth outer
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    # Forehead
    151, 9, 8, 55, 285,
]

# Face oval for mask
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


# ==============
# Model Download
# ==============
def ensure_model(script_dir):
    model_path = os.path.join(script_dir, MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"[INFO] Downloading face landmark model (~29MB)...")
        print(f"       {MODEL_URL}")
        try:
            def progress(count, block_size, total_size):
                pct = int(count * block_size * 100 / total_size)
                print(f"\r       {min(pct,100)}%", end="", flush=True)
            urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=progress)
            print("\n[INFO] Model downloaded successfully.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download model: {e}")
            print("  Download manually from:")
            print(f"  {MODEL_URL}")
            print(f"  and place it as: {model_path}")
            sys.exit(1)
    return model_path


# ===========================
# Landmark Detection Wrappers
# ===========================
class FaceDetectorTasksAPI:
    """Wrapper for MediaPipe Tasks API (0.10.x+)."""

    def __init__(self, model_path, static_mode=False):
        if static_mode:
            running_mode = mp_vision.RunningMode.IMAGE
        else:
            running_mode = mp_vision.RunningMode.VIDEO
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(opts)
        self.static_mode = static_mode
        self._ts = 0

    def get_landmarks(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        h, w = bgr_image.shape[:2]
        if self.static_mode:
            result = self.detector.detect(mp_img)
        else:
            self._ts += 33  # ~30fps timestamps
            result = self.detector.detect_for_video(mp_img, self._ts)
        if not result.face_landmarks:
            return None
        return [
            (int(lm.x * w), int(lm.y * h))
            for lm in result.face_landmarks[0]
        ]

    def close(self):
        self.detector.close()


class FaceDetectorLegacyAPI:
    """Wrapper for MediaPipe Solutions API (0.9.x legacy)."""

    def __init__(self, static_mode=False):
        self.mesh = _face_mesh_module.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def get_landmarks(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w = bgr_image.shape[:2]
        return [
            (int(lm.x * w), int(lm.y * h))
            for lm in results.multi_face_landmarks[0].landmark
        ]

    def close(self):
        self.mesh.close()


def make_detector(model_path, static_mode):
    if USE_TASKS_API:
        return FaceDetectorTasksAPI(model_path, static_mode)
    else:
        return FaceDetectorLegacyAPI(static_mode)


# ========================
# Image Processing Helpers
# ========================
def selected_points(all_landmarks, indices):
    return [all_landmarks[i] for i in indices if i < len(all_landmarks)]


def compute_delaunay(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    pt_map = {}
    for i, p in enumerate(points):
        px = min(max(p[0], rect[0] + 1), rect[0] + rect[2] - 2)
        py = min(max(p[1], rect[1] + 1), rect[1] + rect[3] - 2)
        key = (px, py)
        pt_map[key] = i
        try:
            subdiv.insert((float(px), float(py)))
        except Exception:
            pass

    tri_indices = []
    for t in subdiv.getTriangleList():
        pts = [
            (int(t[0]), int(t[1])),
            (int(t[2]), int(t[3])),
            (int(t[4]), int(t[5])),
        ]
        idxs = []
        for pt in pts:
            for key, idx in pt_map.items():
                if abs(pt[0] - key[0]) < 2 and abs(pt[1] - key[1]) < 2:
                    idxs.append(idx)
                    break
        if len(idxs) == 3:
            tri_indices.append(tuple(idxs))
    return tri_indices


def warp_triangle(src_img, dst_img, src_tri, dst_tri):
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))
    if r1[2] == 0 or r1[3] == 0 or r2[2] == 0 or r2[3] == 0:
        return

    src_off = [(p[0] - r1[0], p[1] - r1[1]) for p in src_tri]
    dst_off = [(p[0] - r2[0], p[1] - r2[1]) for p in dst_tri]

    src_crop = src_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_crop.size == 0:
        return

    mat = cv2.getAffineTransform(np.float32(src_off), np.float32(dst_off))
    warped = cv2.warpAffine(
        src_crop, mat, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_off), 255)

    y1, y2 = r2[1], r2[1] + r2[3]
    x1, x2 = r2[0], r2[0] + r2[2]
    if y2 > dst_img.shape[0] or x2 > dst_img.shape[1] or y1 < 0 or x1 < 0:
        return

    warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
    mask_inv = cv2.bitwise_not(mask)
    bg = cv2.bitwise_and(dst_img[y1:y2, x1:x2], dst_img[y1:y2, x1:x2], mask=mask_inv)
    dst_img[y1:y2, x1:x2] = bg + warped_masked


def create_face_mask(shape, landmarks, blur_size=51, sigma=15):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = np.array(
        [landmarks[i] for i in FACE_OVAL if i < len(landmarks)],
        dtype=np.int32,
    )
    if len(pts) < 3:
        return mask
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), sigma)
    return mask


def color_correct(warped, frame, mask):
    mask_bool = mask > 128
    if not np.any(mask_bool):
        return warped
    result = warped.copy().astype(np.float32)
    frame_f = frame.astype(np.float32)
    for c in range(3):
        src_vals = result[:, :, c][mask_bool]
        tgt_vals = frame_f[:, :, c][mask_bool]
        s_mean = src_vals.mean()
        s_std  = max(src_vals.std(), 1.0)
        t_mean = tgt_vals.mean()
        t_std  = max(tgt_vals.std(), 1.0)
        result[:, :, c][mask_bool] = (
            (result[:, :, c][mask_bool] - s_mean) * (t_std / s_std) + t_mean
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_status(frame, text, color, progress=None):
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                color, 2, cv2.LINE_AA)
    if progress is not None and 0.0 < progress < 1.0:
        bar_w, bar_h = 220, 10
        x0, y0 = 20, 60
        cv2.rectangle(frame, (x0, y0), (x0+bar_w, y0+bar_h), (60, 60, 60), -1)
        fill_w = int(bar_w * progress)
        cv2.rectangle(frame, (x0, y0), (x0+fill_w, y0+bar_h), color, -1)
        cv2.rectangle(frame, (x0, y0), (x0+bar_w, y0+bar_h), (200, 200, 200), 1)


# ====
# Main
# ====
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure model file exists (only needed for Tasks API)
    if USE_TASKS_API:
        model_path = ensure_model(script_dir)
    else:
        model_path = None

    # Load kirk reference image
    kirk_path = os.path.join(script_dir, KIRK_IMAGE_PATH)
    kirk_img = cv2.imread(kirk_path)
    if kirk_img is None:
        print(f"[ERROR] Cannot load '{kirk_path}'")
        sys.exit(1)
    print("[INFO] Loaded kirk.jpg successfully.")

    # Detect landmarks on kirk (static, one-time)
    detector_static = make_detector(model_path, static_mode=True)
    kirk_all = detector_static.get_landmarks(kirk_img)
    detector_static.close()

    if kirk_all is None:
        print("[ERROR] No face detected in kirk.jpg!")
        sys.exit(1)

    kirk_pts = selected_points(kirk_all, SELECTED_LANDMARKS)
    h_k, w_k = kirk_img.shape[:2]
    kirk_tris = compute_delaunay((0, 0, w_k, h_k), kirk_pts)
    print(f"[INFO] Kirk: {len(kirk_pts)} landmarks, {len(kirk_tris)} triangles.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] Camera opened. Press Q/ESC to quit, R to reset.\n")

    # State
    fade_start    = None
    face_detected = False
    fade_done     = False

    detector_video = make_detector(model_path, static_mode=False)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            output = frame.copy()

            user_all = detector_video.get_landmarks(frame)

            if user_all is not None:
                if not face_detected:
                    face_detected = True
                    fade_start    = time.time()
                    print("[INFO] Face detected — fading started!")

                user_pts = selected_points(user_all, SELECTED_LANDMARKS)

                # Warp kirk's triangles onto user's face geometry
                warped = np.zeros_like(frame)
                for tri in kirk_tris:
                    i, j, k = tri
                    if (i >= len(kirk_pts) or j >= len(kirk_pts) or k >= len(kirk_pts) or
                            i >= len(user_pts) or j >= len(user_pts) or k >= len(user_pts)):
                        continue
                    try:
                        warp_triangle(
                            kirk_img, warped,
                            [kirk_pts[i], kirk_pts[j], kirk_pts[k]],
                            [user_pts[i], user_pts[j], user_pts[k]],
                        )
                    except Exception:
                        pass

                # Feathered face mask
                face_mask  = create_face_mask(frame.shape, user_all)
                mask_3f    = cv2.merge([face_mask]*3).astype(np.float32) / 255.0

                # Color-correct warped face
                warped = color_correct(warped, frame, face_mask)

                # Alpha ramp (0 → 1 over FADE_DURATION seconds)
                elapsed   = time.time() - fade_start
                alpha     = min(elapsed / FADE_DURATION, 1.0)
                fade_done = alpha >= 1.0

                blend  = mask_3f * alpha
                output = (
                    frame.astype(np.float32) * (1.0 - blend)
                    + warped.astype(np.float32) * blend
                ).astype(np.uint8)

                if fade_done:
                    draw_status(output, "Kirkified.", (0, 255, 100))
                else:
                    draw_status(output, "Kirkifying..", (0, 230, 255), progress=alpha)

            else:
                if not face_detected:
                    draw_status(output, "Waiting for face...", (100, 100, 255))
                elif fade_done:
                    # Keep showing last status even if face lost briefly
                    draw_status(output, "Kirkified.", (0, 255, 100))

            cv2.imshow("WE ARE CHARLIE KIRK", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("r"):
                fade_start    = None
                face_detected = False
                fade_done     = False
                print("[INFO] Reset!")

    finally:
        detector_video.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
