import time
import cv2
import numpy as np
import mediapipe as mp
from pythonosc.udp_client import SimpleUDPClient

# ------------------ CONFIG ------------------
OSC_IP = "127.0.0.1"   # Max on same machine
OSC_PORT = 7400        # must match [udpreceive 7400] in Max
CAM_INDEX = 0          # change if neededd
# Ranges (match your Max controls)
SIZE_RANGE   = (10.0, 100.0)   # "size" dial range
DECAY_RANGE  = (0.80, 0.98)    # decay number box range

# Response shaping & stabilization
SMOOTH_ALPHA  = 0.12   # 0..1 (lower = smoother). Try 0.08–0.20
DEADZONE_NORM = 0.02   # fraction of control range to ignore (2%)
STEP_SIZE     = 0.5    # snap size in 0.5 steps
STEP_DECAY    = 0.003  # snap decay in 0.003 steps
MAX_UPS       = 30     # max updates per second
GAMMA_X       = 0.9    # <1 = gentler mid response for size
GAMMA_Y       = 0.9    # <1 = gentler mid response for decay
INVERT_Y      = True   # hand up -> higher decay
DRAW_PREVIEW  = True   # show camera window
# --------------------------------------------

def lerp(a, b, t): return a + (b - a) * t   #interpolare , a=start, b=final
def clamp01(x): return max(0.0, min(1.0, x))
def ema(prev, new, a): return a * new + (1 - a) * prev if prev is not None else new
def resp_curve(t, gamma): return clamp01(t) ** gamma
def quantize(v, step): return round(v / step) * step

def within_deadzone(prev, cur, rmin, rmax, dz_frac):
    if prev is None: return False
    return abs(cur - prev) < dz_frac * (rmax - rmin)

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,            
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    client = SimpleUDPClient(OSC_IP, OSC_PORT)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera; try CAM_INDEX=1.")

    sm_size = None
    sm_decay = None
    last_tx_time = 0.0
    last_sent_size = None
    last_sent_decay = None

    print("Running. Press 'q' in the preview window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                xs = [p.x for p in lm]
                ys = [p.y for p in lm]
                cx = float(np.mean(xs))  # 0..1 left->right
                cy = float(np.mean(ys))  # 0..1 top->bottom

                # Normalize & shape
                nx = resp_curve(cx, GAMMA_X)
                ny = resp_curve(1.0 - cy if INVERT_Y else cy, GAMMA_Y)

                # Map to control ranges
                size_target  = lerp(SIZE_RANGE[0],  SIZE_RANGE[1],  nx)
                decay_target = lerp(DECAY_RANGE[0], DECAY_RANGE[1], ny)

                # Smooth
                sm_size  = ema(sm_size,  size_target,  SMOOTH_ALPHA)
                sm_decay = ema(sm_decay, decay_target, SMOOTH_ALPHA)

                # Quantize
                cand_size  = quantize(sm_size,  STEP_SIZE)
                cand_decay = quantize(sm_decay, STEP_DECAY)

                # Rate limit + dead-zone vs last *sent* values
                now = time.time()
                if now - last_tx_time >= 1.0 / MAX_UPS:
                    send_size  = not within_deadzone(last_sent_size,  cand_size,
                                                     *SIZE_RANGE,  DEADZONE_NORM)
                    send_decay = not within_deadzone(last_sent_decay, cand_decay,
                                                     *DECAY_RANGE, DEADZONE_NORM)

                    if send_size:
                        client.send_message("/size", float(cand_size))
                        last_sent_size = cand_size
                    if send_decay:
                        client.send_message("/decay", float(cand_decay))
                        last_sent_decay = cand_decay

                    if send_size or send_decay:
                        last_tx_time = now
                        print(f"TX /size {last_sent_size:.2f}  /decay {last_sent_decay:.3f}")

                if DRAW_PREVIEW:
                    cx_px, cy_px = int(cx * w), int(cy * h)
                    cv2.circle(frame, (cx_px, cy_px), 12, (0, 255, 0), 2)
                    cv2.putText(frame, f"size:  {cand_size:6.1f}",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.putText(frame, f"decay: {cand_decay:6.3f}",
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if DRAW_PREVIEW:
                cv2.imshow("Hand -> Max (/size, /decay)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        if DRAW_PREVIEW:
            cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
