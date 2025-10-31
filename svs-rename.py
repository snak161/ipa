import openslide
import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os
import shutil
from tqdm import tqdm

# ----------------- CONFIG -----------------
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
MIN_SIZE = 200          # minimum width/height for decoding
DEBUG = True            # save crops for failed decodes
# -----------------------------------------

# --- Decode helper ---
def attempt_decode(img):
    try:
        results = decode(img)
        if results:
            return results[0].data.decode("utf-8").strip()
    except:
        pass
    return None

# --- Decode crop with fast variants and auto-resize ---
def decode_variants(crop):
    h, w = crop.shape[:2]
    if min(h, w) < MIN_SIZE:
        scale = MIN_SIZE / min(h, w)
        crop = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    variants = [
        gray,
        cv2.bitwise_not(gray),
        cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    ]
    for v in variants:
        text = attempt_decode(v)
        if text:
            return text
    return None

# --- Detect candidate contours (all plausible squares/rects) ---
def detect_candidate_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3),0)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,35,10)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:  # ignore tiny noise
            continue
        ar = w/float(h)
        if 0.5 < ar < 2.0:    # allow slight rectangles
            candidates.append(cnt)

    return candidates

# --- Crop from min-area rect with padding ---
def crop_from_rect(img, rect, pad_ratio=0.15):
    box = cv2.boxPoints(rect).astype(np.intp)
    width = int(rect[1][0])
    height = int(rect[1][1])

    pad_w = int(width*pad_ratio)
    pad_h = int(height*pad_ratio)
    width += 2*pad_w
    height += 2*pad_h

    src_pts = box.astype("float32")
    dst_pts = np.array([[0,height-1],[0,0],[width-1,0],[width-1,height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    warped = cv2.warpPerspective(img,M,(width,height))

    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    return warped

# --- Process single SVS file ---
def process_svs_file(svs_path, debug_dir=None):
    try:
        slide = openslide.OpenSlide(svs_path)
    except:
        return None

    if "label" not in slide.associated_images:
        return None

    label_img = slide.associated_images["label"]
    img = cv2.cvtColor(np.array(label_img), cv2.COLOR_RGB2BGR)

    contours = detect_candidate_contours(img)

    # Try all candidate contours for decoding
    for idx, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)

        # Default padding
        crop = crop_from_rect(img, rect, pad_ratio=0.15)
        text = decode_variants(crop)
        if text:
            return text

        # Retry larger padding
        crop = crop_from_rect(img, rect, pad_ratio=0.30)
        text = decode_variants(crop)
        if text:
            return text

    # Save debug crops if decoding failed
    if DEBUG and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for idx, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            for pad in [0.15,0.30]:
                crop = crop_from_rect(img, rect, pad_ratio=pad)
                base_name = os.path.splitext(os.path.basename(svs_path))[0]
                crop_name = f"{base_name}_cand{idx}_pad{int(pad*100)}.png"
                cv2.imwrite(os.path.join(debug_dir, crop_name), crop)

    return None

# --- Main batch ---
if __name__ == "__main__":
    input_dir = input("Enter folder containing .SVS files: ").strip()
    if not os.path.isdir(input_dir):
        print("Invalid folder.")
        exit(1)

    debug_dir = os.path.join(input_dir, "_debug_crops") if DEBUG else None
    svs_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".svs")]
    total = len(svs_files)
    print(f"\nFound {total} SVS files.")

    if input("Proceed to decode and rename? (y/n): ").strip().lower() != "y":
        exit(0)

    decoded_count = 0
    rename_plan = []

    for svs in tqdm(svs_files, desc="Decoding SVS files"):
        full_path = os.path.join(input_dir, svs)
        text = process_svs_file(full_path, debug_dir=debug_dir)
        if text:
            decoded_count += 1
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
            new_path = os.path.join(input_dir,f"{safe_name}.svs")
            rename_plan.append((full_path,new_path))

    print(f"\nDecoded {decoded_count}/{total} ({decoded_count/total*100:.1f}%)")
    print(f"{len(rename_plan)} files will be renamed.")

    if rename_plan and input("Confirm rename? (y/n): ").strip().lower() == "y":
        renamed = 0
        for old,new in rename_plan:
            if not os.path.exists(new):
                shutil.move(old,new)
                renamed += 1
        print(f"Renamed {renamed}/{len(rename_plan)} files.")

    if DEBUG:
        print(f"Debug crops saved in: {debug_dir}")