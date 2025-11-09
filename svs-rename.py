import openslide
import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode
import os
from tqdm import tqdm
import shutil
import re
import datetime

# === CONFIGURATION ===
VARIANTS = [
    {"blur": 3, "block": 31, "C": 10},
    {"blur": 5, "block": 25, "C": 5},
    {"blur": 7, "block": 41, "C": 12},
]
BASE_ROTATIONS = [0, 90, 180, 270]
FINE_ROTATION = [-15, -10, -5, 0, 5, 10, 15]
DEBUG = True


# === HELPERS ===
def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9\-_.]", "_", text)


def unique_filename(directory, name, ext=".svs"):
    """Return a unique filename in `directory`, appending _2, _3, etc. if needed."""
    base_name = name
    counter = 2
    full_path = os.path.join(directory, f"{base_name}{ext}")
    while os.path.exists(full_path):
        full_path = os.path.join(directory, f"{base_name}_{counter}{ext}")
        counter += 1
    return full_path


def extract_label_image(svs_path):
    slide = openslide.OpenSlide(svs_path)
    if "label" not in slide.associated_images:
        raise ValueError(f"No 'label' image found in {svs_path}")
    return slide.associated_images["label"]


def preprocess_label(img_pil, blur=3, block=31, C=10, method='adaptive'):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    if method == 'adaptive':
        th = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block, C
        )
    elif method == 'otsu':
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block, C)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    return img, clean


def find_datamatrix_candidates(clean):
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 300000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 0.85 < ar < 1.15:
            candidates.append((x, y, w, h))
    return candidates


def decode_with_full_rotation(crop, debug_path=None, variant_desc=""):
    angles_to_try = []
    for base in BASE_ROTATIONS:
        angles_to_try.extend([base + fine for fine in FINE_ROTATION])
    for angle in angles_to_try:
        (h, w) = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(crop, M, (w, h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        result = decode(rotated)
        if result:
            text = result[0].data.decode("utf-8")
            if DEBUG and debug_path:
                vis = rotated.copy()
                cv2.putText(vis, f"{text} ({variant_desc}, rot={angle})",
                            (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.imwrite(debug_path, vis)
            return text
    return None


def decode_datamatrix(img_bgr, candidates, debug_path=None, variant_desc=""):
    for x, y, w, h in sorted(candidates, key=lambda b: b[2]*b[3], reverse=True):
        pad = 2
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, img_bgr.shape[1]), min(y + h + pad, img_bgr.shape[0])
        crop = img_bgr[y1:y2, x1:x2]
        decoded = decode_with_full_rotation(crop, debug_path, variant_desc)
        if decoded:
            return decoded
    return None


def process_svs_file(svs_path, debug_dir, variants=VARIANTS):
    try:
        label_pil = extract_label_image(svs_path)
        for variant in variants:
            img_bgr, clean = preprocess_label(label_pil, **variant)
            candidates = find_datamatrix_candidates(clean)
            decoded = decode_datamatrix(
                img_bgr, candidates,
                debug_path=os.path.join(debug_dir, os.path.basename(svs_path) + "_debug.png"),
                variant_desc=f"b={variant['blur']}, blk={variant['block']}, C={variant['C']}"
            )
            if decoded:
                return decoded
        return None
    except Exception:
        return None


# === SECOND PASS METHODS WITH LOGGING & CANDIDATE SAVING ===
def second_pass_failed_scans(failed_files, input_dir):
    second_debug_dir = os.path.join(input_dir, "_debug_second_pass")
    os.makedirs(second_debug_dir, exist_ok=True)

    logs_dir = os.path.join(input_dir, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"failed_scans_log_{timestamp}.txt")

    candidates_dir = os.path.join(input_dir, "_candidates")
    os.makedirs(candidates_dir, exist_ok=True)

    log_lines = []

    robust_variants = [
        {"blur": 3, "block": 21, "C": 5, "method": "otsu"},
        {"blur": 5, "block": 31, "C": 10, "method": "gaussian"}
    ]

    recovered_map = {}

    with tqdm(total=len(failed_files), desc="Second Pass", ncols=100) as pbar:
        for f in failed_files:
            path = os.path.join(input_dir, f)
            try:
                label_pil = extract_label_image(path)
                decoded = None
                all_candidates = []
                for var in robust_variants:
                    img_bgr, clean = preprocess_label(label_pil, **var)
                    candidates = find_datamatrix_candidates(clean)
                    all_candidates.extend(candidates)
                    candidates_scaled = []
                    for x, y, w, h in candidates:
                        for scale in [0.9, 1.0, 1.1]:
                            nw, nh = int(w*scale), int(h*scale)
                            nx, ny = x - (nw-w)//2, y - (nh-h)//2
                            candidates_scaled.append((nx, ny, nw, nh))
                    decoded = decode_datamatrix(
                        img_bgr, candidates_scaled,
                        debug_path=os.path.join(second_debug_dir, f + "_debug.png"),
                        variant_desc=f"{var}"
                    )
                    if decoded:
                        recovered_map[f] = decoded
                        break

                # Save all candidate crops
                for idx, (x, y, w, h) in enumerate(all_candidates):
                    pad = 2
                    x1, y1 = max(x - pad, 0), max(y - pad, 0)
                    x2, y2 = min(x + w + pad, img_bgr.shape[1]), min(y + h + pad, img_bgr.shape[0])
                    crop = img_bgr[y1:y2, x1:x2]
                    crop_path = os.path.join(candidates_dir, f"{f}_cand{idx+1}.png")
                    cv2.imwrite(crop_path, crop)

                # Log candidates
                if not decoded and all_candidates:
                    log_lines.append(f"{f} | Candidates saved: {candidates_dir} | Variants: {robust_variants}")
                elif not decoded:
                    log_lines.append(f"{f} | No candidates found | Variants: {robust_variants}")

            except Exception as e:
                log_lines.append(f"{f} | ERROR: {str(e)}")

            pbar.update(1)

    if log_lines:
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines))

    print(f"\nSecond pass finished. Recovered {len(recovered_map)}/{len(failed_files)}.")
    print(f"Debug images saved in: {second_debug_dir}")
    print(f"Candidate crops saved in: {candidates_dir}")
    print(f"Log file for manual review: {log_path}")

    return recovered_map


# === MAIN ENTRY ===
def main():
    input_dir = input("Enter folder with .SVS files: ").strip()
    if not os.path.isdir(input_dir):
        print("❌ Invalid folder.")
        return

    debug_dir = os.path.join(input_dir, "_debug")
    os.makedirs(debug_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".svs")]
    total = len(files)
    if not total:
        print("No .SVS files found.")
        return

    success, errors = 0, 0
    decoded_map = {}
    failed_files = []

    print(f"\nFound {total} SVS files. Processing...\n")
    with tqdm(total=total, desc="First Pass", ncols=100) as pbar:
        for f in files:
            path = os.path.join(input_dir, f)
            result = process_svs_file(path, debug_dir)
            if result:
                success += 1
                decoded_map[f] = result
            else:
                errors += 1
                failed_files.append(f)
            pbar.set_postfix({"ok": success, "err": errors})
            pbar.update(1)

    print(f"\n✅ First pass finished: {success}/{total} decoded successfully ({errors} failed).")

    if decoded_map:
        rename = input("Rename successfully decoded SVS files with decoded text? (y/n): ").strip().lower()
        if rename == "y":
            for old_name, decoded_text in decoded_map.items():
                old_path = os.path.join(input_dir, old_name)
                new_name_safe = safe_filename(decoded_text)
                new_path = unique_filename(input_dir, new_name_safe)
                shutil.move(old_path, new_path)
            print("✅ First-pass renaming complete.")

    if failed_files:
        retry = input("Try second-pass scan for failed files? (y/n): ").strip().lower()
        if retry == "y":
            recovered = second_pass_failed_scans(failed_files, input_dir)
            if recovered:
                rename_retry = input("Rename recovered SVS files from second pass? (y/n): ").strip().lower()
                if rename_retry == "y":
                    for old_name, decoded_text in recovered.items():
                        old_path = os.path.join(input_dir, old_name)
                        new_name_safe = safe_filename(decoded_text)
                        new_path = unique_filename(input_dir, new_name_safe)
                        shutil.move(old_path, new_path)
                    print("✅ Second-pass renaming complete.")


if __name__ == "__main__":
    main()