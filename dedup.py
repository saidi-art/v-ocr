"""Deduplicator: pHash filtering + vertical scroll detection -> minimal image strips."""
import io
import numpy as np
import imagehash
from PIL import Image

HASH_BITS = 64
SIM_THRESHOLD = 0.95          # >95% similar -> discard
MIN_STRIP_FRAC = 0.30         # emit a strip once this much new content accumulated
OVERLAP_PX = 36               # context overlap so lines cut at the seam are readable


def similarity(h1, h2):
    return 1.0 - (h1 - h2) / HASH_BITS


def detect_scroll(prev_gray, cur_gray):
    """Return dy>0 if content moved UP by dy px (scrolling down), else 0.
    cur[r] == prev[r+dy] over a middle band (ignores fixed header/footer)."""
    h = cur_gray.shape[0]
    top, bot = int(h * 0.25), int(h * 0.75)
    band = cur_gray[top:bot]
    best, best_dy = 1e9, 0
    base = np.abs(band - prev_gray[top:bot]).mean()
    maxdy = int(h * 0.7)
    for dy in range(2, maxdy):
        if bot + dy > h:
            break
        err = np.abs(band - prev_gray[top + dy:bot + dy]).mean()
        if err < best:
            best, best_dy = err, dy
    # accept only a clear, good match that beats "no motion"
    if best < 6.0 and best < base * 0.6:
        return best_dy
    return 0


def footer_height(prev_gray, cur_gray):
    """Rows at the bottom identical between frames = fixed footer (input bar, nav bar)."""
    h = cur_gray.shape[0]
    rows = np.abs(cur_gray - prev_gray).mean(axis=1)
    n = 0
    while n < int(h * 0.25) and rows[h - 1 - n] < 1.0:
        n += 1
    return n


def gray(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def build_strips(frames):
    """frames: list of (t_seconds, PIL.Image). Returns (strips, stats).
    strips: list of (t, PIL.Image). stats: dict."""
    stats = {"frames_total": 0, "frames_dup": 0, "frames_sent_full": 0, "frames_cropped": 0}
    strips = []
    prev_hash = None
    prev_gray = None
    vb = None
    acc = 0  # accumulated unseen px since last emitted strip
    last_img = None
    last_t = 0
    for t, img in frames:
        stats["frames_total"] += 1
        hsh = imagehash.phash(img)
        g = gray(img)
        # pHash says "same", and a cheap pixel check confirms nothing scrolled (pHash alone
        # can't tell a 100px scroll of similar-looking text from a static screen)
        if (prev_hash is not None and similarity(hsh, prev_hash) > SIM_THRESHOLD
                and np.abs(g - prev_gray).mean() < 2.0):
            stats["frames_dup"] += 1
            continue
        if prev_gray is None:
            strips.append((t, img)); stats["frames_sent_full"] += 1
        else:
            dy = detect_scroll(prev_gray, g)
            if dy == 0:
                # unknown change (jump, new screen): flush pending, send full frame
                if acc > 0 and last_img is not None:
                    pass
                strips.append((t, img)); stats["frames_sent_full"] += 1; acc = 0
            else:
                acc += dy
                vb = img.height - footer_height(prev_gray, g)
                if acc >= MIN_STRIP_FRAC * img.height:
                    top = max(0, vb - acc - OVERLAP_PX)
                    strips.append((t, img.crop((0, top, img.width, vb))))
                    stats["frames_cropped"] += 1
                    acc = 0
                else:
                    stats["frames_dup"] += 1  # accumulate; counted as skipped
        prev_hash, prev_gray, last_img, last_t = hsh, g, img, t
    # flush remaining unseen area at end
    if acc > 0 and last_img is not None:
        top = max(0, (vb or last_img.height) - acc - OVERLAP_PX)
        strips.append((last_t, last_img.crop((0, top, last_img.width, vb or last_img.height))))
        stats["frames_cropped"] += 1
    stats["strips"] = len(strips)
    return strips, stats
