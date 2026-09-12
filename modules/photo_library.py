"""Face-grouped photo library: scan a folder of stills, group them by person.

The video side of this app already knows how to recognise a face —
``video_ai_editor.face_identity.FaceIdentityBank`` wraps the vendored YuNet
detector and the SFace embedder. This module points that same machinery at a
directory of photographs instead of video frames, and adds the two things a
still-photo library needs that a video pass does not:

* **Offline clustering.** ``FaceIdentityBank.match()`` is sequential: a face
  joins the first identity it is close enough to, because when you are walking
  a video forward that is all you can do. A photo set is finite and available
  all at once, so we can do better — collect every embedding first, then
  agglomerate globally. Order stops mattering and the groupings get tighter.
* **Thumbnails.** The source files are camera originals (10 MB+ each). Nothing
  interactive can afford to decode those on demand, so a scan writes a small
  JPEG per photo once and the browser reads only those.

Everything is persisted next to the photo folder in a single JSON index, so a
rescan is incremental and the app can reopen a library instantly.

Content neutrality: this groups faces into "people" and knows nothing about who
they are. Names are user-supplied at runtime and live only in the user's index.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, Optional

import numpy as np

# Long edge the detector sees. The originals are ~6000px; YuNet gains nothing
# above ~1600 and costs linearly, so downscale first and scale boxes back up.
DETECT_LONG_EDGE = 1600

# Long edge of the cached browse thumbnail.
THUMB_LONG_EDGE = 480

# Long edge of a cropped face chip, used as a cluster's avatar.
FACE_CHIP_EDGE = 160

# Cosine similarity above which two SFace embeddings are the same person.
#
# Deliberately *lower* than face_identity's 0.363. That value is tuned for video,
# where a sequential matcher sees near-identical frames and a loose threshold
# would chain unrelated people together. A photo set is the opposite problem: a
# guest appears a handful of times, in different light, from different angles,
# and too strict a threshold shatters them into a dozen one-photo "people".
# Measured on a 60-photo sample, 0.363 produced 36 singletons out of 62 clusters
# while 0.25 gave 11 out of 40 with the large clusters unchanged. Erring loose
# is also the recoverable direction: a split person is fixed with merge_people(),
# whereas two guests silently fused are much harder to notice.
CLUSTER_THRESHOLD = 0.25

# A detection this weak is usually a face-shaped smudge in the background.
MIN_DETECT_SCORE = 0.7

# Minimum face width/height, in pixels on the DETECT_LONG_EDGE-scaled image.
#
# This is the single most important quality knob, and it is not really about
# detection — YuNet happily finds faces far below it. It is about the *embedding*.
# SFace needs enough pixels to describe a face; below roughly 50px its output
# degrades toward a generic average-face vector. Those degenerate vectors are
# then weakly similar to each other regardless of who they belong to, so they
# collect into large "clusters" of unrelated background guests that look
# convincing by size alone.
#
# Measured over 507 wedding photos (1213 faces): at 32px/0.6 the 4th-6th largest
# clusters had mean intra-similarity ~0.30 and contained visibly different
# people, with one pair of members actually anti-correlated (-0.12). Raising to
# 50px/0.7 dropped those to tight groups at 0.41-0.48 — the same band as the
# known-good clusters — while the two largest (the couple) barely moved
# (178->174, 172->173). Going further to 60px/0.75 kept cleaning up but began
# eating real photos of the couple, so this is the knee.
MIN_FACE_PX = 50

INDEX_FILENAME = "photo_index.json"
THUMB_DIRNAME = ".photo_cache"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _noop(_msg: str) -> None:
    pass


def list_images(folder: str) -> list[str]:
    """Image files directly inside ``folder``, sorted by name."""
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    out = [
        n for n in names
        if n.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(folder, n))
    ]
    out.sort()
    return out


def index_path(folder: str) -> str:
    return os.path.join(folder, THUMB_DIRNAME, INDEX_FILENAME)


def thumb_dir(folder: str) -> str:
    return os.path.join(folder, THUMB_DIRNAME)


def _imread_unicode(path: str):
    """cv2.imread cannot open non-ASCII paths on Windows; go through numpy."""
    import cv2
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imwrite_unicode(path: str, img, quality: int = 85) -> bool:
    """Counterpart to _imread_unicode — cv2.imwrite has the same path limit."""
    import cv2
    try:
        ext = os.path.splitext(path)[1] or ".jpg"
        ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def _shot_time(path: str) -> Optional[str]:
    """EXIF capture time, so the library can sort in the order things happened
    rather than by filename. Falls back to None; the caller then uses mtime."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            # 36867 DateTimeOriginal, 306 DateTime
            for tag in (36867, 306):
                val = exif.get(tag)
                if val:
                    return str(val)
    except Exception:
        pass
    return None


class PhotoLibrary:
    """A scanned folder: photos, the faces found in them, and the people those
    faces group into.

    The index is a plain dict so it serialises straight to JSON::

        {
          "folder": str,
          "photos": {filename: {"faces": [face, ...], "thumb": str,
                                "shot": str|None, "mtime": float}},
          "people": {person_id: {"name": str, "photos": [filename, ...],
                                 "chip": str|None, "size": int}},
          "version": int,
        }

    A ``face`` carries its bounding box in *original* image coordinates, its
    detection score, and the person it was assigned to.
    """

    VERSION = 1

    def __init__(self, folder: str):
        self.folder = os.path.abspath(folder)
        self.photos: dict[str, dict] = {}
        self.people: dict[str, dict] = {}

    # ---------------------------------------------------------------- scan

    def scan(
        self,
        log_fn: Callable[[str], None] = _noop,
        progress_fn: Callable[[int, int], None] = None,
        should_stop: Callable[[], bool] = None,
        rescan: bool = False,
    ) -> int:
        """Detect faces and build thumbnails for every photo in the folder.

        Incremental by default: a photo already in the index with an unchanged
        mtime is skipped, so re-scanning after adding a few shots is cheap.
        Returns the number of photos actually processed.
        """
        import cv2
        from video_ai_editor.face_identity import FaceIdentityBank

        files = list_images(self.folder)
        if not files:
            log_fn("No images found in that folder.")
            return 0

        os.makedirs(thumb_dir(self.folder), exist_ok=True)

        # The bank is used purely as a detector+embedder here; the grouping is
        # done afterwards by cluster_faces(), not by the bank's own matcher.
        bank = FaceIdentityBank(db_path=os.path.join(thumb_dir(self.folder), "_scratch_bank.json"))

        processed = 0
        total = len(files)
        t0 = time.time()

        for i, name in enumerate(files):
            if should_stop is not None and should_stop():
                log_fn("Scan cancelled.")
                break

            path = os.path.join(self.folder, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue

            prior = self.photos.get(name)
            if not rescan and prior and abs(prior.get("mtime", -1) - mtime) < 1e-6:
                if progress_fn:
                    progress_fn(i + 1, total)
                continue

            img = _imread_unicode(path)
            if img is None:
                log_fn(f"Could not read {name} — skipped.")
                if progress_fn:
                    progress_fn(i + 1, total)
                continue

            h, w = img.shape[:2]
            scale = DETECT_LONG_EDGE / float(max(h, w)) if max(h, w) > DETECT_LONG_EDGE else 1.0
            small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img

            try:
                found = bank.detect_faces(small)
            except Exception as e:
                log_fn(f"Face detection failed on {name}: {e}")
                found = []

            faces = []
            for f in found:
                # detect_faces reports confidence as "det_score" (see
                # FaceIdentityBank.detect_faces); "score" is accepted as a
                # fallback so a future rename there cannot silently zero this out.
                score = float(f.get("det_score", f.get("score", 0.0)) or 0.0)
                box = f.get("bbox") or f.get("box")
                if box is None:
                    continue
                # YuNet reports CORNERS (x1, y1, x2, y2) — see the contract at
                # face_identity.detect_faces and its _crop helper. Converting to
                # width/height here keeps the rest of this module in one format.
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                bw, bh = x2 - x1, y2 - y1
                if score < MIN_DETECT_SCORE or min(bw, bh) < MIN_FACE_PX:
                    continue
                emb = f.get("embedding")
                if emb is None:
                    continue
                # Rescale by the ratio actually achieved: `small` is built with
                # int() truncation, so 1/scale is subtly wrong on some sizes.
                inv_x = w / float(small.shape[1])
                inv_y = h / float(small.shape[0])
                faces.append({
                    # stored as [x, y, w, h] in ORIGINAL pixel coordinates
                    "bbox": [x1 * inv_x, y1 * inv_y, bw * inv_x, bh * inv_y],
                    "score": score,
                    "embedding": np.asarray(emb, dtype=np.float32).ravel().tolist(),
                    "person": None,
                })

            thumb_name = self._write_thumb(img, name)

            self.photos[name] = {
                "faces": faces,
                "thumb": thumb_name,
                "shot": _shot_time(path),
                "mtime": mtime,
                "size": [w, h],
            }
            processed += 1

            if progress_fn:
                progress_fn(i + 1, total)
            if processed % 25 == 0:
                rate = processed / max(1e-6, time.time() - t0)
                log_fn(f"Scanned {i + 1}/{total} photos ({rate:.1f}/s)…")

        # The scratch bank was only a vehicle for the models; the index owns the
        # embeddings, so don't leave a stray db behind.
        scratch = os.path.join(thumb_dir(self.folder), "_scratch_bank.json")
        if os.path.exists(scratch):
            try:
                os.remove(scratch)
            except OSError:
                pass

        n_faces = sum(len(p["faces"]) for p in self.photos.values())
        log_fn(f"Scan complete: {len(self.photos)} photos, {n_faces} faces "
               f"in {time.time() - t0:.0f}s.")
        return processed

    def _write_thumb(self, img, name: str) -> str:
        """Cache a small JPEG for the grid. Returns the cache-relative filename."""
        import cv2
        h, w = img.shape[:2]
        s = THUMB_LONG_EDGE / float(max(h, w))
        thumb = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s)))) if s < 1.0 else img
        out_name = os.path.splitext(name)[0] + "_t.jpg"
        _imwrite_unicode(os.path.join(thumb_dir(self.folder), out_name), thumb, quality=82)
        return out_name

    # ------------------------------------------------------------- cluster

    def cluster_faces(
        self,
        threshold: float = CLUSTER_THRESHOLD,
        log_fn: Callable[[str], None] = _noop,
    ) -> int:
        """Group every detected face into people.

        Average-linkage agglomeration over cosine distance. Average linkage
        rather than single linkage because single linkage chains: one ambiguous
        profile shot that sits between two guests would merge both into one
        person, and on a set this size that cascades. Existing user-assigned
        names are preserved by re-attaching them to whichever new cluster
        inherits the majority of the old cluster's faces.

        Returns the number of people found.
        """
        from sklearn.cluster import AgglomerativeClustering

        # Snapshot who each face belonged to BEFORE anything is reset — the
        # name-carrying step below needs the old assignments, and the quality
        # gate in the next loop clears them.
        old_names = {pid: rec.get("name", "") for pid, rec in self.people.items()
                     if rec.get("name")}
        old_of_face = {}
        if old_names:
            for pname, rec in self.photos.items():
                for j, f in enumerate(rec["faces"]):
                    if f.get("person") in old_names:
                        old_of_face[(pname, j)] = f["person"]

        names: list[str] = []
        idx: list[int] = []
        vecs: list[np.ndarray] = []
        skipped = 0
        for pname, rec in self.photos.items():
            W, H = rec.get("size", [0, 0])
            # Re-apply the quality gate here, not just at scan time. The stored
            # index may predate a change to these constants, and the embeddings
            # are already on disk — so tightening the filter must take effect on
            # a Regroup rather than forcing a full rescan of every photo.
            det_scale = DETECT_LONG_EDGE / float(max(W, H)) if max(W, H) else 1.0
            for j, f in enumerate(rec["faces"]):
                f["person"] = None      # a filtered-out face belongs to nobody
                if float(f.get("score", 0.0)) < MIN_DETECT_SCORE:
                    skipped += 1
                    continue
                if float(f["bbox"][2]) * det_scale < MIN_FACE_PX:
                    skipped += 1
                    continue
                v = np.asarray(f["embedding"], dtype=np.float32)
                n = np.linalg.norm(v)
                if n < 1e-6:
                    skipped += 1
                    continue
                vecs.append(v / n)
                names.append(pname)
                idx.append(j)

        if not vecs:
            if skipped:
                log_fn(f"No faces clear enough to group — all {skipped} "
                       f"detection(s) were too small or too unclear.")
            else:
                log_fn("No faces to group.")
            self.people = {}
            return 0

        X = np.vstack(vecs)

        if len(X) == 1:
            labels = np.array([0])
        else:
            # Embeddings are L2-normalised, so cosine distance = 1 - dot.
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1.0 - threshold,
                metric="cosine",
                linkage="average",
            )
            labels = model.fit_predict(X)

        # Build people records.
        people: dict[str, dict] = {}
        for label, pname, j in zip(labels, names, idx):
            pid = f"p{int(label):04d}"
            self.photos[pname]["faces"][j]["person"] = pid
            rec = people.setdefault(pid, {"name": "", "photos": [], "chip": None, "size": 0})
            rec["size"] += 1
            if pname not in rec["photos"]:
                rec["photos"].append(pname)

        # Carry names across: each old name goes to the new cluster holding most
        # of its faces, so renaming survives a rescan.
        if old_of_face:
            tally: dict[str, dict[str, int]] = {}
            for (pname, j), old_pid in old_of_face.items():
                new_pid = self.photos[pname]["faces"][j].get("person")
                if new_pid:
                    tally.setdefault(old_pid, {}).setdefault(new_pid, 0)
                    tally[old_pid][new_pid] += 1
            taken: set[str] = set()
            for old_pid, counts in sorted(tally.items(), key=lambda kv: -sum(kv[1].values())):
                for new_pid, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
                    if new_pid in people and new_pid not in taken:
                        people[new_pid]["name"] = old_names[old_pid]
                        taken.add(new_pid)
                        break

        for pid, rec in people.items():
            rec["photos"].sort()

        self.people = people
        msg = f"Grouped {len(X)} faces into {len(people)} people."
        if skipped:
            msg += (f" Ignored {skipped} face(s) too small or unclear to "
                    f"identify reliably.")
        log_fn(msg)
        return len(people)

    def build_face_chips(self, log_fn: Callable[[str], None] = _noop) -> int:
        """Crop one representative face per person for use as its avatar.

        Picks the highest-scoring, largest detection — the cluster's clearest
        look at that person.
        """
        import cv2
        made = 0
        for pid, rec in self.people.items():
            best = None  # (rank, photo_name, bbox)
            for pname in rec["photos"]:
                for f in self.photos[pname]["faces"]:
                    if f.get("person") != pid:
                        continue
                    x, y, w, h = f["bbox"]
                    rank = float(f.get("score", 0)) * (w * h) ** 0.5
                    if best is None or rank > best[0]:
                        best = (rank, pname, f["bbox"])
            if best is None:
                continue

            _, pname, (x, y, w, h) = best
            img = _imread_unicode(os.path.join(self.folder, pname))
            if img is None:
                continue
            H, W = img.shape[:2]
            # Pad the box out to include hair and chin — a tight detector box
            # cropped to the eyes alone is hard to recognise at avatar size.
            pad = 0.35
            x0 = max(0, int(x - w * pad))
            y0 = max(0, int(y - h * pad))
            x1 = min(W, int(x + w * (1 + pad)))
            y1 = min(H, int(y + h * (1 + pad)))
            if x1 <= x0 or y1 <= y0:
                continue
            chip = img[y0:y1, x0:x1]
            s = FACE_CHIP_EDGE / float(max(chip.shape[:2]))
            if s < 1.0:
                chip = cv2.resize(chip, (max(1, int(chip.shape[1] * s)),
                                         max(1, int(chip.shape[0] * s))))
            chip_name = f"chip_{pid}.jpg"
            if _imwrite_unicode(os.path.join(thumb_dir(self.folder), chip_name), chip, 88):
                rec["chip"] = chip_name
                made += 1
        log_fn(f"Built {made} face thumbnails.")
        return made

    # -------------------------------------------------------------- naming

    def rename_person(self, person_id: str, name: str) -> bool:
        rec = self.people.get(person_id)
        if rec is None:
            return False
        rec["name"] = name.strip()
        return True

    def merge_people(self, keep_id: str, merge_id: str) -> bool:
        """Fold one person into another — the manual fix for a split cluster
        (same guest appearing as two people because of a hat or a profile)."""
        keep = self.people.get(keep_id)
        gone = self.people.get(merge_id)
        if keep is None or gone is None or keep_id == merge_id:
            return False
        for pname, rec in self.photos.items():
            for f in rec["faces"]:
                if f.get("person") == merge_id:
                    f["person"] = keep_id
        for p in gone["photos"]:
            if p not in keep["photos"]:
                keep["photos"].append(p)
        keep["photos"].sort()
        keep["size"] += gone["size"]
        if not keep.get("name") and gone.get("name"):
            keep["name"] = gone["name"]
        del self.people[merge_id]
        return True

    def display_name(self, person_id: str) -> str:
        rec = self.people.get(person_id) or {}
        if rec.get("name"):
            return rec["name"]
        # Unnamed people get a stable number so the user can still refer to one.
        return f"Person {person_id[1:].lstrip('0') or '0'}"

    # -------------------------------------------------------------- views

    def people_by_size(self) -> list[tuple[str, dict]]:
        """People ordered by how many photos they appear in — the order you want
        for naming, since the couple and immediate family come first."""
        return sorted(
            self.people.items(),
            key=lambda kv: (-len(kv[1]["photos"]), kv[0]),
        )

    def photos_for_person(self, person_id: str) -> list[str]:
        rec = self.people.get(person_id)
        return list(rec["photos"]) if rec else []

    def photos_with_all(self, person_ids: Iterable[str]) -> list[str]:
        """Photos containing every one of these people — 'just the two of us'."""
        ids = [p for p in person_ids]
        if not ids:
            return []
        sets = [set(self.photos_for_person(p)) for p in ids]
        common = set.intersection(*sets) if sets else set()
        return sorted(common)

    def photos_with_any(self, person_ids: Iterable[str]) -> list[str]:
        out: set[str] = set()
        for p in person_ids:
            out.update(self.photos_for_person(p))
        return sorted(out)

    def photos_with_no_faces(self) -> list[str]:
        """Venue, rings, cake — the detail shots. Worth their own view.

        Counts every detection, including faces too small to have been given an
        identity. A shot of the room with distant guests in it is not a detail
        shot, even though nobody in it is identifiable.
        """
        return sorted(n for n, r in self.photos.items() if not r["faces"])

    def photos_by_group_size(self, minimum: int) -> list[str]:
        """Photos containing at least `minimum` faces — identified or not, for
        the same reason as photos_with_no_faces: this asks about the picture,
        not about who the library managed to recognise."""
        return sorted(n for n, r in self.photos.items() if len(r["faces"]) >= minimum)

    def sort_chronologically(self, names: list[str]) -> list[str]:
        """EXIF capture order, falling back to mtime then filename."""
        def key(n: str):
            rec = self.photos.get(n) or {}
            return (str(rec.get("shot") or ""), rec.get("mtime", 0.0), n)
        return sorted(names, key=key)

    def thumb_path(self, photo_name: str) -> Optional[str]:
        rec = self.photos.get(photo_name)
        if not rec or not rec.get("thumb"):
            return None
        p = os.path.join(thumb_dir(self.folder), rec["thumb"])
        return p if os.path.exists(p) else None

    def chip_path(self, person_id: str) -> Optional[str]:
        rec = self.people.get(person_id)
        if not rec or not rec.get("chip"):
            return None
        p = os.path.join(thumb_dir(self.folder), rec["chip"])
        return p if os.path.exists(p) else None

    def photo_path(self, photo_name: str) -> str:
        return os.path.join(self.folder, photo_name)

    # ------------------------------------------------------------ persist

    def to_dict(self) -> dict:
        return {
            "version": self.VERSION,
            "folder": self.folder,
            "photos": self.photos,
            "people": self.people,
        }

    def save(self, path: str | None = None) -> bool:
        """Persist the index. Uses the project's shared atomic writer so a crash
        mid-write leaves the previous index intact rather than a truncated one —
        and because it already knows how to encode the numpy scalars that come
        back from the detector."""
        path = path or index_path(self.folder)
        try:
            from modules.video_cache import atomic_write_json
            atomic_write_json(path, self.to_dict())
            return True
        except Exception as e:
            print(f"PhotoLibrary.save failed: {e}")
            return False

    def load(self, path: str | None = None) -> bool:
        path = path or index_path(self.folder)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if int(data.get("version", 0)) != self.VERSION:
                print("PhotoLibrary: index version mismatch — rescan needed.")
                return False
            self.photos = data.get("photos", {}) or {}
            self.people = data.get("people", {}) or {}
            return True
        except Exception as e:
            print(f"PhotoLibrary.load failed: {e}")
            return False


# ------------------------------------------------------------------ export


def export_photos(
    library: PhotoLibrary,
    photo_names: Iterable[str],
    dest_dir: str,
    long_edge: int = 0,
    quality: int = 90,
    log_fn: Callable[[str], None] = _noop,
    progress_fn: Callable[[int, int], None] = None,
    should_stop: Callable[[], bool] = None,
) -> tuple[int, int]:
    """Write the given photos into ``dest_dir``.

    ``long_edge`` of 0 copies the original file untouched — byte-for-byte, EXIF
    intact. Any other value re-encodes to that long edge, for sharing. Never
    writes over an existing file; a name collision gets a numeric suffix.

    Returns ``(written, failed)``.
    """
    import shutil
    import cv2

    names = list(photo_names)
    os.makedirs(dest_dir, exist_ok=True)
    written = 0
    failed = 0

    for i, name in enumerate(names):
        if should_stop is not None and should_stop():
            log_fn("Export cancelled.")
            break

        src = library.photo_path(name)
        base, ext = os.path.splitext(name)
        out_ext = ext if long_edge <= 0 else ".jpg"
        dest = os.path.join(dest_dir, base + out_ext)
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{base}_{n}{out_ext}")
            n += 1

        try:
            if long_edge <= 0:
                shutil.copy2(src, dest)     # copy2 keeps mtime + EXIF
            else:
                img = _imread_unicode(src)
                if img is None:
                    raise OSError("unreadable")
                h, w = img.shape[:2]
                s = float(long_edge) / max(h, w)
                if s < 1.0:
                    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                                     interpolation=cv2.INTER_AREA)
                if not _imwrite_unicode(dest, img, quality):
                    raise OSError("encode failed")
            written += 1
        except Exception as e:
            log_fn(f"Failed to export {name}: {e}")
            failed += 1

        if progress_fn:
            progress_fn(i + 1, len(names))

    log_fn(f"Exported {written} photo(s) to {dest_dir}"
           + (f" — {failed} failed." if failed else "."))
    return written, failed
