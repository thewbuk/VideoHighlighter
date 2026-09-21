"""Tests for the face-grouped photo library.

These are deliberately logic-only: no image decoding, no models, no real
photographs. Face records are synthesised directly into the index, because the
parts worth pinning down are the grouping, the filtered views and the export
naming — not whether OpenCV can read a JPEG.

The one thing these tests exist to catch above all: the bbox convention. YuNet
returns corners (x1, y1, x2, y2) while this module stores width/height, and
getting that backwards produces boxes that look plausible in a debugger but
crop the wrong region — see test_bbox_stored_as_width_height.
"""

from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from modules.vision.photo_library import (
    PhotoLibrary,
    export_photos,
    list_images,
    index_path,
    thumb_dir,
)


# Modules that must be the genuine article for clustering to run. sklearn is
# the obvious one; torch is here because importing sklearn pulls in scipy, and
# scipy's array-API probe calls issubclass(cls, torch.Tensor) — against a
# MagicMock that raises "arg 2 must be a class" and takes the whole test with it.
_CLUSTER_DEPS = ("sklearn", "torch")


@pytest.fixture
def real_sklearn():
    """Lend the real scikit-learn to clustering tests, then give the shims back.

    conftest replaces these with MagicMocks so logic-only CI can import the
    engine without them. That is right for the suite at large but fatal here:
    AgglomerativeClustering().fit_predict() would return a MagicMock instead of
    a label array, and every face would silently land in no cluster at all.
    Same borrow-and-restore dance as conftest.real_opencv(), for the same
    reason — the rest of the suite is written against the shims.
    """
    saved = {n: m for n, m in sys.modules.items()
             if n.split(".")[0] in _CLUSTER_DEPS and isinstance(m, MagicMock)}
    for name in list(saved):
        del sys.modules[name]
    try:
        import sklearn.cluster  # noqa: F401
    except ImportError:
        sys.modules.update(saved)
        pytest.skip("scikit-learn not installed")
    try:
        yield
    finally:
        # Drop the real modules and put the shims back exactly as they were, so
        # later tests in the same session still see what conftest gave them.
        for name in [n for n in list(sys.modules)
                     if n.split(".")[0] in _CLUSTER_DEPS]:
            del sys.modules[name]
        sys.modules.update(saved)


def _unit(vec) -> list[float]:
    v = np.asarray(vec, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _person_vector(seed: int, jitter: float = 0.0, dim: int = 128) -> list[float]:
    """A stable 128-d direction per seed, optionally nudged.

    Small jitter keeps a face inside its own cluster; large jitter pushes it
    out. That is enough to exercise the clustering without real embeddings.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=dim)
    if jitter:
        base = base + np.random.default_rng(seed + 1000).normal(size=dim) * jitter
    return _unit(base)


# A face big enough to clear MIN_FACE_PX once scaled to the detection size.
# The index stores original-image pixels, and clustering rescales by
# DETECT_LONG_EDGE/long_edge before applying the threshold — on the 4000x6000
# frame used here that is a factor of 6000/1600, so a face must be ~190px in
# the original to count. 400 keeps these fixtures clear of the boundary.
_FACE_W = 400.0
_FACE_H = 480.0


def _library(tmp_path, layout: dict[str, list[int]]) -> PhotoLibrary:
    """Build an index directly: ``{photo_name: [person_seed, ...]}``."""
    lib = PhotoLibrary(str(tmp_path))
    for i, (name, seeds) in enumerate(layout.items()):
        lib.photos[name] = {
            "faces": [
                {
                    "bbox": [10.0, 20.0, _FACE_W, _FACE_H],
                    "score": 0.9,
                    "embedding": _person_vector(s),
                    "person": None,
                }
                for s in seeds
            ],
            "thumb": None,
            "shot": f"2026:08:27 10:{i:02d}:00",
            "mtime": 1000.0 + i,
            "size": [4000, 6000],
        }
    return lib


# --------------------------------------------------------------- clustering


def test_identical_faces_group_together(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [1], "c.jpg": [2]})
    n = lib.cluster_faces()
    assert n == 2
    groups = {len(r["photos"]) for r in lib.people.values()}
    assert groups == {2, 1}


def test_distinct_people_stay_separate(real_sklearn, tmp_path):
    lib = _library(tmp_path, {f"p{i}.jpg": [i] for i in range(6)})
    assert lib.cluster_faces() == 6


def test_cluster_assigns_person_to_every_face(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1, 2], "b.jpg": [1]})
    lib.cluster_faces()
    for rec in lib.photos.values():
        for face in rec["faces"]:
            assert face["person"] is not None


def test_names_survive_regrouping(real_sklearn, tmp_path):
    """Renaming is the user's investment; re-clustering must not discard it."""
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [1], "c.jpg": [2]})
    lib.cluster_faces()
    biggest = lib.people_by_size()[0][0]
    lib.rename_person(biggest, "Alex")

    lib.cluster_faces()          # regroup at the same threshold
    assert "Alex" in [r["name"] for r in lib.people.values()]


def test_empty_library_clusters_to_nothing(real_sklearn, tmp_path):
    lib = PhotoLibrary(str(tmp_path))
    assert lib.cluster_faces() == 0
    assert lib.people == {}


def test_single_face_forms_one_person(real_sklearn, tmp_path):
    """The n==1 path skips sklearn entirely; make sure it still produces a
    person rather than falling through with an empty index."""
    lib = _library(tmp_path, {"only.jpg": [7]})
    assert lib.cluster_faces() == 1
    assert len(lib.people_by_size()[0][1]["photos"]) == 1


def test_tiny_faces_are_not_clustered(real_sklearn, tmp_path):
    """Small faces must be excluded from grouping entirely.

    Regression test for the defect that made this worth doing: below roughly
    50px (at detection scale) SFace embeddings collapse toward a generic
    average-face vector, so unrelated background guests come out weakly similar
    to each other and collect into a large, convincing-looking cluster of
    different people. Two distinct identities at a tiny size must therefore
    produce no people at all, rather than one merged group.
    """
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [2]})
    for rec in lib.photos.values():
        for face in rec["faces"]:
            face["bbox"] = [10.0, 20.0, 40.0, 48.0]     # ~11px once scaled down

    assert lib.cluster_faces() == 0
    for rec in lib.photos.values():
        for face in rec["faces"]:
            assert face["person"] is None


def test_low_confidence_faces_are_not_clustered(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [1]})
    for rec in lib.photos.values():
        for face in rec["faces"]:
            face["score"] = 0.4

    assert lib.cluster_faces() == 0


def test_filtered_faces_still_count_as_people_in_the_photo(real_sklearn, tmp_path):
    """A photo full of distant guests is not a 'detail shot'.

    The no-people and group-shot views deliberately count every detection, not
    just the identifiable ones — they describe the picture, not the library's
    recognition rate.
    """
    lib = _library(tmp_path, {"crowd.jpg": [1, 2, 3], "ring.jpg": []})
    for face in lib.photos["crowd.jpg"]["faces"]:
        face["bbox"] = [10.0, 20.0, 40.0, 48.0]
    lib.cluster_faces()

    assert lib.photos_with_no_faces() == ["ring.jpg"]
    assert lib.photos_by_group_size(3) == ["crowd.jpg"]


# ------------------------------------------------------------------ naming


def test_display_name_falls_back_to_number(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1]})
    lib.cluster_faces()
    pid = next(iter(lib.people))
    assert lib.display_name(pid).startswith("Person")
    lib.rename_person(pid, "Sam")
    assert lib.display_name(pid) == "Sam"


def test_merge_people_combines_photos_and_keeps_name(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [2]})
    lib.cluster_faces()
    ids = [p for p, _ in lib.people_by_size()]
    assert len(ids) == 2
    lib.rename_person(ids[1], "Robin")

    assert lib.merge_people(ids[0], ids[1]) is True
    assert ids[1] not in lib.people
    assert set(lib.people[ids[0]]["photos"]) == {"a.jpg", "b.jpg"}
    # The surviving group had no name, so it inherits the other's.
    assert lib.people[ids[0]]["name"] == "Robin"
    # Every face now points at the surviving id.
    for rec in lib.photos.values():
        for face in rec["faces"]:
            assert face["person"] == ids[0]


def test_merge_rejects_nonsense(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1]})
    lib.cluster_faces()
    pid = next(iter(lib.people))
    assert lib.merge_people(pid, pid) is False
    assert lib.merge_people(pid, "nope") is False


# ------------------------------------------------------------------- views


def test_photos_with_all_is_intersection(real_sklearn, tmp_path):
    lib = _library(tmp_path, {
        "both.jpg": [1, 2],
        "just1.jpg": [1],
        "just2.jpg": [2],
    })
    lib.cluster_faces()
    by_seed = {}
    for pid, rec in lib.people.items():
        # identify each cluster by a photo only it appears in
        by_seed[tuple(sorted(rec["photos"]))] = pid
    p1 = next(p for p, r in lib.people.items() if "just1.jpg" in r["photos"])
    p2 = next(p for p, r in lib.people.items() if "just2.jpg" in r["photos"])

    assert lib.photos_with_all([p1, p2]) == ["both.jpg"]
    assert lib.photos_with_any([p1, p2]) == ["both.jpg", "just1.jpg", "just2.jpg"]


def test_photos_with_all_of_nothing_is_empty(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1]})
    lib.cluster_faces()
    assert lib.photos_with_all([]) == []


def test_no_face_and_group_views(real_sklearn, tmp_path):
    lib = _library(tmp_path, {
        "empty.jpg": [],
        "solo.jpg": [1],
        "crowd.jpg": [1, 2, 3, 4],
    })
    lib.cluster_faces()
    assert lib.photos_with_no_faces() == ["empty.jpg"]
    assert lib.photos_by_group_size(3) == ["crowd.jpg"]
    assert set(lib.photos_by_group_size(1)) == {"solo.jpg", "crowd.jpg"}


def test_chronological_sort_uses_exif_then_name(tmp_path):
    lib = _library(tmp_path, {"z.jpg": [1], "a.jpg": [2]})
    # _library stamps increasing times in insertion order: z then a.
    assert lib.sort_chronologically(["a.jpg", "z.jpg"]) == ["z.jpg", "a.jpg"]


def test_people_by_size_is_descending(real_sklearn, tmp_path):
    lib = _library(tmp_path, {
        "a.jpg": [1], "b.jpg": [1], "c.jpg": [1], "d.jpg": [2],
    })
    lib.cluster_faces()
    counts = [len(r["photos"]) for _, r in lib.people_by_size()]
    assert counts == sorted(counts, reverse=True)


# -------------------------------------------------------------- bbox format


def test_bbox_stored_as_width_height(tmp_path):
    """Regression guard for the detector's coordinate convention.

    detect_faces hands back CORNERS; this module stores width/height. If the
    conversion is dropped, a face's stored 'width' becomes its right edge — a
    number several times too large that still looks like a plausible box. The
    symptom downstream is avatar crops full of background, so assert the shape
    of a stored box directly: it must be small relative to the image, and
    x + w / y + h must stay inside the frame.
    """
    lib = _library(tmp_path, {"a.jpg": [1]})
    W, H = lib.photos["a.jpg"]["size"]
    x, y, w, h = lib.photos["a.jpg"]["faces"][0]["bbox"]
    assert 0 <= x < W and 0 <= y < H
    assert x + w <= W and y + h <= H
    assert w < W / 2 and h < H / 2


# ---------------------------------------------------------------- persistence


def test_save_and_load_round_trip(real_sklearn, tmp_path):
    lib = _library(tmp_path, {"a.jpg": [1], "b.jpg": [1]})
    lib.cluster_faces()
    pid = next(iter(lib.people))
    lib.rename_person(pid, "Jo")
    assert lib.save() is True
    assert os.path.exists(index_path(str(tmp_path)))

    fresh = PhotoLibrary(str(tmp_path))
    assert fresh.load() is True
    assert set(fresh.photos) == {"a.jpg", "b.jpg"}
    assert [r["name"] for r in fresh.people.values() if r["name"]] == ["Jo"]


def test_load_missing_index_is_false(tmp_path):
    assert PhotoLibrary(str(tmp_path)).load() is False


def test_load_rejects_future_version(tmp_path):
    """A newer build's index must not be half-read by an older one."""
    lib = _library(tmp_path, {"a.jpg": [1]})
    lib.save()
    p = index_path(str(tmp_path))
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["version"] = PhotoLibrary.VERSION + 99
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)

    assert PhotoLibrary(str(tmp_path)).load() is False


def test_load_survives_corrupt_index(tmp_path):
    os.makedirs(thumb_dir(str(tmp_path)), exist_ok=True)
    with open(index_path(str(tmp_path)), "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    assert PhotoLibrary(str(tmp_path)).load() is False


# --------------------------------------------------------------------- files


def test_list_images_filters_and_sorts(tmp_path):
    for n in ["b.JPG", "a.jpg", "notes.txt", "c.png"]:
        (tmp_path / n).write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.jpg").write_bytes(b"x")

    got = list_images(str(tmp_path))
    assert got == ["a.jpg", "b.JPG", "c.png"]     # sorted, no txt, not recursive


def test_list_images_on_missing_folder(tmp_path):
    assert list_images(str(tmp_path / "nope")) == []


# -------------------------------------------------------------------- export


def test_export_copies_originals_and_avoids_collisions(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for n in ("a.jpg", "b.jpg"):
        (src / n).write_bytes(b"original-bytes")
    dest = tmp_path / "out"

    lib = PhotoLibrary(str(src))
    lib.photos = {"a.jpg": {"faces": [], "thumb": None, "shot": None,
                            "mtime": 1.0, "size": [10, 10]},
                  "b.jpg": {"faces": [], "thumb": None, "shot": None,
                            "mtime": 2.0, "size": [10, 10]}}

    written, failed = export_photos(lib, ["a.jpg", "b.jpg"], str(dest))
    assert (written, failed) == (2, 0)
    assert (dest / "a.jpg").read_bytes() == b"original-bytes"

    # Exporting again must not overwrite what is already there.
    written, failed = export_photos(lib, ["a.jpg"], str(dest))
    assert (written, failed) == (1, 0)
    assert (dest / "a_1.jpg").exists()
    assert (dest / "a.jpg").read_bytes() == b"original-bytes"


def test_export_counts_unreadable_files_as_failed(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    lib = PhotoLibrary(str(src))
    lib.photos = {"ghost.jpg": {"faces": [], "thumb": None, "shot": None,
                                "mtime": 1.0, "size": [10, 10]}}
    written, failed = export_photos(lib, ["ghost.jpg"], str(tmp_path / "out"))
    assert (written, failed) == (0, 1)


def test_export_of_nothing_is_harmless(tmp_path):
    lib = PhotoLibrary(str(tmp_path))
    assert export_photos(lib, [], str(tmp_path / "out")) == (0, 0)


def test_export_stops_when_cancelled(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    names = []
    for i in range(5):
        n = f"{i}.jpg"
        (src / n).write_bytes(b"x")
        names.append(n)
    lib = PhotoLibrary(str(src))
    lib.photos = {n: {"faces": [], "thumb": None, "shot": None,
                      "mtime": 1.0, "size": [10, 10]} for n in names}

    calls = {"n": 0}

    def stop_after_two():
        calls["n"] += 1
        return calls["n"] > 2

    written, _ = export_photos(lib, names, str(tmp_path / "out"),
                               should_stop=stop_after_two)
    assert written < len(names)
