"""The Photos tab: browse a scanned photo folder grouped by the people in it.

Layout is a people sidebar on the left, a thumbnail grid on the right. Picking
people filters the grid; the grid's current contents are what Export writes out,
so "what you see is what you get" — there is no separate export selection model
to keep in sync.

Everything slow (scanning, exporting) runs on a QThread and reports through
signals, matching the worker convention in main.py.
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget,
    QListWidgetItem, QLineEdit, QComboBox, QFileDialog, QProgressBar,
    QAbstractItemView, QSplitter, QMessageBox, QInputDialog, QSpinBox,
    QCheckBox, QGroupBox,
)

from modules.vision.photo_library import PhotoLibrary, export_photos, CLUSTER_THRESHOLD

# Grid cell size. Thumbnails are cached at 480px long edge, so 180 leaves room
# to grow the cell without going back to the originals.
GRID_ICON = 180


class PhotoScanWorker(QThread):
    """Scan + cluster a folder. Emits progress so the bar can move during the
    per-photo pass, which is the slow part (JPEG decode dominates)."""

    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(object)      # PhotoLibrary on success, None on failure

    def __init__(self, folder: str, threshold: float, rescan: bool = False):
        super().__init__()
        self.folder = folder
        self.threshold = threshold
        self.rescan = rescan
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            lib = PhotoLibrary(self.folder)
            if not self.rescan:
                lib.load()
            lib.scan(
                log_fn=self.log.emit,
                progress_fn=lambda i, n: self.progress.emit(i, n),
                should_stop=self._cancel.is_set,
                rescan=self.rescan,
            )
            if self._cancel.is_set():
                self.done.emit(None)
                return
            lib.cluster_faces(threshold=self.threshold, log_fn=self.log.emit)
            lib.build_face_chips(log_fn=self.log.emit)
            lib.save()
            self.done.emit(lib)
        except Exception as e:
            self.log.emit(f"Photo scan failed: {e}")
            self.done.emit(None)


class PhotoExportWorker(QThread):
    """Copy or re-encode a set of photos into a chosen folder."""

    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(int, int)    # written, failed

    def __init__(self, library, names, dest, long_edge):
        super().__init__()
        self.library = library
        self.names = list(names)
        self.dest = dest
        self.long_edge = long_edge
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        try:
            w, f = export_photos(
                self.library, self.names, self.dest,
                long_edge=self.long_edge,
                log_fn=self.log.emit,
                progress_fn=lambda i, n: self.progress.emit(i, n),
                should_stop=self._cancel.is_set,
            )
            self.done.emit(w, f)
        except Exception as e:
            self.log.emit(f"Export failed: {e}")
            self.done.emit(0, len(self.names))


class PhotoTab(QWidget):
    """Self-contained Photos tab.

    Built as its own widget rather than inline in the main window's __init__
    because that constructor is already thousands of lines; the tab only needs
    a log callback from the host.
    """

    def __init__(self, log_fn=None, parent=None):
        super().__init__(parent)
        self._log = log_fn or (lambda m: print(m))
        self.library: PhotoLibrary | None = None
        self.scan_worker: PhotoScanWorker | None = None
        self.export_worker: PhotoExportWorker | None = None
        self._shown: list[str] = []     # photo names currently in the grid

        root = QVBoxLayout(self)

        # ---- folder row -------------------------------------------------
        top = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Pick a folder of photos…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_folder)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self._start_scan)
        self.rescan_btn = QPushButton("Rescan all")
        self.rescan_btn.setToolTip(
            "Re-read every photo, ignoring the saved index.\n"
            "Use after changing the grouping sensitivity."
        )
        self.rescan_btn.clicked.connect(lambda: self._start_scan(rescan=True))
        top.addWidget(QLabel("Folder:"))
        top.addWidget(self.folder_edit, 1)
        top.addWidget(browse)
        top.addWidget(self.scan_btn)
        top.addWidget(self.rescan_btn)
        root.addLayout(top)

        # ---- grouping sensitivity --------------------------------------
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Grouping sensitivity:"))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(15, 45)
        self.threshold_spin.setValue(int(CLUSTER_THRESHOLD * 100))
        self.threshold_spin.setSuffix(" %")
        self.threshold_spin.setToolTip(
            "How similar two faces must be to count as the same person.\n"
            "Lower = fewer, larger groups (people merged together).\n"
            "Higher = more, smaller groups (one person split up).\n"
            "Change this, then press Regroup."
        )
        opts.addWidget(self.threshold_spin)
        self.regroup_btn = QPushButton("Regroup")
        self.regroup_btn.setToolTip("Re-group the faces already scanned — no re-reading of photos.")
        self.regroup_btn.clicked.connect(self._regroup)
        opts.addWidget(self.regroup_btn)
        opts.addStretch()
        root.addLayout(opts)

        # ---- progress ---------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        # ---- main split: people | photos --------------------------------
        split = QSplitter(Qt.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("People (most photos first)"))
        self.people_list = QListWidget()
        self.people_list.setIconSize(QSize(64, 64))
        self.people_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.people_list.itemSelectionChanged.connect(self._refresh_grid)
        self.people_list.itemDoubleClicked.connect(self._rename_person)
        lv.addWidget(self.people_list, 1)

        btns = QHBoxLayout()
        rename = QPushButton("Rename")
        rename.setToolTip("Give this person a name (or double-click them).")
        rename.clicked.connect(lambda: self._rename_person(self.people_list.currentItem()))
        merge = QPushButton("Merge")
        merge.setToolTip(
            "Select two or more people, then Merge to combine them.\n"
            "Use when the same guest was split into separate groups."
        )
        merge.clicked.connect(self._merge_selected)
        btns.addWidget(rename)
        btns.addWidget(merge)
        lv.addLayout(btns)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        filt = QHBoxLayout()
        filt.addWidget(QLabel("Show:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "All photos",
            "Selected people (any)",
            "Selected people (all together)",
            "Photos with no people",
            "Group shots (3+ people)",
        ])
        self.view_combo.currentIndexChanged.connect(self._refresh_grid)
        filt.addWidget(self.view_combo)
        self.chrono_check = QCheckBox("Sort by time taken")
        self.chrono_check.setChecked(True)
        self.chrono_check.stateChanged.connect(self._refresh_grid)
        filt.addWidget(self.chrono_check)
        filt.addStretch()
        self.count_label = QLabel("")
        filt.addWidget(self.count_label)
        rv.addLayout(filt)

        self.grid = QListWidget()
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(GRID_ICON, GRID_ICON))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(6)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.itemDoubleClicked.connect(self._open_photo)
        rv.addWidget(self.grid, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 700])
        root.addWidget(split, 1)

        # ---- export -----------------------------------------------------
        ex = QGroupBox("Export")
        exl = QHBoxLayout(ex)
        exl.addWidget(QLabel("Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems([
            "Originals (full quality)",
            "Large (2048px)",
            "Medium (1280px)",
            "Small (800px)",
        ])
        exl.addWidget(self.size_combo)
        self.export_shown_btn = QPushButton("Export shown")
        self.export_shown_btn.setToolTip("Export every photo currently in the grid.")
        self.export_shown_btn.clicked.connect(lambda: self._export(selected_only=False))
        self.export_sel_btn = QPushButton("Export selected")
        self.export_sel_btn.setToolTip("Export only the photos you have highlighted in the grid.")
        self.export_sel_btn.clicked.connect(lambda: self._export(selected_only=True))
        exl.addStretch()
        exl.addWidget(self.export_shown_btn)
        exl.addWidget(self.export_sel_btn)
        root.addWidget(ex)

        self._set_busy(False)

    # ------------------------------------------------------------ helpers

    def _set_busy(self, busy: bool):
        for w in (self.scan_btn, self.rescan_btn, self.regroup_btn,
                  self.export_shown_btn, self.export_sel_btn):
            w.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _pick_folder(self):
        start = self.folder_edit.text().strip() or os.path.expanduser("~")
        d = QFileDialog.getExistingDirectory(self, "Choose a photo folder", start)
        if d:
            self.folder_edit.setText(d)
            # Reopen instantly if this folder was scanned before.
            lib = PhotoLibrary(d)
            if lib.load():
                self.library = lib
                self._log(f"Loaded saved index: {len(lib.photos)} photos, {len(lib.people)} people.")
                self._refresh_people()
                self._refresh_grid()

    # -------------------------------------------------------------- scan

    def _start_scan(self, rescan: bool = False):
        folder = self.folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No folder", "Pick a folder of photos first.")
            return
        self._set_busy(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.scan_worker = PhotoScanWorker(
            folder, self.threshold_spin.value() / 100.0, rescan=rescan
        )
        self.scan_worker.log.connect(self._log)
        self.scan_worker.progress.connect(self._on_progress)
        self.scan_worker.done.connect(self._on_scan_done)
        self.scan_worker.start()

    def _on_progress(self, i: int, n: int):
        self.progress.setRange(0, max(1, n))
        self.progress.setValue(i)

    def _on_scan_done(self, lib):
        self._set_busy(False)
        self.scan_worker = None
        if lib is None:
            return
        self.library = lib
        self._refresh_people()
        self._refresh_grid()

    def _regroup(self):
        """Re-cluster without re-reading photos — the embeddings are already in
        the index, so changing sensitivity is near-instant."""
        if not self.library:
            QMessageBox.information(self, "Nothing scanned", "Scan a folder first.")
            return
        th = self.threshold_spin.value() / 100.0
        self.library.cluster_faces(threshold=th, log_fn=self._log)
        self.library.build_face_chips(log_fn=self._log)
        self.library.save()
        self._refresh_people()
        self._refresh_grid()

    # ------------------------------------------------------------ people

    def _refresh_people(self):
        self.people_list.clear()
        if not self.library:
            return
        for pid, rec in self.library.people_by_size():
            n = len(rec["photos"])
            item = QListWidgetItem(f"{self.library.display_name(pid)}  ({n})")
            item.setData(Qt.UserRole, pid)
            chip = self.library.chip_path(pid)
            if chip:
                px = QPixmap(chip)
                if not px.isNull():
                    item.setIcon(QIcon(px))
            self.people_list.addItem(item)

    def _selected_people(self) -> list[str]:
        return [i.data(Qt.UserRole) for i in self.people_list.selectedItems()]

    def _rename_person(self, item):
        if not item or not self.library:
            return
        pid = item.data(Qt.UserRole)
        cur = self.library.people.get(pid, {}).get("name", "")
        name, ok = QInputDialog.getText(
            self, "Name this person", "Name:", text=cur
        )
        if ok:
            self.library.rename_person(pid, name)
            self.library.save()
            self._refresh_people()

    def _merge_selected(self):
        if not self.library:
            return
        ids = self._selected_people()
        if len(ids) < 2:
            QMessageBox.information(
                self, "Pick two or more",
                "Select two or more people in the list, then press Merge."
            )
            return
        # Keep the largest group — it is most likely the "real" one and its name
        # is the one the user probably already set.
        ids.sort(key=lambda p: -len(self.library.people[p]["photos"]))
        keep = ids[0]
        for other in ids[1:]:
            self.library.merge_people(keep, other)
        self.library.build_face_chips(log_fn=lambda m: None)
        self.library.save()
        self._log(f"Merged {len(ids)} groups into {self.library.display_name(keep)}.")
        self._refresh_people()
        self._refresh_grid()

    # -------------------------------------------------------------- grid

    def _current_names(self) -> list[str]:
        if not self.library:
            return []
        mode = self.view_combo.currentIndex()
        sel = self._selected_people()
        if mode == 0:
            names = sorted(self.library.photos.keys())
        elif mode == 1:
            names = self.library.photos_with_any(sel)
        elif mode == 2:
            names = self.library.photos_with_all(sel)
        elif mode == 3:
            names = self.library.photos_with_no_faces()
        else:
            names = self.library.photos_by_group_size(3)
        if self.chrono_check.isChecked():
            names = self.library.sort_chronologically(names)
        return names

    def _refresh_grid(self):
        self.grid.clear()
        names = self._current_names()
        self._shown = names
        for n in names:
            item = QListWidgetItem(n)
            item.setData(Qt.UserRole, n)
            t = self.library.thumb_path(n) if self.library else None
            if t:
                px = QPixmap(t)
                if not px.isNull():
                    item.setIcon(QIcon(px))
            item.setSizeHint(QSize(GRID_ICON + 20, GRID_ICON + 40))
            self.grid.addItem(item)
        self.count_label.setText(f"{len(names)} photo(s)")

    def _open_photo(self, item):
        """Double-click opens the full-size original in the OS viewer — cheaper
        and better than building a viewer, and it is what people expect."""
        if not item or not self.library:
            return
        path = self.library.photo_path(item.data(Qt.UserRole))
        try:
            os.startfile(path)          # Windows; this app is Windows-packaged
        except AttributeError:
            import subprocess, sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
        except Exception as e:
            self._log(f"Could not open {path}: {e}")

    # ------------------------------------------------------------ export

    def _export(self, selected_only: bool):
        if not self.library:
            QMessageBox.information(self, "Nothing to export", "Scan a folder first.")
            return
        if selected_only:
            names = [i.data(Qt.UserRole) for i in self.grid.selectedItems()]
        else:
            names = list(self._shown)
        if not names:
            QMessageBox.information(
                self, "Nothing to export",
                "No photos are selected." if selected_only else "The grid is empty."
            )
            return

        dest = QFileDialog.getExistingDirectory(self, "Export photos to…")
        if not dest:
            return

        # Refuse to write into the library itself — that would land new files
        # beside the originals and the next scan would pick them up as photos.
        if os.path.abspath(dest) == os.path.abspath(self.library.folder):
            QMessageBox.warning(
                self, "Pick another folder",
                "Choose a folder outside the photo library, otherwise the "
                "exported copies become part of the library on the next scan."
            )
            return

        long_edge = [0, 2048, 1280, 800][self.size_combo.currentIndex()]
        self._set_busy(True)
        self.export_worker = PhotoExportWorker(self.library, names, dest, long_edge)
        self.export_worker.log.connect(self._log)
        self.export_worker.progress.connect(self._on_progress)
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.start()

    def _on_export_done(self, written: int, failed: int):
        self._set_busy(False)
        self.export_worker = None
        if failed:
            QMessageBox.warning(self, "Export finished",
                                f"Exported {written} photo(s); {failed} failed.")
        else:
            QMessageBox.information(self, "Export finished",
                                    f"Exported {written} photo(s).")
