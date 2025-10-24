from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

from PIL import Image
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QSplitter,
)

from . import __version__
from .core import (
    ALL_SELECTIONS,
    SELECTION_INCORRECT,
    SELECTION_SIMILAR,
    SELECTION_VALID,
    abs_path,
    create_progress_from_output_dir,
    load_progress,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Classifier Statistics Reviewer v{__version__}")
        self.resize(1100, 800)

        self.progress = None  # type: ignore
        self.current_rel_path: Optional[str] = None
        self.current_label: Optional[str] = None  # Track which label we're working on
        self.unsaved_changes = False
        self._current_qimage_bytes = None  # keep bytes alive

        # UI setup
        self._create_actions()
        self._create_menu_and_toolbar()
        self._create_central_widgets()
        self.setStatusBar(QStatusBar(self))

        self.update_ui_state()

    # ----- UI construction -----
    def _create_actions(self) -> None:
        self.act_open_dir = QAction("Open Output Directory...", self)
        self.act_open_dir.triggered.connect(self.on_open_output_dir)

        self.act_load_progress = QAction("Load Progress...", self)
        self.act_load_progress.triggered.connect(self.on_load_progress)

        self.act_save_progress = QAction("Save Progress...", self)
        self.act_save_progress.triggered.connect(self.on_save_progress)

        self.act_export_stats = QAction("Export Statistics...", self)
        self.act_export_stats.triggered.connect(self.on_export_stats)

        self.act_exit = QAction("Exit", self)
        self.act_exit.triggered.connect(self.close)

    def _create_menu_and_toolbar(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.act_open_dir)
        file_menu.addAction(self.act_load_progress)
        file_menu.addAction(self.act_save_progress)
        file_menu.addAction(self.act_export_stats)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        toolbar.addAction(self.act_open_dir)
        toolbar.addAction(self.act_load_progress)
        toolbar.addAction(self.act_save_progress)
        toolbar.addAction(self.act_export_stats)

    def _create_central_widgets(self) -> None:
        # Main splitter: left (tree) | right (content)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Label tree
        self.label_tree = QTreeWidget()
        self.label_tree.setHeaderLabel("Labels")
        self.label_tree.setMinimumWidth(200)
        self.label_tree.setMaximumWidth(400)
        self.label_tree.itemClicked.connect(self.on_tree_item_clicked)
        main_splitter.addWidget(self.label_tree)
        
        # Right side: Image viewer
        right_container = QWidget()
        vbox = QVBoxLayout(right_container)

        # Big label for the classifier label
        self.lbl_label = QLabel("")
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.lbl_label.setFont(font)
        self.lbl_label.setAlignment(Qt.AlignCenter)
        self.lbl_label.setStyleSheet("color: #333;")
        vbox.addWidget(self.lbl_label)

        # Info label (smaller)
        self.lbl_info = QLabel("Open an output directory or load a progress file to start.")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet("color: #555;")
        vbox.addWidget(self.lbl_info)

        # Image area
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setBackgroundRole(self.image_label.backgroundRole())
        self.image_label.setMinimumSize(400, 300)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        vbox.addWidget(self.scroll_area, stretch=1)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_valid = QPushButton(SELECTION_VALID)
        self.btn_similar = QPushButton(SELECTION_SIMILAR)
        self.btn_incorrect = QPushButton(SELECTION_INCORRECT)

        # Beautify buttons
        btn_style = (
            "QPushButton { padding: 10px 16px; font-size: 14px; }"
            "QPushButton:hover { background-color: #f0f0f0; }"
        )
        for b in (self.btn_valid, self.btn_similar, self.btn_incorrect):
            b.setStyleSheet(btn_style)

        self.btn_valid.clicked.connect(lambda: self.on_select(SELECTION_VALID))
        self.btn_similar.clicked.connect(lambda: self.on_select(SELECTION_SIMILAR))
        self.btn_incorrect.clicked.connect(lambda: self.on_select(SELECTION_INCORRECT))

        # Keyboard shortcuts: 1 / 2 / 3
        self.btn_valid.setShortcut("1")
        self.btn_similar.setShortcut("2")
        self.btn_incorrect.setShortcut("3")

        btn_row.addWidget(self.btn_valid)
        btn_row.addWidget(self.btn_similar)
        btn_row.addWidget(self.btn_incorrect)
        vbox.addLayout(btn_row)

        main_splitter.addWidget(right_container)
        main_splitter.setStretchFactor(0, 1)  # Left tree doesn't stretch much
        main_splitter.setStretchFactor(1, 3)  # Right content stretches more
        
        self.setCentralWidget(main_splitter)

        # Subtle overall stylesheet
        self.setStyleSheet(
            "QMainWindow { background: #fafafa; }"
            "QStatusBar { color: #333; }"
        )

    # ----- File actions -----
    def populate_label_tree(self) -> None:
        """Populate the tree widget with labels and their files."""
        # Save expanded state before clearing
        expanded_labels = set()
        for i in range(self.label_tree.topLevelItemCount()):
            item = self.label_tree.topLevelItem(i)
            if item and item.isExpanded():
                label_data = item.data(0, Qt.UserRole)
                if label_data and label_data.get("type") == "label":
                    expanded_labels.add(label_data["label"])
        
        self.label_tree.clear()
        if not self.progress:
            return
        
        labels = self.progress.get_all_labels()
        for label in labels:
            # Create parent item for label
            label_item = QTreeWidgetItem(self.label_tree)
            label_item.setText(0, label)
            label_item.setData(0, Qt.UserRole, {"type": "label", "label": label})
            
            # Add child items for files
            files = self.progress.get_label_files(label)
            for file_path in files:
                file_item = QTreeWidgetItem(label_item)
                filename = os.path.basename(file_path)
                # Check if reviewed
                is_reviewed = file_path in self.progress.decisions
                prefix = "✓ " if is_reviewed else "○ "
                file_item.setText(0, f"{prefix}{filename}")
                file_item.setData(0, Qt.UserRole, {"type": "file", "path": file_path})
            
            # Restore expanded state
            if label in expanded_labels:
                label_item.setExpanded(True)
            else:
                label_item.setExpanded(False)
    
    def on_tree_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle clicks on tree items."""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        if data["type"] == "label":
            # User clicked on a label - set it as active and load first file
            label = data["label"]
            self.load_label(label)
        elif data["type"] == "file":
            # User clicked on a specific file - just expand/collapse or could load that file
            pass
    
    def load_label(self, label: str) -> None:
        """Load a specific label and show its first unreviewed file."""
        if not self.progress:
            return
        
        self.current_label = label
        self.progress.set_active_label(label)
        self.show_next()
        self.update_ui_state()
        self.statusBar().showMessage(f"Working on label: {label}")

    def on_open_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not directory:
            return
        try:
            prog = create_progress_from_output_dir(directory)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load directory:\n{e}")
            return

        self.progress = prog
        self.unsaved_changes = False
        self.statusBar().showMessage(f"Loaded output dir: {directory}")
        self.populate_label_tree()
        # Load first label alphabetically by default
        labels = self.progress.get_all_labels()
        if labels:
            self.load_label(labels[0])
        else:
            self.show_next()
        self.update_ui_state()

    def on_load_progress(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open progress file", filter="Progress JSON (*.json)"
        )
        if not file_path:
            return
        try:
            prog = load_progress(file_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load progress:\n{e}")
            return

        self.progress = prog
        self.unsaved_changes = False
        self.statusBar().showMessage(f"Loaded progress: {file_path}")
        self.populate_label_tree()
        # Load first label alphabetically by default
        labels = self.progress.get_all_labels()
        if labels:
            self.load_label(labels[0])
        else:
            self.show_next()
        self.update_ui_state()

    def on_save_progress(self) -> None:
        if not self.progress:
            QMessageBox.information(self, "Nothing to save", "No progress to save yet.")
            return
        default_name = f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save progress", default_name, filter="Progress JSON (*.json)"
        )
        if not file_path:
            return
        try:
            self.progress.save(file_path)
            self.unsaved_changes = False
            self.statusBar().showMessage(f"Saved progress: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save progress:\n{e}")

    # ----- Core flows -----
    def show_next(self) -> None:
        if not self.progress:
            self.current_rel_path = None
            self.set_image(None)
            return
        rel = self.progress.next_rel_path()
        self.current_rel_path = rel
        if rel is None:
            self.set_image(None)
            # Check if current label is done or all is done
            if self.current_label:
                remaining_in_label = self.progress.get_remaining_for_label(self.current_label)
                if not remaining_in_label:
                    # Current label is complete
                    self.lbl_label.setText(f"{self.current_label} - Complete ✅")
                    self.lbl_info.setText(
                        "This label is finished! Please select another label from the tree to continue."
                    )
                    # Refresh tree to show all files as reviewed
                    self.populate_label_tree()
                    return
            # All done
            self.lbl_label.setText("All Done ✅")
            self.lbl_info.setText("All snippets reviewed. You can save your progress or open another directory.")
            return
        # Load and show image
        abs_p = abs_path(self.progress.output_dir, rel)
        self.load_and_show_image(abs_p)
        label = self.progress.get_label(rel)
        self.lbl_label.setText(label)
        remaining_in_label = len(self.progress.get_remaining_for_label(label)) if self.current_label else self.progress.remaining
        self.lbl_info.setText(
            f"File: {rel} | Label Remaining: {remaining_in_label} | Total Reviewed: {self.progress.reviewed}/{self.progress.total}"
        )

    def on_select(self, selection: str) -> None:
        if not self.progress or not self.current_rel_path:
            return
        try:
            self.progress.make_decision(self.current_rel_path, selection)
            self.unsaved_changes = True
            # Refresh the tree to update the checkmark
            self.populate_label_tree()
        finally:
            self.show_next()
            self.update_ui_state()

    # ----- Export statistics -----
    def on_export_stats(self) -> None:
        if not self.progress or not self.progress.decisions:
            QMessageBox.information(self, "No data", "There are no decisions to summarize yet.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select folder to save statistics")
        if not out_dir:
            return
        try:
            self._export_stats_to_dir(out_dir)
            QMessageBox.information(self, "Export complete", f"Statistics exported to:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", f"Could not export statistics:\n{e}")

    def _export_stats_to_dir(self, out_dir: str) -> None:
        import csv
        import json
        from collections import Counter, defaultdict
        import os
        import math
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        from sklearn.metrics import fbeta_score

        os.makedirs(out_dir, exist_ok=True)

        # Overall counts by selection
        by_sel = Counter(self.progress.decisions.values())

        # Per-label counts by selection
        per_label = defaultdict(lambda: Counter())
        for rel_path, sel in self.progress.decisions.items():
            label = self.progress.get_label(rel_path)
            per_label[label][sel] += 1

        # Save CSV counts by label and selection
        csv_path = os.path.join(out_dir, "counts_by_label_and_selection.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["label"] + list(ALL_SELECTIONS) + ["total"]
            writer.writerow(header)
            for label in sorted(per_label.keys()):
                row = [label] + [per_label[label].get(sel, 0) for sel in ALL_SELECTIONS]
                row.append(sum(per_label[label].values()))
                writer.writerow(row)

        # Per-label metrics (hard/soft accuracy) with 95% Wilson CI for hard accuracy
        def wilson_ci(success: int, n: int, z: float = 1.96) -> tuple[float, float]:
            if n == 0:
                return (0.0, 0.0)
            phat = success / n
            denom = 1 + z * z / n
            centre = phat + (z * z) / (2 * n)
            margin = z * ((phat * (1 - phat) + (z * z) / (4 * n)) / n) ** 0.5
            low = (centre - margin) / denom
            high = (centre + margin) / denom
            return (max(0.0, low), min(1.0, high))

        labels_sorted = sorted(per_label.keys())
        metrics_rows = []
        for label in labels_sorted:
            v = per_label[label].get(SELECTION_VALID, 0)
            s = per_label[label].get(SELECTION_SIMILAR, 0)
            i = per_label[label].get(SELECTION_INCORRECT, 0)
            n = v + s + i
            hard_acc = (v / n) if n else 0.0
            soft_acc = ((v + 0.5 * s) / n) if n else 0.0
            ci_low, ci_high = wilson_ci(v, n) if n else (0.0, 0.0)
            
            # Calculate F2 scores (Hard: Valid vs Incorrect, Soft: Valid+Similar vs Incorrect)
            # For F2 score, we need binary labels: 1 for positive class, 0 for negative
            # Hard: Valid=1 (positive), Incorrect=0 (negative), Similar excluded
            # Soft: Valid+Similar=1 (positive), Incorrect=0 (negative)
            hard_f2 = 0.0
            soft_f2 = 0.0
            
            if v + i > 0:  # Need at least some valid or incorrect for hard F2
                y_true_hard = [1] * v + [0] * i
                y_pred_hard = [1] * v + [1] * i  # All predicted as positive (the label)
                if len(set(y_true_hard)) > 1:  # Need both classes present
                    hard_f2 = fbeta_score(y_true_hard, y_pred_hard, beta=2, zero_division=0.0)
                elif v > 0:  # All valid, perfect score
                    hard_f2 = 1.0
            
            if v + s + i > 0:  # For soft F2
                y_true_soft = [1] * (v + s) + [0] * i
                y_pred_soft = [1] * (v + s) + [1] * i  # All predicted as positive
                if len(set(y_true_soft)) > 1:  # Need both classes present
                    soft_f2 = fbeta_score(y_true_soft, y_pred_soft, beta=2, zero_division=0.0)
                elif (v + s) > 0:  # All valid+similar, perfect score
                    soft_f2 = 1.0
            
            metrics_rows.append((label, n, v, s, i, hard_acc, soft_acc, ci_low, ci_high, hard_f2, soft_f2))

        metrics_csv = os.path.join(out_dir, "per_label_metrics.csv")
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "label",
                    "total",
                    "valid",
                    "similar",
                    "incorrect",
                    "hard_accuracy",
                    "soft_accuracy",
                    "hard_ci_low",
                    "hard_ci_high",
                    "hard_f2_score",
                    "soft_f2_score",
                ]
            )
            for row in metrics_rows:
                writer.writerow(row)

        # Bar chart: overall counts
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = list(range(len(ALL_SELECTIONS)))
        ys = [by_sel.get(sel, 0) for sel in ALL_SELECTIONS]
        ax.bar(xs, ys, color=["#4caf50", "#ffb300", "#e53935"])  # green, amber, red
        ax.set_xticks(xs)
        ax.set_xticklabels(ALL_SELECTIONS, rotation=20, ha="right")
        ax.set_title("Overall Decisions")
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "overall_decisions.png"), dpi=150)
        plt.close(fig)

        # Stacked bar by label
        labels_sorted = sorted(per_label.keys())
        width = 0.6
        fig2, ax2 = plt.subplots(figsize=(max(6, 0.4 * len(labels_sorted) + 2), 5))
        bottoms = [0] * len(labels_sorted)
        colors = ["#4caf50", "#ffb300", "#e53935"]
        for i, sel in enumerate(ALL_SELECTIONS):
            vals = [per_label[label].get(sel, 0) for label in labels_sorted]
            ax2.bar(labels_sorted, vals, width, bottom=bottoms, label=sel, color=colors[i])
            bottoms = [a + b for a, b in zip(bottoms, vals)]
        ax2.set_title("Decisions by Predicted Label")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(axis="y", linestyle=":", alpha=0.4)
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "per_label_stacked.png"), dpi=150)
        plt.close(fig2)

        # Accuracy chart per label (hard vs soft); CI bars for hard only
        if labels_sorted:
            x = list(range(len(labels_sorted)))
            hard = [r[5] for r in metrics_rows]
            soft = [r[6] for r in metrics_rows]
            ci_low = [r[7] for r in metrics_rows]
            ci_high = [r[8] for r in metrics_rows]
            hard_err = [[h - l for h, l in zip(hard, ci_low)], [h2 - h for h2, h in zip(ci_high, hard)]]

            fig3, ax3 = plt.subplots(figsize=(max(6, 0.45 * len(labels_sorted) + 2), 5))
            width = 0.38
            x_soft = [xi + width for xi in x]
            bars1 = ax3.bar(x, hard, width, label="Hard accuracy", color="#4caf50")
            ax3.errorbar(x, hard, yerr=hard_err, fmt="none", ecolor="#2e7d32", capsize=3, linewidth=1)
            bars2 = ax3.bar(x_soft, soft, width, label="Soft accuracy", color="#90caf9")
            ax3.set_xticks([xi + width / 2 for xi in x])
            ax3.set_xticklabels(labels_sorted, rotation=20, ha="right")
            ax3.set_ylim(0, 1)
            ax3.set_ylabel("Accuracy")
            ax3.set_title("Per-label Accuracy (hard vs soft)")
            ax3.legend()
            ax3.grid(axis="y", linestyle=":", alpha=0.4)
            fig3.tight_layout()
            fig3.savefig(os.path.join(out_dir, "per_label_accuracy.png"), dpi=150)
            plt.close(fig3)
            
            # F2 score chart per label (hard vs soft)
            hard_f2 = [r[9] for r in metrics_rows]
            soft_f2 = [r[10] for r in metrics_rows]
            
            fig4, ax4 = plt.subplots(figsize=(max(6, 0.45 * len(labels_sorted) + 2), 5))
            width = 0.38
            x_soft_f2 = [xi + width for xi in x]
            bars1 = ax4.bar(x, hard_f2, width, label="Hard F2", color="#8e24aa")
            bars2 = ax4.bar(x_soft_f2, soft_f2, width, label="Soft F2", color="#ce93d8")
            ax4.set_xticks([xi + width / 2 for xi in x])
            ax4.set_xticklabels(labels_sorted, rotation=20, ha="right")
            ax4.set_ylim(0, 1)
            ax4.set_ylabel("F2 Score")
            ax4.set_title("Per-label F2 Scores (Hard vs Soft)")
            ax4.legend()
            ax4.grid(axis="y", linestyle=":", alpha=0.4)
            fig4.tight_layout()
            fig4.savefig(os.path.join(out_dir, "per_label_f2_scores.png"), dpi=150)
            plt.close(fig4)

        # Simple metrics JSON (given available data)
        total = sum(by_sel.values())
        valid = by_sel.get(SELECTION_VALID, 0)
        similar = by_sel.get(SELECTION_SIMILAR, 0)
        incorrect = by_sel.get(SELECTION_INCORRECT, 0)
        soft_accuracy = (valid + 0.5 * similar) / total if total else 0.0
        hard_accuracy = valid / total if total else 0.0
        
        # Calculate overall F2 scores
        hard_f2_overall = 0.0
        soft_f2_overall = 0.0
        
        if valid + incorrect > 0:
            y_true_hard = [1] * valid + [0] * incorrect
            y_pred_hard = [1] * valid + [1] * incorrect
            if len(set(y_true_hard)) > 1:
                hard_f2_overall = fbeta_score(y_true_hard, y_pred_hard, beta=2, zero_division=0.0)
            elif valid > 0:
                hard_f2_overall = 1.0
        
        if valid + similar + incorrect > 0:
            y_true_soft = [1] * (valid + similar) + [0] * incorrect
            y_pred_soft = [1] * (valid + similar) + [1] * incorrect
            if len(set(y_true_soft)) > 1:
                soft_f2_overall = fbeta_score(y_true_soft, y_pred_soft, beta=2, zero_division=0.0)
            elif (valid + similar) > 0:
                soft_f2_overall = 1.0
        
        metrics = {
            "total": total,
            "valid": valid,
            "similar": similar,
            "incorrect": incorrect,
            "hard_accuracy": hard_accuracy,
            "soft_accuracy": soft_accuracy,
            "hard_f2_score": hard_f2_overall,
            "soft_f2_score": soft_f2_overall,
            "notes": "Hard = Valid vs Incorrect; Soft = (Valid+Similar) vs Incorrect. F2 score emphasizes recall over precision.",
        }
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # ----- Helpers -----
    def update_ui_state(self) -> None:
        has_progress = self.progress is not None
        has_item = has_progress and self.current_rel_path is not None
        for btn in (self.btn_valid, self.btn_similar, self.btn_incorrect):
            btn.setEnabled(bool(has_item))

        if self.progress:
            counts = self.progress.counts_by_selection()
            self.statusBar().showMessage(
                f"Valid: {counts.get(ALL_SELECTIONS[0], 0)} | "
                f"Similar: {counts.get(ALL_SELECTIONS[1], 0)} | "
                f"Incorrect: {counts.get(ALL_SELECTIONS[2], 0)}"
            )
        else:
            self.statusBar().clearMessage()

    def load_and_show_image(self, file_path: str) -> None:
        try:
            img = Image.open(file_path)
            img = img.convert("RGBA")
            qimg = self._pil_to_qimage(img)
            self.set_image(qimg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{file_path}\n{e}")
            self.set_image(None)

    def _pil_to_qimage(self, img: Image.Image) -> QImage:
        # Convert PIL Image to QImage (RGBA8888)
        data = img.tobytes("raw", "RGBA")
        self._current_qimage_bytes = data  # hold reference
        qimg = QImage(
            data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888
        )
        return qimg

    def set_image(self, qimg: Optional[QImage]) -> None:
        if qimg is None:
            self.image_label.clear()
            return
        pixmap = QPixmap.fromImage(qimg)
        # Scale to fit scroll area
        area_size = self.scroll_area.viewport().size()
        scaled = pixmap.scaled(
            area_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Rescale current image on window resize
        if self.image_label.pixmap():
            # Reload current image to rescale cleanly
            if self.progress and self.current_rel_path:
                abs_p = abs_path(self.progress.output_dir, self.current_rel_path)
                if os.path.exists(abs_p):
                    self.load_and_show_image(abs_p)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.unsaved_changes and self.progress and self.progress.reviewed > 0:
            resp = QMessageBox.question(
                self,
                "Save progress?",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if resp == QMessageBox.Cancel:
                event.ignore()
                return
            if resp == QMessageBox.Yes:
                self.on_save_progress()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
