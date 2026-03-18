#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from button_styles import style_button
from gpu_gui_runtime import gpu_binary_path, outputs_dir, project_root_path, temp_output_base


class GpuImageGui(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.binary = gpu_binary_path(project_root)
        self.schema = self._load_schema()
        self.scene_registry = self._load_scene_registry()
        self.process: QProcess | None = None
        self.render_start = 0.0
        self.latest_pixmap: QPixmap | None = None
        self.current_output: Path | None = None
        self.current_scene_name: str = self.schema.get('defaults', {}).get('scene', 'starter')

        self.setWindowTitle('SR-RT GPU Image Renderer')
        self.resize(1360, 860)
        self._build_ui()
        self._update_scene_details()

    def _load_schema(self) -> dict:
        if not self.binary.exists():
            return {
                'choices': {'scene': ['starter']},
                'defaults': {'scene': 'starter'},
                'render': {'width': 960, 'height': 540},
            }
        result = subprocess.run([str(self.binary), '--print-options-schema'], capture_output=True, text=True, cwd=self.project_root, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or 'Failed to query GPU renderer options schema')
        return json.loads(result.stdout)

    def _load_scene_registry(self) -> dict:
        if not self.binary.exists():
            return {'scenes': []}
        result = subprocess.run([str(self.binary), '--print-scene-registry'], capture_output=True, text=True, cwd=self.project_root, check=False)
        if result.returncode != 0:
            return {'scenes': []}
        return json.loads(result.stdout)

    def _scene_info(self, name: str) -> dict:
        for scene in self.scene_registry.get('scenes', []):
            if scene.get('name') == name:
                return scene
        return {'name': name, 'label': name, 'description': ''}

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        form = QFormLayout()

        self.scene_combo = QComboBox()
        for name in self.schema.get('choices', {}).get('scene', ['starter']):
            scene = self._scene_info(name)
            self.scene_combo.addItem(scene.get('label', name), userData=name)
        default_scene = self.schema.get('defaults', {}).get('scene', 'starter')
        for i in range(self.scene_combo.count()):
            if self.scene_combo.itemData(i) == default_scene:
                self.scene_combo.setCurrentIndex(i)
                break
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 8192)
        self.width_spin.setValue(int(self.schema.get('render', {}).get('width', 960)))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 8192)
        self.height_spin.setValue(int(self.schema.get('render', {}).get('height', 540)))

        form.addRow('Scene', self.scene_combo)
        form.addRow('Width', self.width_spin)
        form.addRow('Height', self.height_spin)
        left.addLayout(form)

        self.scene_detail = QLabel('')
        self.scene_detail.setWordWrap(True)
        left.addWidget(self.scene_detail)

        button_row = QHBoxLayout()
        self.render_btn = QPushButton('Render')
        style_button(self.render_btn, primary=True)
        self.render_btn.clicked.connect(self._on_render_clicked)
        self.save_btn = QPushButton('Save PNG')
        style_button(self.save_btn, primary=False)
        self.save_btn.clicked.connect(self._on_save_clicked)
        self.save_btn.hide()
        button_row.addWidget(self.render_btn)
        button_row.addWidget(self.save_btn)
        left.addLayout(button_row)

        self.status = QLabel('Idle')
        self.status.setWordWrap(True)
        left.addWidget(self.status)
        left.addStretch(1)
        layout.addLayout(left, 0)

        self.image_label = QLabel('GPU render output will appear here')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)
        layout.addWidget(self.image_scroll, 1)

    def _on_scene_changed(self) -> None:
        self.current_scene_name = str(self.scene_combo.currentData() or self.scene_combo.currentText())
        self._update_scene_details()

    def _update_scene_details(self) -> None:
        scene = self._scene_info(self.current_scene_name)
        counts = scene.get('primitive_counts', {})
        pieces = []
        if counts:
            pieces.append(f"spheres {counts.get('spheres', 0)}, planes {counts.get('planes', 0)}, triangles {counts.get('triangles', 0)}")
        desc = scene.get('description', '')
        text = desc
        if pieces:
            text = f"{desc}\nPrimitives: {pieces[0]}" if desc else f"Primitives: {pieces[0]}"
        self.scene_detail.setText(text)

    def _refresh_pixmap(self) -> None:
        if self.latest_pixmap is None:
            return
        viewport = self.image_scroll.viewport().size()
        if viewport.width() <= 1 or viewport.height() <= 1:
            self.image_label.setPixmap(self.latest_pixmap)
            return
        pix = self.latest_pixmap
        if pix.width() > viewport.width() or pix.height() > viewport.height():
            pix = pix.scaled(viewport, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(pix)
        self.image_label.adjustSize()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _save_output_path(self) -> Path:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        base = outputs_dir(self.project_root) / f"{self.current_scene_name}_{stamp}.png"
        suffix = 1
        path = base
        while path.exists():
            path = outputs_dir(self.project_root) / f"{self.current_scene_name}_{stamp}_{suffix}.png"
            suffix += 1
        return path

    def _on_save_clicked(self) -> None:
        if self.latest_pixmap is None or self.latest_pixmap.isNull():
            QMessageBox.warning(self, 'Nothing To Save', 'Render an image first.')
            return
        path = self._save_output_path()
        if not self.latest_pixmap.save(str(path), 'PNG'):
            QMessageBox.warning(self, 'Save Failed', f'Could not save PNG to:\n{path}')
            return
        self.status.setText(f'Saved PNG to {path}')

    def _on_render_clicked(self) -> None:
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            return
        if not self.binary.exists():
            QMessageBox.critical(self, 'Missing Binary', f'Could not find: {self.binary}')
            return

        self.current_scene_name = str(self.scene_combo.currentData() or self.scene_combo.currentText())
        self.current_output = temp_output_base().with_suffix('.ppm')
        self.current_output.parent.mkdir(parents=True, exist_ok=True)
        self.save_btn.hide()

        args = ['--scene', self.current_scene_name, '--width', str(self.width_spin.value()), '--height', str(self.height_spin.value()), '--output', str(self.current_output)]
        self.process = QProcess(self)
        self.process.setProgram(str(self.binary))
        self.process.setArguments(args)
        self.process.setWorkingDirectory(str(self.project_root))
        self.process.readyReadStandardError.connect(self._on_output)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)
        self.process.start()
        self.render_start = time.time()
        self.render_btn.setText('Cancel')
        self.status.setText('Rendering...')

    def _on_output(self) -> None:
        if not self.process:
            return
        out = bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='ignore')
        err = bytes(self.process.readAllStandardError()).decode('utf-8', errors='ignore')
        text = (out + '\n' + err).strip()
        if text:
            self.status.setText(text.splitlines()[-1])

    def _on_finished(self, exit_code: int) -> None:
        self.render_btn.setText('Render')
        if exit_code != 0 or self.current_output is None or not self.current_output.exists():
            QMessageBox.warning(self, 'Render Failed', 'GPU render failed.')
            self.status.setText(f'Render failed (exit {exit_code})')
            return
        pix = QPixmap(str(self.current_output))
        if pix.isNull():
            self.status.setText(f'Render finished, but failed to load image: {self.current_output}')
            return
        self.latest_pixmap = pix
        self.save_btn.show()
        self._refresh_pixmap()
        self.status.setText(f'Done in {time.time() - self.render_start:.2f}s | {self.current_output}')


def main() -> int:
    app = QApplication(sys.argv)
    win = GpuImageGui(project_root_path(Path(__file__)))
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
