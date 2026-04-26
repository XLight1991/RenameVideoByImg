import os
import sys
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import open_clip

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QMessageBox,
    QProgressBar,
    QGroupBox,
    QFormLayout,
)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
VIDEO_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm'}


def is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def list_files(folder: str, kind: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if kind == 'image' and is_image_file(path):
            files.append(path)
        elif kind == 'video' and is_video_file(path):
            files.append(path)
    return sorted(files)


def sanitize_filename(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, '_')
    return name.strip()


def ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    folder = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    i = 1
    while True:
        candidate = os.path.join(folder, f'{stem}_{i}{ext}')
        if not os.path.exists(candidate):
            return candidate
        i += 1


def crop_bottom(img: Image.Image, crop_bottom_ratio: float) -> Image.Image:
    if crop_bottom_ratio <= 0:
        return img
    w, h = img.size
    new_h = max(1, int(h * (1 - crop_bottom_ratio)))
    return img.crop((0, 0, w, new_h))


def pil_open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def extract_video_frames(video_path: str, num_frames: int, crop_bottom_ratio: float) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {video_path}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int).tolist()
    else:
        indices = None

    frames = []
    if indices is not None:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frames.append(crop_bottom(img, crop_bottom_ratio))
    else:
        tries = 0
        while len(frames) < num_frames and tries < 120:
            ok, frame = cap.read()
            tries += 1
            if not ok or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            frames.append(crop_bottom(img, crop_bottom_ratio))

    cap.release()

    if not frames:
        raise RuntimeError(f'视频未能抽取到帧: {video_path}')

    return frames


class ClipMatcher:
    def __init__(self, crop_bottom_ratio: float = 0.12):
        self.crop_bottom_ratio = crop_bottom_ratio
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model.eval().to(self.device)
        self.image_paths: List[str] = []
        self.image_feats: Optional[torch.Tensor] = None

    @torch.no_grad()
    def encode_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img) for img in pil_images]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def build_image_index(self, image_paths: List[str]) -> None:
        feats = []
        kept_paths = []
        for path in image_paths:
            try:
                img = crop_bottom(pil_open_rgb(path), self.crop_bottom_ratio)
                feat = self.encode_images([img])[0].cpu()
                feats.append(feat)
                kept_paths.append(path)
            except Exception:
                continue
        if not feats:
            raise RuntimeError('没有成功编码任何图片。')
        self.image_paths = kept_paths
        self.image_feats = torch.stack(feats, dim=0)

    def match_video(self, video_path: str, frames: int) -> Tuple[str, float, List[Tuple[str, float]]]:
        if self.image_feats is None or not self.image_paths:
            raise RuntimeError('请先建立图片索引。')
        pil_frames = extract_video_frames(video_path, frames, self.crop_bottom_ratio)
        frame_feats = self.encode_images(pil_frames).cpu()
        sims = frame_feats @ self.image_feats.T
        mean_sims = sims.mean(dim=0)
        best_idx = int(torch.argmax(mean_sims).item())
        best_score = float(mean_sims[best_idx].item())
        topk = min(3, mean_sims.shape[0])
        vals, idxs = torch.topk(mean_sims, k=topk)
        top_matches = [(self.image_paths[int(i)], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]
        return self.image_paths[best_idx], best_score, top_matches


@dataclass
class Config:
    image_dir: str
    video_dir: str
    frames: int
    crop_bottom_ratio: float
    threshold: float
    dry_run: bool
    copy_instead_of_rename: bool


class Worker(QThread):
    log = Signal(str)
    progress = Signal(int, int)
    done = Signal()
    failed = Signal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def run(self):
        try:
            image_paths = list_files(self.config.image_dir, 'image')
            video_paths = list_files(self.config.video_dir, 'video')

            if not image_paths:
                raise RuntimeError('图片文件夹里没有找到图片。')
            if not video_paths:
                raise RuntimeError('视频文件夹里没有找到视频。')

            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.log.emit(f'使用设备: {device_name}')
            self.log.emit(f'找到 {len(image_paths)} 张图片，{len(video_paths)} 个视频')
            self.log.emit('正在加载 CLIP 模型，请稍候...')

            matcher = ClipMatcher(crop_bottom_ratio=self.config.crop_bottom_ratio)
            self.log.emit('正在编码图片...')
            matcher.build_image_index(image_paths)

            total = len(video_paths)
            renamed = 0
            skipped = 0

            for idx, video_path in enumerate(video_paths, start=1):
                self.progress.emit(idx, total)
                self.log.emit('')
                self.log.emit(f'[{idx}/{total}] 处理视频: {os.path.basename(video_path)}')
                try:
                    best_img, score, top_matches = matcher.match_video(video_path, self.config.frames)
                except Exception as e:
                    self.log.emit(f'  失败: {e}')
                    skipped += 1
                    continue

                img_stem = os.path.splitext(os.path.basename(best_img))[0]
                video_ext = os.path.splitext(video_path)[1]
                target_name = sanitize_filename(img_stem) + video_ext
                target_path = ensure_unique_path(os.path.join(os.path.dirname(video_path), target_name))

                self.log.emit(f'  最佳匹配: {os.path.basename(best_img)}')
                self.log.emit(f'  相似度: {score:.4f}')
                self.log.emit('  Top3:')
                for p, s in top_matches:
                    self.log.emit(f'    - {os.path.basename(p)} : {s:.4f}')

                if score < self.config.threshold:
                    self.log.emit('  -> 相似度低于阈值，跳过自动重命名')
                    skipped += 1
                    continue

                if self.config.dry_run:
                    self.log.emit(f'  -> 预览模式，将改名为: {os.path.basename(target_path)}')
                    continue

                if self.config.copy_instead_of_rename:
                    shutil.copy2(video_path, target_path)
                    self.log.emit(f'  -> 已复制为: {os.path.basename(target_path)}')
                else:
                    os.rename(video_path, target_path)
                    self.log.emit(f'  -> 已重命名为: {os.path.basename(target_path)}')
                renamed += 1

            self.log.emit('')
            self.log.emit(f'完成。成功处理: {renamed}，跳过: {skipped}，总计: {total}')
            self.done.emit()
        except Exception as e:
            self.failed.emit(str(e))


class DropLineEdit(QLineEdit):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setAcceptDrops(True)
        self.setPlaceholderText(f'可拖拽{title}文件夹到这里，或点击右侧按钮选择')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                if os.path.isdir(path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if os.path.isdir(path):
                self.setText(path)
                event.acceptProposedAction()
                return
        event.ignore()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle('CLIP 视频重命名助手')
        self.resize(860, 680)
        self.init_ui()

    def init_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            '把图片文件夹和视频文件夹拖进来。程序会用 CLIP 比较视频画面和图片内容，\n'
            '自动把视频改成对应的图片名。适合人物轻微说话、轻微位置变化、底部有字幕的场景。'
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        root.addWidget(self._build_path_box('图片文件夹', is_image=True))
        root.addWidget(self._build_path_box('视频文件夹', is_image=False))

        settings_box = QGroupBox('参数设置')
        settings_form = QFormLayout(settings_box)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 20)
        self.frames_spin.setValue(1)
        settings_form.addRow('每个视频抽帧数', self.frames_spin)

        self.crop_spin = QDoubleSpinBox()
        self.crop_spin.setRange(0.0, 0.5)
        self.crop_spin.setSingleStep(0.01)
        self.crop_spin.setDecimals(2)
        self.crop_spin.setValue(0.12)
        settings_form.addRow('裁掉底部字幕比例', self.crop_spin)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.75)
        settings_form.addRow('自动改名阈值', self.threshold_spin)

        self.dry_run_check = QCheckBox('仅预览，不真正改名')
        self.dry_run_check.setChecked(True)
        settings_form.addRow(self.dry_run_check)

        self.copy_check = QCheckBox('复制出新文件，不改原视频名')
        settings_form.addRow(self.copy_check)

        root.addWidget(settings_box)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton('开始匹配')
        self.start_btn.clicked.connect(self.start_work)
        self.clear_btn = QPushButton('清空日志')
        self.clear_btn.clicked.connect(lambda: self.log_edit.clear())
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        root.addWidget(self.log_edit, 1)

    def _build_path_box(self, title: str, is_image: bool) -> QGroupBox:
        box = QGroupBox(title)
        layout = QHBoxLayout(box)
        line = DropLineEdit(title)
        button = QPushButton('选择文件夹')
        button.clicked.connect(lambda: self.choose_folder(line))
        layout.addWidget(line, 1)
        layout.addWidget(button)
        if is_image:
            self.image_dir_edit = line
        else:
            self.video_dir_edit = line
        return box

    def choose_folder(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if path:
            line_edit.setText(path)

    def append_log(self, text: str):
        self.log_edit.append(text)

    def start_work(self):
        image_dir = self.image_dir_edit.text().strip()
        video_dir = self.video_dir_edit.text().strip()

        if not os.path.isdir(image_dir):
            QMessageBox.warning(self, '提示', '请选择有效的图片文件夹。')
            return
        if not os.path.isdir(video_dir):
            QMessageBox.warning(self, '提示', '请选择有效的视频文件夹。')
            return

        config = Config(
            image_dir=image_dir,
            video_dir=video_dir,
            frames=self.frames_spin.value(),
            crop_bottom_ratio=self.crop_spin.value(),
            threshold=self.threshold_spin.value(),
            dry_run=self.dry_run_check.isChecked(),
            copy_instead_of_rename=self.copy_check.isChecked(),
        )

        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.append_log('开始任务...')

        self.worker = Worker(config)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.on_progress)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_progress(self, current: int, total: int):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def on_done(self):
        self.start_btn.setEnabled(True)
        QMessageBox.information(self, '完成', '处理完成。请查看下方日志。')

    def on_failed(self, msg: str):
        self.start_btn.setEnabled(True)
        self.append_log(f'\n任务失败: {msg}')
        QMessageBox.critical(self, '错误', msg)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
