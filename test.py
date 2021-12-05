from bisect import bisect
from enum import Enum
import sys

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from PIL import Image, ImageQt, ImageFilter, ImageDraw


class Overlay:
    Nothing = 0
    Transparent = 1
    Red = 2
    Blue = 3
    Green = 4
    Fixed = 5

    @classmethod
    def color(cls, value):
        return {
            Overlay.Transparent: (0, 0, 0, 0),
            Overlay.Red: (255, 0, 0, 128),
            Overlay.Green: (0, 255, 0, 128),
            Overlay.Blue: (0, 0, 255, 128),
            Overlay.Fixed: (0, 0, 0, 128),
        }[value]


def draw_hex(drawer, x, y, r, dy, color):
    w = 3
    points = [
        (x + r, y - dy),
        (x, y - 2*dy),
        (x - r, y - dy),
        (x - r, y - dy),
        (x - r, y + dy),
        (x, y + 2*dy),
        (x + r, y + dy),
        (x + r, y - dy),
        (x, y - 2*dy),
    ]
    drawer.polygon(points[:-2], fill=color)
    drawer.line(points, 'white', width=w, joint='curve')


class ImageMask:

    def __init__(self, path, blur=0.0, default=Overlay.Transparent):
        self.image = image = Image.open(path)

        blur_size = blur * max(image.size)
        self.blurred_image = image.filter(ImageFilter.GaussianBlur(blur_size))

        width, height = image.size
        self.radius = radius = width / 10 / 2
        self.delta = delta = radius / np.sqrt(3)

        self.ex_xpts = ex_xpts = np.linspace(0, width, 11)
        self.in_xpts = (ex_xpts[:-1] + ex_xpts[1:]) / 2
        ny = int(np.ceil(height / 6 / delta))
        self.ypts = ypts = np.arange(-ny, ny+1) * 3 * delta + height / 2

        self.vis = np.ones((len(ex_xpts), len(ypts)), dtype=int) * default

    def make_image(self):
        mask = Image.new('RGBA', self.image.size)
        mask.paste(self.image, (0, 0))
        drawer = ImageDraw.Draw(mask)

        for iy, y in enumerate(self.ypts):
            xs = self.ex_xpts if iy % 2 == 0 else self.in_xpts
            for ix, x in enumerate(xs):
                if self.vis[ix, iy] != Overlay.Nothing:
                    draw_hex(drawer, x, y, self.radius, self.delta, Overlay.color(self.vis[ix, iy]))

        image = self.blurred_image.copy()
        image.paste(mask, (0, 0), mask)
        self.qimage = ImageQt.ImageQt(image)
        return QPixmap.fromImage(self.qimage)

    def closest_in_row(self, row, x, y):
        xs = self.ex_xpts if row % 2 == 0 else self.in_xpts
        i = np.argmin(np.abs(xs - x))
        dist = np.linalg.norm([xs[i] - x, self.ypts[row] - y])
        return (i, row), dist

    def modify(self, relx, rely, value):
        if not 0 <= relx <= 1 or not 0 <= rely <= 1:
            return False

        width, height = self.image.size
        xc = relx * width
        yc = rely * height

        iy = bisect(self.ypts, yc)

        blk, dist = self.closest_in_row(iy, xc, yc)
        if iy > 1:
            alt_blk, alt_dist = self.closest_in_row(iy - 1, xc, yc)
            if alt_dist < dist:
                blk = alt_blk

        i, j = blk
        if self.vis[i, j] != int(value) and self.vis[i, j] != Overlay.Fixed:
            self.vis[i, j] = int(value)
            return True

        return False


class ImageView(QLabel):

    def __init__(self, masker):
        super().__init__()
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.Alignment(0x84))
        self.masker = masker
        self.orig_pixmap = masker.make_image()

    def resize(self):
        pixmap = self.orig_pixmap.scaled(self.width(), self.height(), 1, 1)
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        self.resize()

    def reveal(self, x, y, value):
        pixmap = self.pixmap()
        px = self.width() - pixmap.width()
        py = self.height() - pixmap.height()
        relx = (x - px / 2) / pixmap.width()
        rely = (y - py / 2) / pixmap.height()

        changed = self.masker.modify(relx, rely, value)
        if changed:
            self.orig_pixmap = self.masker.make_image()
            self.resize()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.reveal(event.x(), event.y(), Overlay.Red)

    def mousePressEvent(self, event):
        self.reveal(event.x(), event.y(), Overlay.Red)


class MainWidget(QWidget):

    def __init__(self):
        super().__init__()

        self.image = ImageView(ImageMask(sys.argv[1], blur=40/1000))

        self.label = QLabel()
        self.label.setMaximumHeight(25)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
        color: rgb(200, 200, 200);
        font-size: 20px;
        font-weight: bold;
        """)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.image)
        self.layout().addWidget(self.label)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Margarine')
        self.setStyleSheet('background-color: black;')
        self.main = MainWidget()
        self.setCentralWidget(self.main)


app = QApplication(sys.argv)
main = MainWindow()
main.showMaximized()
app.exec_()
