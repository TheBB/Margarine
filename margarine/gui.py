from string import ascii_lowercase
import sys

import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


KEY_MAP = {
    Qt.Key_Space: 'SPC',
    Qt.Key_Escape: 'ESC',
    Qt.Key_Tab: 'TAB',
    Qt.Key_Return: 'RET',
    Qt.Key_Backspace: 'BSP',
    Qt.Key_Delete: 'DEL',
    Qt.Key_Up: 'UP',
    Qt.Key_Down: 'DOWN',
    Qt.Key_Left: 'LEFT',
    Qt.Key_Right: 'RIGHT',
    Qt.Key_Minus: '-',
    Qt.Key_Plus: '+',
    Qt.Key_Equal: '=',
}
KEY_MAP.update({
    getattr(Qt, 'Key_{}'.format(s.upper())): s
    for s in ascii_lowercase
})


def key_to_text(event):
    ctrl = event.modifiers() & Qt.ControlModifier
    shift = event.modifiers() & Qt.ShiftModifier

    try:
        text = KEY_MAP[event.key()]
    except KeyError:
        return

    if shift and text.isupper():
        text = 'S-{}'.format(text)
    elif shift:
        text = text.upper()
    if ctrl:
        text = 'C-{}'.format(text)

    return text


class ImageView(QLabel):

    def __init__(self, parent, masker=None):
        super().__init__()
        self.setMinimumSize(1, 1)
        self.setAlignment(Qt.Alignment(0x84))
        self.masker = masker
        self.parent = parent

        if masker:
            self.orig_pixmap = masker.make_image()
        else:
            self.orig_pixmap = None

    def set_masker(self, masker):
        self.masker = masker
        self.orig_pixmap = masker.make_image()
        self.resize()

    def resize(self):
        if self.orig_pixmap:
            pixmap = self.orig_pixmap.scaled(self.width(), self.height(), 1, 1)
            self.setPixmap(pixmap)

    def resizeEvent(self, event):
        self.resize()

    def relative(self, event):
        pixmap = self.pixmap()
        px = self.width() - pixmap.width()
        py = self.height() - pixmap.height()
        relx = (event.x() - px / 2) / pixmap.width()
        rely = (event.y() - py / 2) / pixmap.height()
        return relx, rely


    # def reveal(self, x, y, method='modify'):
    #     if not self.masker:
    #         pass

    #     pixmap = self.pixmap()
    #     px = self.width() - pixmap.width()
    #     py = self.height() - pixmap.height()
    #     relx = (x - px / 2) / pixmap.width()
    #     rely = (y - py / 2) / pixmap.height()

    #     changed = getattr(self.masker, method)(relx, rely)
    #     if changed:
    #         self.orig_pixmap = self.masker.make_image()
    #         self.resize()

    def mouseMoveEvent(self, event):
        self.parent.program.mouse(*self.relative(event), event.buttons() & Qt.LeftButton)

    def mousePressEvent(self, event):
        self.parent.program.mouse(*self.relative(event), event.buttons() & Qt.LeftButton)


class MainWidget(QWidget):

    def __init__(self, program):
        super().__init__()

        self.stack = []
        self.image = ImageView(self)

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

        self.timer = QTimer(self)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(lambda: self.program.key('b'))

        self.overlay = QLabel(self)
        self.overlay.setFrameStyle(Qt.FramelessWindowHint)
        self.overlay.setVisible(False)
        self.overlay.setWordWrap(True)
        self.overlay.setStyleSheet("""
        background-color: rgba(0,0,0,0.7);
        color: rgba(200,200,200,1);
        font-size: 35px;
        font-weight: bold;
        """)

        self.launch_program(program)

    @property
    def program(self):
        return self.stack[-1]

    def launch_program(self, program):
        self.stack.append(program)
        program.new_masker.connect(self.new_masker)
        program.new_message.connect(self.new_message)
        program.new_flash.connect(self.new_flash)
        program.launch_subprogram.connect(self.launch_program)
        program.dispatch_program.connect(self.dispatch_program)
        program.start()

    def dispatch_program(self, result):
        self.stack.pop()
        if self.stack:
            self.stack[-1].restart(result)
        else:
            QApplication.instance().quit()

    def resize(self):
        self.overlay.setGeometry(0, self.height()//2 - 150, self.width(), 300)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize()

    def new_masker(self, masker):
        self.image.set_masker(masker)

    def new_message(self, message):
        self.label.setText(message)

    def new_flash(self, flash):
        if flash:
            self.overlay.setText(f'<div align="center">{flash}</div>')
            self.overlay.setVisible(True)
        else:
            self.overlay.setVisible(False)

    def key(self, text):
        if text == 'C-t':
            if self.timer.isActive():
                self.timer.stop()
            else:
                self.timer.start(350)
            return
        self.program.key(text)


class MainWindow(QMainWindow):

    def __init__(self, program):
        super().__init__()
        self.setWindowTitle('Margarine')
        self.setStyleSheet('background-color: black;')
        self.main = MainWidget(program)
        self.setCentralWidget(self.main)

    def keyPressEvent(self, event):
        text = key_to_text(event)
        if text is None:
            return
        self.main.key(text)


def run_gui(program):
    app = QApplication(sys.argv)
    main = MainWindow(program)
    main.showMaximized()
    app.exec_()
