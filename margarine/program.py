from pathlib import Path
import random
import shutil
import uuid

from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np

from .config import config
from .image import Overlay, Picture


class DatabaseProgram(QObject):

    new_masker = pyqtSignal(object)
    new_message = pyqtSignal(str)
    new_flash = pyqtSignal(str)
    must_exit = pyqtSignal()

    def __init__(self, lowest_level=0, highest_level=-1):
        super().__init__()
        self.lowest_level = lowest_level
        self.highest_level = highest_level
        self.proceed_chance = config.advance_prob

    def next_image(self):
        image = Picture.from_db()
        image.mark_seen()
        masker = image.make_masker(
            blur=config.blur,
            default=Overlay.Transparent,
            target=Overlay.Nothing,
            require_privilege=True,
        )
        masker.assign(*np.where(image.mask <= self.lowest_level), Overlay.Nothing, privilege=True)
        self.new_masker.emit(masker)
        self.masker = masker
        self.mask = image.mask.copy()
        self.new_message.emit(f"ID {image.ident}")

        if random.random() < config.message_prob / 100:
            self.new_flash.emit(random.choice(config.messages))
        else:
            self.new_flash.emit(None)

    def start(self):
        self.next_image()

    def key(self, text):
        if text == 'b':
            self.reveal_random()
        elif text == 'C-e':
            self.eject()
            self.next_image()
        else:
            self.next_image()

    def eject(self):
        name = f'ejected-{uuid.uuid1()}.png'
        self.masker.image.save(name, 'PNG')

    def reveal_random(self):
        mask = self.mask > self.lowest_level
        if self.highest_level >= 0:
            mask &= self.mask <= self.highest_level
        ncells = np.sum(mask)
        proc_chance = self.proceed_chance * np.min(self.mask[self.mask > self.lowest_level])
        if random.random() * (ncells + proc_chance) <= proc_chance:
            self.next_image()
            return

        try:
            lowest = np.min(self.mask[self.mask > self.lowest_level])
        except ValueError:
            return
        if self.highest_level >= 0 and lowest > self.highest_level:
            return
        index = random.choice(*np.where(self.mask == lowest))
        self.mask[index] = 0
        self.masker.assign(index, Overlay.Nothing, privilege=True)
        self.new_masker.emit(self.masker)


class ClassifyProgram(QObject):

    new_masker = pyqtSignal(object)
    new_message = pyqtSignal(str)
    new_flash = pyqtSignal(str)
    must_exit = pyqtSignal()

    def __init__(self, filenames):
        super().__init__()
        self.filenames = iter(filenames)
        self.levels = config.block_levels

    def start(self):
        self.mode = 'classifying'
        self.next_candidate()

    def key(self, text):
        if self.mode == 'classifying':
            if text == 'i':
                self.masker.invert(Overlay.Transparent, Overlay.Red)
                self.new_masker.emit(self.masker)
            elif text == 'x':
                self.next_candidate()
            elif text == 'C-x':
                self.dispatch_image('skipped')
                self.next_candidate()
            else:
                self.next_level()
        elif self.mode == 'collisions' and text != 'x':
            self.next_collision(text)

    def next_collision(self, text):
        if text == 'a':
            self.start_classification()
            return

        if text == 'r':
            self.next_candidate()
            return

        if self.collisions:
            pic, rest = self.collisions[0], self.collisions[1:]
            self.new_masker.emit(pic.make_masker(default=Overlay.Nothing))
            self.new_message.emit(f"Potential collisions: {len(rest)} remaining (A: accept, R: reject)")
            self.collisions = rest

    def next_candidate(self):
        try:
            filename = next(self.filenames)
        except StopIteration:
            self.must_exit.emit()
            return

        self.picture = Picture.from_file(filename)

        idents = list(self.picture.find_colliding_idents())
        if idents:
            self.collisions = [Picture.from_db(ident) for ident in idents]
            self.mode = 'collisions'
            self.new_masker.emit(self.picture.make_masker(default=Overlay.Nothing))
            self.new_message.emit(f"Potential collisions: candidate image (A: accept, R: reject)")
            return

        self.start_classification()

    def start_classification(self):
        self.masker = self.picture.make_masker(
            default=Overlay.Transparent,
            target=Overlay.Red,
            init_mask=True
        )

        self.mode = 'classifying'
        self.current_level = 0
        self.next_level()

    def next_level(self):
        if self.current_level > 0:
            self.picture.mask[self.masker.marked()] = self.current_level
            self.masker.convert(invert=True)
            self.masker.convert(Overlay.Transparent)

        self.current_level += 1

        if self.current_level < len(self.levels):
            self.new_message.emit(f"{self.picture.filename} - select: {self.levels[self.current_level]}")
            self.new_masker.emit(self.masker)
        else:
            self.commit_to_db()
            self.next_candidate()

    def dispatch_image(self, target='done'):
        tgt = Path(self.picture.filename).parent / f'.margarine-{target}'
        tgt.mkdir(exist_ok=True, parents=True)
        try:
            shutil.move(self.picture.filename, tgt)
        except shutil.Error:
            pass

    def commit_to_db(self):
        self.dispatch_image()
        self.picture.save_to_db()
