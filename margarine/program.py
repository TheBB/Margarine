from pathlib import Path
import random
import shutil
import uuid

from typing import List

from PyQt5.QtCore import pyqtSignal, QObject
import numpy as np

from .config import config
from .image import MaskManager, Picture, BlurMask, MaskPicker, PureMask


def shuffle(nums):
    nums = list(nums)
    random.shuffle(nums)
    return nums


class Program(QObject):

    new_masker = pyqtSignal(object)
    new_message = pyqtSignal(str)
    new_flash = pyqtSignal(str)
    launch_subprogram = pyqtSignal(object)
    dispatch_program = pyqtSignal(object)

    def start(self):
        pass

    def restart(self, rval):
        pass

    def key(self, text):
        pass

    def mouse(self, relx, rely, left):
        pass

    def _close(self, retval=None):
        self.dispatch_program.emit(retval)


class RevealPlan:

    initial: List[int]
    plan: List[int]
    index: int

    def __init__(self, image: Picture, manager: MaskManager, lo: int, hi: int):
        self.initial, = list(np.where(image.mask <= lo))

        hi = np.max(image.mask) if hi == -1 else hi
        to_uncover = set(np.where(image.mask > lo)[0])

        plan = []
        while to_uncover:
            boundary = manager.boundary_of(to_uncover)
            minval = min(image.mask[k] for k in boundary)
            if minval > hi:
                break
            indexes = {k for k in boundary if image.mask[k] == minval}
            plan.extend(shuffle(indexes))
            to_uncover -= indexes

        self.plan = plan
        self.index = 0

    def __next__(self):
        if self.index >= len(self.plan):
            raise StopIteration
        retval = self.plan[self.index]
        self.index += 1
        return retval

    def __len__(self):
        return len(self.plan) - self.index


class DisplayImageProgram(Program):

    def __init__(self, picture: Picture, lo: int = 0, hi: int = -1):
        super().__init__()
        self.picture = picture
        manager = picture.mask_manager()
        self.plan = RevealPlan(picture, manager, lo, hi)
        self.mask = BlurMask(manager, blur=config.blur)
        self.mask.uncover(self.plan.initial, privilege=True)

    def start(self):
        self.new_masker.emit(self.mask)
        self.new_message.emit(f"ID {self.picture.ident}")

        if random.random() < config.message_prob / 100:
            self.new_flash.emit(random.choice(config.messages))
        else:
            self.new_flash.emit(None)

    def key(self, text):
        if text == 'b':
            self.uncover()
        elif text == 'r':
            self.picture.revisit = True
            self.picture.save()
        else:
            self._close()

    def uncover(self):
        try:
            index = next(self.plan)
        except StopIteration:
            return

        self.mask.uncover(index, privilege=True)
        self.new_masker.emit(self.mask)


class DatabaseProgram(Program):

    def __init__(self, lowest_level=0, highest_level=-1):
        super().__init__()
        self.lowest_level = lowest_level
        self.highest_level = highest_level

    def next_image(self):
        picture = Picture.from_db()
        picture.mark_seen()
        self.launch_subprogram.emit(DisplayImageProgram(picture, self.lowest_level, self.highest_level))

    def start(self):
        self.next_image()

    def restart(self, _):
        self.next_image()


class CollisionProgram(Program):

    def __init__(self, picture, collisions):
        super().__init__()
        self.pictures = [picture, *collisions]
        self.labels = ['Original picture', *(f'Collision #{i+1}/{len(collisions)}' for i in range(len(collisions)))]
        self.next_index = 0

    def key(self, text):
        if text in ('LEFT', 'BSP'):
            self.previous_image()
        elif text == 'r':
            self.pictures[0].dispatch('.margarine-skipped', relative=True, as_dir=True)
            self._close(False)
        elif text == 'a':
            self._close(True)
        else:
            self.next_image()

    def start(self):
        self.emit_image()

    def next_image(self):
        if self.next_index < len(self.pictures) - 1:
            self.next_index += 1
            self.emit_image()

    def previous_image(self):
        if self.next_index > 0:
            self.next_index -= 1
            self.emit_image()

    def emit_image(self):
        self.new_masker.emit(PureMask(self.pictures[self.next_index]))
        self.new_message.emit(f'{self.labels[self.next_index]} - (R) reject (A) accept')


class ClassifyImageProgram(Program):

    def __init__(self, filename, levels):
        super().__init__()
        self.levels = levels
        self.picture = Picture.from_file(filename)
        self.manager = self.picture.mask_manager()
        self.mask = self.manager.zeros(dtype=int)
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= 0)[0]))
        self.current_level = 1

    def start(self):
        collisions = list(self.picture.find_colliding_pictures())
        if collisions:
            self.launch_subprogram.emit(CollisionProgram(self.picture, collisions))
        else:
            self.start_classification()

    def restart(self, approved):
        if not approved:
            self._close()
            return
        self.start_classification()

    def start_classification(self):
        self.new_masker.emit(self.masker)
        self.new_message.emit(f"{self.picture.filename} - select: {self.levels[self.current_level]}")

    def next_level(self):
        self.mask[self.masker.marked] = self.current_level
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= self.current_level)[0]))
        self.current_level += 1

        if self.current_level < len(self.levels):
            self.new_message.emit(f"{self.picture.filename} - select: {self.levels[self.current_level]}")
            self.new_masker.emit(self.masker)
        else:
            self.commit_to_db()
            self._close()

    def undo(self):
        if self.current_level < 2:
            return

        self.current_level -= 1

        l = self.current_level - 1
        was_selected = np.where(self.mask > l)
        self.mask[np.where(self.mask >= l)] = l
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= l)[0]))
        self.masker.vis[was_selected] = 2
        self.new_message.emit(f"{self.picture.filename} - select: {self.levels[self.current_level]}")
        self.new_masker.emit(self.masker)

    def mouse(self, relx, rely, left):
        value = 2 if left else 1
        if self.masker.modify(relx, rely, value=value):
            self.new_masker.emit(self.masker)

    def key(self, text):
        if text == 'i':
            self.masker.invert()
            self.new_masker.emit(self.masker)
        elif text == 'x':
            self._close()
        elif text == 'C-x':
            self.picture.dispatch('.margarine-skipped', relative=True, as_dir=True)
            self._close()
        elif text == 'u':
            self.undo()
        else:
            self.next_level()

    def commit_to_db(self):
        self.picture.dispatch('.margarine-done', relative=True, as_dir=True)

        self.picture.mask_nx = self.manager.nx
        self.picture.mask_ny = self.manager.ny
        self.picture.mask_rfac = self.manager.rfac
        self.picture._mask = ','.join(map(str, self.mask))
        self.picture.save()


class RevisitImageProgram(Program):

    def __init__(self, picture, levels):
        super().__init__()
        self.levels = levels
        self.picture = picture
        self.orig_mask = picture.mask
        self.manager = self.picture.mask_manager()
        self.mask = self.manager.zeros(dtype=int)
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= 0)[0]))
        self.masker.vis[(self.orig_mask > 0) & (self.mask >= 0)] = 2
        self.current_level = 1

    def start(self):
        self.new_masker.emit(self.masker)
        self.new_message.emit(f"{self.picture.ident} - select: {self.levels[self.current_level]}")

    def next_level(self):
        self.mask[self.masker.marked] = self.current_level
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= self.current_level)[0]))
        self.masker.vis[(self.orig_mask > self.current_level) & (self.mask >= self.current_level)] = 2
        self.current_level += 1

        if self.current_level < len(self.levels):
            self.new_message.emit(f"{self.picture.ident} - select: {self.levels[self.current_level]}")
            self.new_masker.emit(self.masker)
        else:
            self.commit_to_db()
            self._close()

    def undo(self):
        if self.current_level < 2:
            return

        self.current_level -= 1

        l = self.current_level - 1
        was_selected = np.where(self.mask > l)
        self.mask[np.where(self.mask >= l)] = l
        self.masker = MaskPicker(self.manager, set(np.where(self.mask >= l)[0]))
        self.masker.vis[was_selected] = 2
        self.new_message.emit(f"{self.picture.filename} - select: {self.levels[self.current_level]}")
        self.new_masker.emit(self.masker)

    def mouse(self, relx, rely, left):
        value = 2 if left else 1
        if self.masker.modify(relx, rely, value=value):
            self.new_masker.emit(self.masker)

    def key(self, text):
        if text == 'i':
            self.masker.invert()
            self.new_masker.emit(self.masker)
        elif text == 'C-x':
            self._close()
        elif text == 'u':
            self.undo()
        else:
            self.next_level()

    def commit_to_db(self):
        self.picture._mask = ','.join(map(str, self.mask))
        self.picture.revisit = False
        self.picture.save()


class ClassifyProgram(Program):

    def __init__(self, filenames):
        super().__init__()
        self.filenames = iter(filenames)
        self.levels = config.block_levels

    def next_image(self):
        try:
            filename = next(self.filenames)
        except StopIteration:
            self._close()
            return
        self.launch_subprogram.emit(ClassifyImageProgram(filename, config.block_levels))

    def start(self):
        self.next_image()

    def restart(self, _):
        self.next_image()


class RevisitProgram(Program):

    def __init__(self):
        super().__init__()

    def next_image(self):
        image = Picture.get_revisit()
        if image is None:
            self._close()
            return
        self.launch_subprogram.emit(RevisitImageProgram(image, config.block_levels))

    def start(self):
        self.next_image()

    def restart(self, _):
        self.next_image()

