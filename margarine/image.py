from bisect import bisect
from enum import IntEnum
from io import BytesIO
import imagehash
import uuid

import numpy as np

from PyQt5.QtGui import QPixmap

from PIL import Image, ImageQt, ImageFilter, ImageDraw
from numpy.lib.arraysetops import isin
from numpy.testing._private.utils import requires_memory

from . import config


def tonk(s):
    return int(sum(j<<i for i,j in enumerate(s.hash.flatten().astype('i8')[::-1])))


def untonk(s):
    return np.unpackbits(np.array([s], dtype='>i8').view(np.uint8))


class Picture:

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as buf:
            image = buf.read()
        obj = cls(image)
        obj.filename = str(filename)
        return obj

    @classmethod
    def from_db(cls, ident=None):
        ident, image, mask, fingerprint = config.get_image(ident=ident)
        mask = np.array(list(map(int, mask.split(','))))
        return cls(image, ident=ident, mask=mask, fingerprint=fingerprint)

    def __init__(self, image, ident=None, mask=None, fingerprint=None):
        self.image = image
        self.ident = ident
        self.fingerprint = fingerprint
        self.mask = mask

    def mark_seen(self):
        config.mark_seen(self.ident)

    def compute_fingerprint(self):
        if self.fingerprint is None:
            with BytesIO(self.image) as buf:
                self.fingerprint = imagehash.phash(Image.open(buf))
        return self.fingerprint

    def make_masker(self, *args, init_mask=False, **kwargs):
        with BytesIO(self.image) as buf:
            masker = ImageMask(buf, *args, **kwargs)
        if init_mask:
            self.mask = masker.vis.copy().flatten()
            self.mask[:] = 0
        return masker

    def save_to_db(self):
        assert self.ident is None
        fingerprint = tonk(self.compute_fingerprint())
        mask = ','.join(map(str, self.mask))
        config.insert_image(self.image, mask, fingerprint)

    def find_colliding_idents(self, threshold=10):
        fingerprint = untonk(tonk(self.compute_fingerprint()))
        for ident, other in config.get_fingerprints():
            diff = np.count_nonzero(fingerprint != untonk(other))
            if diff < threshold:
                yield ident

    def eject(self):
        name = f'ejected-{uuid.uuid1()}.png'
        self.image.save(name, 'PNG')
        config.delete_image(self.ident)


class Overlay(IntEnum):
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


def draw_hex(drawer, x, y, r, dy, color, w=3):
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

    def __init__(self, path_or_fp, blur=0, default=Overlay.Transparent, target=Overlay.Nothing, require_privilege=False):
        self.image = image = Image.open(path_or_fp)

        blur_size = blur * max(image.size) / 1000
        self.blurred_image = image.filter(ImageFilter.GaussianBlur(blur_size))

        width, height = image.size
        self.radius = radius = max(width / 1920, height / 1080) * 1920 / 50
        self.delta = delta = radius / np.sqrt(3)

        nx = int(np.ceil(width / radius))
        self.ex_xpts = ex_xpts = np.arange(-1, nx+1) * radius * 2
        self.in_xpts = (ex_xpts[:-1] + ex_xpts[1:]) / 2
        ny = int(np.ceil(height / 6 / delta))
        self.ypts = ypts = np.arange(-ny, ny+1) * 3 * delta + height / 2

        self.vis = np.ones((len(ex_xpts), len(ypts)), dtype=int) * int(default)
        self.default = default
        self.target = target
        self.require_privilege = require_privilege

    def image_data(self):
        with BytesIO() as buf:
            self.image.save(buf, 'JPEG', quality=90)
            return buf.getvalue()

    def make_image(self):
        mask = Image.new('RGBA', self.image.size)
        mask.paste(self.image, (0, 0))
        drawer = ImageDraw.Draw(mask)

        width, height = self.image.size
        fac = max(width / 1920, height / 1080)
        w = int(4 * fac)
        for iy, y in enumerate(self.ypts):
            xs = self.ex_xpts if iy % 2 == 0 else self.in_xpts
            for ix, x in enumerate(xs):
                if self.vis[ix, iy] != Overlay.Nothing:
                    draw_hex(drawer, x, y, self.radius, self.delta, Overlay.color(self.vis[ix, iy]), w=w)

        image = self.blurred_image.copy()
        image.paste(mask, (0, 0), mask)
        image = image.convert('RGBA')
        self.qimage = ImageQt.ImageQt(image)
        return QPixmap.fromImage(self.qimage)

    def closest_in_row(self, row, x, y):
        xs = self.ex_xpts if row % 2 == 0 else self.in_xpts
        i = np.argmin(np.abs(xs - x))
        dist = np.linalg.norm([xs[i] - x, self.ypts[row] - y])
        return (i, row), dist

    def undo(self, relx, rely):
        return self.modify(relx, rely, self.default)

    def modify(self, relx, rely, value=None, privilege=False):
        if not 0 <= relx <= 1 or not 0 <= rely <= 1:
            return False

        width, height = self.image.size
        xc = relx * width
        yc = rely * height

        iy = bisect(self.ypts, yc)

        blk, dist = self.closest_in_row(iy, xc, yc)
        if iy > 0:
            alt_blk, alt_dist = self.closest_in_row(iy - 1, xc, yc)
            if alt_dist < dist:
                blk = alt_blk

        k = np.ravel_multi_index(blk, self.vis.shape)
        return self.assign(k, value=value, privilege=privilege)

    def assign(self, indexes, value=None, privilege=False):
        if self.require_privilege and not privilege:
            return False
        if value is None:
            value = self.target

        if np.isscalar(indexes):
            indexes = np.array([indexes])

        to_change = np.logical_and(self.vis.flat[indexes] != value, self.vis.flat[indexes] != Overlay.Fixed)
        self.vis.flat[indexes[to_change]] = value
        return to_change.any()

    def invert(self, a, b):
        ai = np.where(self.vis == a)
        bi = np.where(self.vis == b)
        self.vis[ai] = b
        self.vis[bi] = a

    def convert(self, to=Overlay.Fixed, frm=None, invert=False):
        if frm is None:
            frm = self.target
        if invert:
            self.vis[self.vis != frm] = to
        else:
            self.vis[self.vis == frm] = to

    def marked(self, value=None):
        if value is None:
            value = self.target
        return self.vis.flatten() == value
