from bisect import bisect
from enum import IntEnum
from io import BytesIO
from pathlib import Path
import shutil
import imagehash
import uuid

from cached_property import cached_property
import numpy as np
import peewee as pw

from PyQt5.QtGui import QPixmap

from PIL import Image, ImageQt, ImageFilter, ImageDraw

from typing import Optional, Tuple

from . import config


def tonk(s):
    return int(sum(j<<i for i,j in enumerate(s.hash.flatten().astype('i8')[::-1])))


def untonk(s):
    return np.unpackbits(np.array([s], dtype='>i8').view(np.uint8))


class Picture(pw.Model):

    class Meta:
        database = config.db
        db_table = 'pictures'

    filename: Optional[str] = None

    ident = pw.PrimaryKeyField(column_name='id')
    _image = pw.BlobField(column_name='image')
    _mask = pw.TextField(column_name='mask')
    mask_nx = pw.IntegerField()
    mask_ny = pw.IntegerField()
    mask_rfac = pw.DoubleField()
    _fingerprint = pw.IntegerField(column_name='fingerprint')
    seen = pw.BooleanField(default=False)
    revisit = pw.BooleanField(default=False)

    @cached_property
    def raw_image(self):
        with BytesIO(self._image) as buf:
            img = Image.open(buf)
            img.load()
        return img

    @cached_property
    def image(self):
        img = self.raw_image
        img = img.convert('RGBA')
        w, h = img.size
        fac = w / 1920 if w / h > 1920 / 1080 else h / 1080
        nw, nh = int(w / fac), int(h / fac)
        return img.resize((nw, nh))

    @cached_property
    def mask(self):
        return np.array(list(map(int, self._mask.split(','))))

    @cached_property
    def fingerprint(self):
        if self._fingerprint is None:
            self._fingerprint = tonk(imagehash.phash(self.image))
        return untonk(self._fingerprint)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as buf:
            image = buf.read()
        obj = cls(_image=image)
        obj.filename = filename
        return obj

    @classmethod
    def from_db(cls, ident=None):
        if ident is not None:
            return cls.get(cls.ident == ident)
        query = cls.select().where(cls.seen == False, cls.revisit == False)
        if not query.exists():
            cls.update(seen=False).execute()
        return query.order_by(pw.fn.RANDOM()).get()

    @classmethod
    def get_revisit(cls):
        query = cls.select().where(cls.revisit == True)
        if not query.exists():
            return None
        return query.get()

    def mark_seen(self):
        self.seen = True
        self.save()

    def compute_fingerprint(self):
        return self.fingerprint

    def save_to_db(self):
        assert self.ident is None
        fingerprint = tonk(self.compute_fingerprint())
        mask = ','.join(map(str, self.mask))
        config.insert_image(self.image, mask, fingerprint)

    def mask_manager(self):
        return MaskManager(self.image, self.mask_nx, self.mask_ny, self.mask_rfac)

    def find_colliding_pictures(self, threshold=10):
        fingerprint = self.compute_fingerprint()
        for other in Picture.select():
            diff = np.count_nonzero(fingerprint != other.fingerprint)
            if diff < threshold:
                yield other

    def dispatch(self, path: Path, relative=True, as_dir=True, delete=False):
        if relative:
            path = Path(self.filename).parent / path
        if path.is_dir() or as_dir:
            filename = self.filename
            if filename is None:
                filename = f'{self.ident:08}.png'
            path = path / filename
        elif not path.suffix:
            path = path.with_suffix('.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        self.raw_image.save(str(path), 'PNG')

        if delete:
            if self.filename:
                Path(self.filename).unlink()
            self.delete_instance()

    def eject(self):
        name = f'ejected-{uuid.uuid1()}.png'
        self.image.save(name, 'PNG')
        self.delete_instance()


class MaskManager:

    image: Image.Image

    radius: float
    delta: float

    centers: np.ndarray
    neighbors: np.ndarray

    @staticmethod
    def propose_size(width: int, height: int) -> Tuple[int, int, float]:
        radius = max(width / 1920, height / 1080) * 1920 / 50
        rfac = radius / width
        delta = delta = radius / np.sqrt(3)
        nx = int(np.ceil(width / radius))
        ny = int(np.ceil(height / 6 / delta))
        return nx, ny, rfac

    def __init__(self, image: Image.Image, nx: Optional[int] = None, ny: Optional[int] = None, rfac: Optional[float] = None):
        self.image = image
        width, height = image.size

        if nx is None or ny is None or rfac is None:
            nx, ny, rfac = MaskManager.propose_size(width, height)
        self.nx = nx
        self.ny = ny
        self.rfac = rfac

        self.radius = width * rfac
        self.delta = self.radius / np.sqrt(3)

        ex_xpts = np.arange(-1, nx+2) * self.radius * 2
        in_xpts = (ex_xpts[:-1] + ex_xpts[1:]) / 2
        ex_xpts = ex_xpts[:-1]
        ypts = np.arange(-ny, ny+1) * 3 * self.delta + height / 2

        centers = np.empty((len(ex_xpts), len(ypts), 2), dtype=float)
        centers[..., 1] = ypts.reshape(1, -1)
        centers[:, ::2, 0] = ex_xpts.reshape(-1, 1)
        centers[:, 1::2, 0] = in_xpts.reshape(-1, 1)
        self.centers = centers.reshape(-1, 2)

        shp = (len(ex_xpts), len(ypts))
        neighbors = -np.ones((len(self.centers), 6), dtype=int)
        i = np.arange(len(self.centers))
        ix, iy = np.unravel_index(i, shp)

        neighbors[:,0] = np.ravel_multi_index([ix+1, iy], shp, mode='clip')
        neighbors[:,1] = np.ravel_multi_index([ix-1, iy], shp, mode='clip')
        neighbors[:,2] = np.ravel_multi_index([ix - (iy % 2 == 0), iy-1], shp, mode='clip')
        neighbors[:,3] = np.ravel_multi_index([ix + 1 - (iy % 2 == 0), iy-1], shp, mode='clip')
        neighbors[:,4] = np.ravel_multi_index([ix + 1 - (iy % 2 == 0), iy+1], shp, mode='clip')
        neighbors[:,5] = np.ravel_multi_index([ix - (iy % 2 == 0), iy+1], shp, mode='clip')
        neighbors[neighbors == i.reshape(-1,1)] = -1
        self.neighbors = neighbors

    def zeros(self, dtype):
        return np.zeros((len(self.centers),), dtype=dtype)

    def ones(self, dtype):
        return np.ones((len(self.centers),), dtype=dtype)

    def draw_hex(self, drawer: ImageDraw.ImageDraw, index: int, color: Tuple[int, int, int, int], width: int = 3):
        x, y = self.centers[index]
        r = self.radius
        d = self.delta

        points = [
            (x + r, y - d),
            (x, y - 2*d),
            (x - r, y - d),
            (x - r, y - d),
            (x - r, y + d),
            (x, y + 2*d),
            (x + r, y + d),
            (x + r, y - d),
            (x, y - 2*d),
        ]
        drawer.polygon(points[:-2], fill=color)
        drawer.line(points, 'white', width=width, joint='curve')

    def neighbors_of(self, indices):
        return set(self.neighbors[list(indices)].flatten()) - indices - {-1}

    def boundary_of(self, indices):
        retval = set()
        for k in indices:
            neighbors = self.neighbors[k]
            if set(neighbors) - indices:
                retval.add(k)
        return retval


class BlurMask:

    def __init__(self, manager: MaskManager, blur=0, require_privilege=False):
        self.manager = manager

        blur_size = blur * max(self.manager.image.size) / 700
        self.blurred_image = self.manager.image.filter(ImageFilter.GaussianBlur(blur_size))
        self.vis = self.manager.zeros(dtype=bool)

        self.require_privilege = require_privilege

    def make_image(self):
        mask = Image.new('RGBA', self.manager.image.size)
        mask.paste(self.manager.image, (0, 0))
        # mask = self.manager.image.convert('RGBA')
        drawer = ImageDraw.Draw(mask)

        width, height = self.manager.image.size
        fac = max(width / 1920, height / 1080)
        w = int(4 * fac)

        for ix in np.where(~self.vis)[0]:
            self.manager.draw_hex(drawer, ix, (0, 0, 0, 0), width=w)

        image = self.blurred_image.copy()
        image.paste(mask, (0, 0), mask)
        image = image.convert('RGBA')
        self.qimage = ImageQt.ImageQt(image)
        return QPixmap.fromImage(self.qimage)

    def uncover(self, indexes, privilege=False):
        if self.require_privilege and not privilege:
            return False

        if np.isscalar(indexes):
            indexes = np.array([indexes])

        self.vis.flat[indexes] = True

    def modify(self, relx, rely):
        pass


class MaskPicker:

    def __init__(self, manager: MaskManager, allowed_indices):
        self.manager = manager
        self.vis = self.manager.zeros(dtype=int)
        self.vis[list(allowed_indices)] = 1
        self.translucency = 180

    @property
    def marked(self):
        return np.where(self.vis == 2)[0]

    def make_image(self):
        mask = Image.new('RGBA', self.manager.image.size)
        mask.paste(self.manager.image, (0, 0))
        drawer = ImageDraw.Draw(mask)

        width, height = self.manager.image.size
        fac = max(width / 1920, height / 1080)
        w = int(4 * fac)

        for ix, vis in enumerate(self.vis):
            if vis == 0:
                self.manager.draw_hex(drawer, ix, (0, 0, 0, self.translucency), width=w)
            elif vis == 1:
                self.manager.draw_hex(drawer, ix, (0, 0, 0, 0), width=w)
            elif vis == 2:
                self.manager.draw_hex(drawer, ix, (255, 0, 0, self.translucency * 128 // 180), width=w)

        m = self.manager.image.copy()
        m.paste(mask, (0, 0), mask)
        m = m.convert('RGBA')
        self.qimage = ImageQt.ImageQt(m)
        return QPixmap.fromImage(self.qimage)

    def invert(self):
        ones, = np.where(self.vis == 1)
        twos, = np.where(self.vis == 2)
        self.vis[ones] = 2
        self.vis[twos] = 1

    def modify(self, relx, rely, value=2):
        if not 0 <= relx <= 1 or not 0 <= rely <= 1:
            return False

        width, height = self.manager.image.size
        xc = relx * width
        yc = rely * height

        loc = np.argmin(np.linalg.norm(self.manager.centers - (xc, yc), axis=1))
        if self.vis[loc] != 0 and self.vis[loc] != value:
            self.vis[loc] = value
            return True

        return False

    def undo(self, relx, rely):
        return self.modify(relx, rely, value=1)


class PureMask:

    def __init__(self, picture: Picture):
        self.picture = picture

    def make_image(self):
        self.image = self.picture.image
        self.qimage = ImageQt.ImageQt(self.image)
        return QPixmap.fromImage(self.qimage)


# class ImageMask:

#     def __init__(self, manager: MaskManager, blur=0, default=None, target=None, require_privilege=False):
#         blur_size = blur * max(manager.image.size) / 700
#         self.blurred_image = manager.image.filter(ImageFilter.GaussianBlur(blur_size))
#         self.manager = manager
#         self.vis = manager.ones(dtype=int) * int(0)
#         self.default = default
#         self.target = target
#         self.require_privilege = require_privilege

#     def image_data(self):
#         with BytesIO() as buf:
#             self.manager.image.save(buf, 'JPEG', quality=90)
#             return buf.getvalue()

    # def make_image(self):
    #     mask = Image.new('RGBA', self.manager.image.size)
    #     mask.paste(self.manager.image, (0, 0))
    #     drawer = ImageDraw.Draw(mask)

    #     width, height = self.manager.image.size
    #     fac = max(width / 1920, height / 1080)
    #     w = int(4 * fac)
    #     for flat, vis in enumerate(self.vis):
    #         if vis != Overlay.Nothing:
    #             self.manager.draw_hex(drawer, flat, Overlay.color(vis), width=w)

    #     image = self.blurred_image.copy()
    #     image.paste(mask, (0, 0), mask)
    #     image = image.convert('RGBA')
    #     self.qimage = ImageQt.ImageQt(image)
    #     return QPixmap.fromImage(self.qimage)

    # def closest_in_row(self, row, x, y):
    #     xs = self.ex_xpts if row % 2 == 0 else self.in_xpts
    #     i = np.argmin(np.abs(xs - x))
    #     dist = np.linalg.norm([xs[i] - x, self.manager.ypts[row] - y])
    #     return (i, row), dist

    # def undo(self, relx, rely):
    #     return self.modify(relx, rely, self.default)

    # def modify(self, relx, rely, value=None, privilege=False):
    #     if not 0 <= relx <= 1 or not 0 <= rely <= 1:
    #         return False

    #     width, height = self.manager.image.size
    #     xc = relx * width
    #     yc = rely * height

    #     iy = bisect(self.manager.ypts, yc)

    #     blk, dist = self.closest_in_row(iy, xc, yc)
    #     if iy > 0:
    #         alt_blk, alt_dist = self.closest_in_row(iy - 1, xc, yc)
    #         if alt_dist < dist:
    #             blk = alt_blk

    #     k = np.ravel_multi_index(blk, self.vis.shape)
    #     return self.assign(k, value=value, privilege=privilege)

    # def assign(self, indexes, value=None, privilege=False):
    #     if self.require_privilege and not privilege:
    #         return False
    #     if value is None:
    #         value = self.target

    #     if np.isscalar(indexes):
    #         indexes = np.array([indexes])

    #     to_change = np.logical_and(self.vis.flat[indexes] != value, self.vis.flat[indexes] != Overlay.Fixed)
    #     self.vis.flat[indexes[to_change]] = value
    #     return to_change.any()

    # def invert(self, a, b):
    #     ai = np.where(self.vis == a)
    #     bi = np.where(self.vis == b)
    #     self.vis[ai] = b
    #     self.vis[bi] = a

    # def convert(self, to=Overlay.Fixed, frm=None, invert=False):
    #     if frm is None:
    #         frm = self.target
    #     if invert:
    #         self.vis[self.vis != frm] = to
    #     else:
    #         self.vis[self.vis == frm] = to

    # def marked(self, value=None):
    #     if value is None:
    #         value = self.target
    #     return self.vis.flatten() == value
