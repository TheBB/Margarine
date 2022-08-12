from pathlib import Path
import peewee as pw

from xdg import xdg_config_home, xdg_data_home

import yaml


class Config:

    def __init__(self, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        self.block_levels = data.get('levels', ['A', 'B', 'C'])
        self.advance_prob = data.get('advance-prob', 0.0)
        self.message_prob = data.get('message-prob', 0.0)
        self.messages = data.get('messages', [])
        self.step = data.get('step', 0.1)

        blurs = data.get('blurs', [40])
        ampl = data.get('blur-amplify', 1.0)
        self.blurs = [b * ampl for b in blurs]


(xdg_data_home() / 'margarine').mkdir(parents=True, exist_ok=True)

config = Config(xdg_config_home() / 'margarine' / 'config.yaml')
db = pw.SqliteDatabase(xdg_data_home() / 'margarine' / 'db.sqlite3')
db.connect()


# cur = db.cursor()
# cur.execute("""
#     CREATE TABLE IF NOT EXISTS pictures (
#         id integer primary key,
#         image blob,
#         mask text,
#         fingerprint integer,
#         seen integer default 0
#     )
# """)
# db.commit()


# def insert_image(image, mask, fingerprint):
#     cur.execute(
#         'INSERT INTO pictures (image, mask, fingerprint) VALUES (?, ?, ?)',
#         (image, mask, fingerprint)
#     )
#     db.commit()


# def get_image(ident=None):
#     if ident is None:
#         cur.execute("""
#             SELECT id, image, mask, fingerprint
#             FROM pictures
#             WHERE seen = 0
#             ORDER BY RANDOM() LIMIT 1;
#         """)
#         retval = cur.fetchone()
#         if retval is None:
#             cur.execute('UPDATE pictures SET seen=0')
#             db.commit()
#             return get_image()
#         return retval
#     else:
#         cur.execute("""
#             SELECT id, image, mask, fingerprint
#             FROM pictures WHERE id = ?;
#         """, (ident,))
#     return cur.fetchone()


# def delete_image(ident):
#     cur.execute('DELETE FROM pictures WHERE id=?', (ident,))
#     db.commit()


# def mark_seen(ident):
#     cur.execute('UPDATE pictures SET seen=1 WHERE id=?', (ident,))
#     db.commit()


# def get_fingerprints():
#     cur.execute('SELECT id, fingerprint FROM pictures;')
#     while True:
#         data = cur.fetchone()
#         if data is not None:
#             yield data
#         else:
#             break
