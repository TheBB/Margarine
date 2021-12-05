#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='Margarine',
    version='0.1',
    description='Image database',
    author='Eivind Fonn',
    author_email='evfonn@gmail.com',
    license='GPL3',
    url='https://github.com/TheBB/Margarine',
    py_modules=['margarine'],
    entry_points={
        'console_scripts': ['margarine=margarine.__main__:main'],
    },
    install_requires=[
        # 'click',
        # 'imagehash',
        # 'inflect',
        'numpy',
        'pillow',
        'pyqt5',
        # 'requests',
        # 'sqlalchemy',
        # 'selenium',
        # 'tqdm',
        # 'xdg',
        # 'yapsy',
    ],
)
