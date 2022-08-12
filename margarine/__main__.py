import click
from functools import partial
from random import shuffle

from .program import (
    AllImagesProgram, ClassifyProgram, DatabaseProgram, RevisitProgram,
    DisplayHeatmapProgram, DisplayImageProgram
)
from .gui import run_gui


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(slideshow)


@main.command()
@click.option('--random', is_flag=True)
@click.argument('filenames', nargs=-1, type=click.Path(exists=True))
def classify(random, filenames):
    filenames = list(filenames)
    if random:
        shuffle(filenames)
    run_gui(ClassifyProgram(filenames))


@main.command()
def revisit():
    run_gui(RevisitProgram())


@main.command()
def all_images():
    run_gui(AllImagesProgram())


@main.command()
@click.option('--level-min', 'lowest_level', type=float, default=0)
@click.option('--level-max', 'highest_level', type=float, default=3)
def slideshow(lowest_level, highest_level):
    displayer = partial(
        DisplayImageProgram,
        lo=min(lowest_level, highest_level),
        hi=highest_level,
    )
    run_gui(DatabaseProgram(displayer))


@main.command()
def heatmap():
    run_gui(DatabaseProgram(DisplayHeatmapProgram))
