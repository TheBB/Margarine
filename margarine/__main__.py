import click
from random import shuffle

from .program import ClassifyProgram, DatabaseProgram
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
@click.option('--level-min', 'lowest_level', type=int, default=0)
@click.option('--level-max', 'highest_level', type=int, default=2)
def slideshow(**kwargs):
    run_gui(DatabaseProgram(**kwargs))
