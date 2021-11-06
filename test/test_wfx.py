import click
from a3mdutils.qm import WaveFunction


@click.command()
@click.argument('NAME')
def read(name):

    u = WaveFunction.from_wfx(filename=name)
    return u


if __name__ == "__main__":

    read()
