import click
from a3mdutils.qm import WaveFunction
from a3mdutils


@click.command()
@click.argument('NAME')
def write_volume(name):

    u = WaveFunction.from_wfx(filename=name)


    return u


if __name__ == "__main__":

    read()
