import a3mdnet
import torch
from a3mdnet.models.ddnn import A3MDnet
from a3mdnet.functions import alt_distance_vectors
from a3mdnet import MODELS
from a3mdutils.qm import WaveFunction
from a3mdnet.sampling import IntegrationGrid
from a3mdnet.input import mol2_to_tensor, mol_to_tensor
import numpy as np
import click
import time


@click.group()
def cli():
    pass


@click.command()
@click.option('--model', default='tdnn3', type=click.Choice(['tdnn3']))
@click.option('--units', default='bohr', type=click.Choice(['bohr', 'angstrom']))
@click.option('--device', default='cuda:0', type=click.Choice(['cuda:0', 'cpu']))
@click.argument('name')
@click.argument('coords')
@click.argument('output')
def upon_coordinates(name, coords, output, model, units, device):
    start = time.time()
    device = torch.device(device)
    u = mol2_to_tensor(name, device=device)
    coords = np.loadtxt(coords)
    xyz = torch.from_numpy(coords).unsqueeze(0).to(device)
    dv = alt_distance_vectors(xyz, u.coordinates, torch.float, device)
    model: A3MDnet = a3mdnet.MODELS[model]
    p, _ = model.forward(dv, u.atomic_numbers, u.coordinates, u.charge)
    p = p.data.squeeze(0).cpu().numpy()
    np.savetxt(np.stack([coords, p], axis=0), output, fmt='%12.8e')
    end = time.time()
    print("TE: {:12.4f}".format(end - start))

@click.command()
@click.option('--output_type', type=click.Choice(['csv', 'npy', 'xyz']), default='xyz')
@click.option('--radial_resolution', type=int, default=20)
@click.option('--grid', type=click.Choice(['minimal', 'xtcoarse', 'coarse']), default='minimal')
@click.option('--device', type=str, default='cpu')
@click.argument('name')
@click.argument('output')
def generate_grid(name, output,  device , output_type, radial_resolution, grid):
    device = torch.device(device)
    mm = mol2_to_tensor(name, device=device)
    u = mol_to_tensor(mm, device=device, dtype=torch.float)
    ig = IntegrationGrid(
        grid=grid, radial_resolution=radial_resolution, softening=3, rm=5.0
    ).to(device)
    z, _, w = ig.sample(u.atomic_numbers, u.coordinates)
    z = z.squeeze(0)
    w = w.squeeze(0)
    z = z.cpu().numpy()
    w = w.cpu().numpy()

    if output_type == 'csv':

        np.savetxt(output + '.grid.csv', z, fmt='%16.8e', delimiter=' ')
        np.savetxt(output + '.weights.csv', w, fmt='%16.8e', delimiter=' ')

    elif output_type == 'npy':

        np.save(output + '.grid.npy', z)
        np.save(output + '.weights.npy', w)

    if output_type == 'xyz':

        with open(output + '.grid.xyz', 'w') as f:
            f.write('{:d}\n'.format(z.shape[0]))
            for i in range(z.shape[0]):
                f.write('{:18.6e} {:18.6e} {:18.6e}\n'.format(*z[i, :]))

        np.savetxt(output + '.weights.csv', w, fmt='%16.8e', delimiter=' ')


@click.command()
@click.option('--program', type=click.Choice(['g09', 'orca']), default='orca')
@click.argument('name')
@click.argument('potential')
@click.argument('weights')
def wfn_dft_energy(name, potential, weights, program):
    tf_factor = 3/10 * np.power(3 * (np.pi ** 2), 2/3)
    dx_factor = (3/4) * np.power(3 / np.pi, 1/3)
    wfn = WaveFunction.from_file(name, prefetch_dm=True, program=program)
    potential_values = []
    grid = []
    with open(potential) as f:
        for line in f.readlines()[1:]:
            x, y, z, w = [float(i) for i in line.split()]
            potential_values.append(w)
            grid.append([x, y, z])
    potential = np.array(w)
    grid = np.array(grid)
    weights = np.loadtxt(weights)
    z = wfn.get_atomic_numbers()
    r = wfn.get_coordinates()
    # calculating density
    p = wfn.eval(grid)
    # eval nuclear potential on grid
    grid = grid.reshape(grid.shape[0], 1, 3)
    r = r.reshape(1, r.shape[0], 3)
    potential_nuclear = (np.power(np.linalg.norm(grid - r, axis=2), -1) * z.reshape(1, -1)).sum(1)
    potential_electron = potential - potential_nuclear
    # eval nuclear nuclear potential
    vee = (potential_electron * p * weights).sum()
    dx = - (weights * np.power(p, 4/3)).sum() * dx_factor
    tf = (weights * np.power(p, 5/3)).sum() * tf_factor
    print('{:s} {:16.8e} {:16.8e} {:16.8e}'.format(name, vee, dx, tf))


@click.command()
@click.option('--output_type', type=click.Choice(['csv', 'npy']), default='csv')
@click.option('--radial_resolution', type=int, default=20)
@click.option('--grid', type=click.Choice(['minimal', 'xtcoarse', 'coarse']), default='minimal')
@click.argument('name')
@click.argument('device')
def many_generate_grid(name, device, output_type, radial_resolution, grid):
    device = torch.device(device)
    with open(name) as f:
        contents = [i.strip() for i in f.readlines()]

    for mol_name in contents:
        print('-- generating grid : {:s} / {:s} / {:d}'.format(mol_name, grid, radial_resolution))
        u = mol2_to_tensor(mol_name, device=device)
        output = mol_name.replace('.mol2', '')
        ig = IntegrationGrid(
            device=device, dtype=torch.float, grid=grid, radial_resolution=radial_resolution, softening=3, rm=5.0
        )
        z, _, w = ig.integration_grid(u.coordinates, u.labels)
        z = z.get_cartessian().squeeze(0)
        w = w.squeeze(0)
        z = z.cpu().numpy()
        w = w.cpu().numpy()

        if output_type == 'csv':

            np.savetxt(output + '.grid.csv', z, fmt='%16.8e', delimiter=' ')
            np.savetxt(output + '.weights.csv', w, fmt='%16.8e', delimiter=' ')

        elif output_type == 'npy':

            np.save(output + '.grid.npy', z)
            np.save(output + '.weights.npy', w)


cli.add_command(upon_coordinates)
cli.add_command(generate_grid)
cli.add_command(wfn_dft_energy)
cli.add_command(many_generate_grid)

if __name__ == '__main__':

    cli()
    print("finished!")