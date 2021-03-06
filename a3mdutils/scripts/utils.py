from a3mdutils.molecules import QmSetUp
from a3mdutils.qm import WaveFunction, WaveFunctionHDF5
from a3mdutils.parsers import forces
from a3mdutils.molecules import Mol2
from a3mdutils.utils import rename_atoms
from a3mdutils import BABEL2STANDARD
from a3mdutils.molecules import get_segment_charge
import numpy as np
import json
import h5py
import click
import time
import sys


def do_many(input_file, input_format, fun):
    with open(input_file) as f:
        if input_format == 'txt':
            inx = [i.strip() for i in f.readlines()]
        elif input_format == 'json':
            inx = json.load(f)
        else:
            print("unknown format. Use either json or txt")
            sys.exit()
    for line in inx:
        fun(line)


def __prepare_qm(name, charge, multiplicity, wfn, population, basis, method, nprocs, program, use_mol2_charge):
    start = time.time()
    mm = Mol2(name)
    adcs = []
    if population != "none":
        adcs.append('pop={:s}'.format(population))
    if wfn:
        adcs.append('output=wfn')
        if method == 'MP2':
            adcs.append('density=current')
    if not use_mol2_charge:
        mm.charges = (charge / mm.get_number_atoms()) * np.ones(mm.get_number_atoms(), dtype='float64')
    mm.multiplicity = multiplicity
    qmstp = QmSetUp(
        basis=basis, method=method, calculation_type='single', nprocs=nprocs,
        additional_commands=adcs, verbose=False
    )
    if program == 'g09':
        qmstp.write_g09(name.replace('.mol2', '.g09.input'), mm, wfn)
    elif program == 'orca':
        qmstp.write_orca(name.replace('.mol2', '.orca'), mm)
    else:
        print("unknown program. Use either ORCA or G09")
        sys.exit()

    print("TE : {:12.4f}".format(time.time() - start))



def __update_mol2(name, wfn, gaussian_log, output, charges):
    """
    updates a mol2 file using npa charges from gaussian output and coordinates of a wfn
    """
    from a3mdutils.qm import WaveFunction, GaussianLog
    from a3mdutils.molecules import UNITS_TABLE
    mm = Mol2(name)
    wfn_instance = WaveFunction.from_file(filename=wfn, program='g09', prefetch_dm=False)
    glog = GaussianLog(file=gaussian_log, method='', charges=charges, verbose=False)
    gdict = glog.read()
    mm.coordinates = wfn_instance.get_coordinates() * UNITS_TABLE['au']['angstrom']
    mm.charges = np.array(gdict['charges'], dtype='float64')
    mm.write(output)


def __update_many_mol2(inp, suffix, wfn_suffix, g09_suffix, input_type, charges):
    """
    updates many mol2 files at the same time
    """
    from a3mdutils.qm import WaveFunction, GaussianLog
    from a3mdutils.molecules import UNITS_TABLE
    if input_type == 'json':
        with open(inp) as f:
            input_contents = json.load(f)
    elif input_type == 'txt':
        with open(inp) as f:
            input_contents = [i.strip() for i in f.readlines()]
    else:
        print("unknown format. use either csv or json")
        sys.exit()

    for i, name in enumerate(input_contents):
        mm = Mol2(name + '.mol2')
        wfn = name + wfn_suffix
        gaussian_log = name + g09_suffix
        wfn_instance = WaveFunction.from_file(filename=wfn, program='g09', prefetch_dm=False)
        glog = GaussianLog(file=gaussian_log, method='', charges=charges, verbose=False)
        mm.coordinates = wfn_instance.get_coordinates() * UNITS_TABLE['au']['angstrom']
        try:
            gdict = glog.read()
            mm.charges = np.array(gdict['charges'], dtype='float64')
        except RuntimeError:
            print('error : cant read file {:s}'.format(name + g09_suffix))
            print('-- skipping charges for {:s}'.format(name + '.mol2'))
        mm.write(name + suffix)


def __convert_sample(name, random, input_type, output_type, npoints):
    """
    simple and fast conversion from csv to npy and viceversa
    """
    if input_type == "npy":
        sample = np.load(name)
    elif input_type == "csv":
        sample = np.loadtxt(name)
    else:
        print("unrecognized format {:s}".format(input_type))
        sys.exit()

    if random:
        n = sample.shape[0]
        perm = np.random.permutation(np.arange(n))
        sample = sample[perm]

    if npoints is not None:
        sample = sample[:npoints]

    if output_type == 'csv':
        np.savetxt(name.replace('.npy', '.csv'), sample)
    elif output_type == 'npy':
        np.save(name.replace('.csv', '.npy'), sample)


def __extract_forces(g09log, output, name=None, output_format='csv', symmetry=False):
    """
    extracts forces in Ha/Angstrom from g09 output
    """
    with open(g09log) as f:
        n, fx = forces([i.strip() for i in f.readlines()], symmetry=symmetry)
    if output_format == 'mol2':
        mm = Mol2(name)
        mm.coordinates = np.array(fx)
        mm.write(output)
    elif output_format == 'csv':
        np.savetxt(output, np.array(fx), delimiter=' ', fmt='%12.6e')
    else:
        for fx_ in fx:
            print("{:18.6e} {:18.6e} {:18.6e}".format(fx_[0], fx_[1], fx_[2]))


def __random_rotation(name, out, n=100):
    mm = Mol2(name)
    x = mm.get_coordinates()
    x = x - x.mean(axis=0)
    for i in range(n):
        u = (2.0 * np.random.rand(1)) - 1.0
        u = np.clip(u, -0.999999, 0.999999)
        theta = np.arccos(u)  # * np.sign(u)
        phi = np.random.rand(1) * 2.0 * np.pi

        rotx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)]
        ], dtype='float64')

        rotz = np.array([
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0]
        ], dtype='float64')

        y = np.dot(rotz, x.T)
        y = np.dot(rotx, y)
        mm.coordinates = y.T
        mm.write(out + '_%03d.mol2' % i)


def __rename_mol2_as(name, out, reference):
    mm = Mol2(name)
    mm = rename_atoms(mm, BABEL2STANDARD[reference])
    mm.write(out)


def __compile_wfn(name, out, save_dm, save_coeff, program, input_type):
    if input_type == 'wfn':
        wfn = WaveFunction.from_file(name, program=program)
    elif input_type == 'wfx':
        wfn = WaveFunction.from_wfx(name)
    else:
        raise IOError('use either wfn or wfx')
    suffix = '.{:s}'.format(input_type)
    f = WaveFunctionHDF5(out, mode='w-')
    f.add(name.replace(suffix, ''), wfn, save_dm=save_dm, save_coeff=save_coeff)
    f.close()


def __many_compile_wfn(name, out, save_dm, save_coeff, program, input_type):
    with open(name) as f:
        contents = [i.strip() for i in f.readlines()]

    g = WaveFunctionHDF5(out, mode='w-')

    if input_type == 'wfn':
        load_function = WaveFunction.from_file
    elif input_type == 'wfx':
        load_function = WaveFunction.from_wfx
    else:
        raise IOError('use either wfn or wfx')

    suffix = '.{:s}'.format(input_type)

    for wfn_name in contents:
        print(".. {:s}".format(wfn_name))
        wfn = load_function(wfn_name, program=program)
        g.add(wfn_name.replace(suffix, ''), wfn, save_dm=save_dm, save_coeff=save_coeff)
    f.close()


def __compile_mol2(name, out):
    mm = Mol2(file=name)
    atom_types = mm.atom_types
    segments = mm.segment_idx
    segment_ids = [i for i in mm.get_all_segs()]
    segment_ids = [i[:-1] for i in segment_ids]
    segcharges = get_segment_charge(segment_ids)
    ref_name = name.replace('.mol2', '')
    output_dict = dict()
    output_dict[ref_name] = dict()
    output_dict[ref_name]['atom_types'] = atom_types
    output_dict[ref_name]['segments'] = segments
    output_dict[ref_name]['segment_charges'] = segcharges
    with open(out, 'w') as f:
        json.dump(output_dict, f, indent=4)


def __many_compile_mol2(name, out):

    with open(name) as f:
        contents = [i.strip() for i in f.readlines()]

    output_dict = dict()
    for mol in contents:
        mm = Mol2(file=mol)
        atom_types = mm.atom_types
        segments = mm.segment_idx
        segment_ids = [i for i in mm.get_all_segs()]
        segment_ids = [i[:-1] for i in segment_ids]
        segcharges = get_segment_charge(segment_ids)
        ref_name = mol.replace('.mol2', '')
        output_dict[ref_name] = dict()
        output_dict[ref_name]['atom_types'] = atom_types
        output_dict[ref_name]['segments'] = segments
        output_dict[ref_name]['segment_charges'] = segcharges
        print('-- {:s} completed'.format(mol))
    with open(out, 'w') as f:
        json.dump(output_dict, f)


def __merge_h5(name, out):

    with open(name) as f:
        contents = [i.strip() for i in f.readlines()]

    f = h5py.File(out, 'w')

    for file in contents:
        print('processing {:s} file'.format(file))
        tmp = h5py.File(file, 'r')
        keys = tmp.keys()
        for k in keys:
            print('\t adding external link {:s}'.format(k))
            try:
                f[k] = h5py.ExternalLink(file, k)
            except RuntimeError:
                print('WARNING : link {:s} already exists. Skipping'.format(k))

    f.close()


@click.group()
def cli():
    """
    a3md utils
    
    a set of scripts to work with molecular representations and their electron densities

    """

    pass


@click.command()
@click.option('--charge', default=0, help='charge for qm simulation')
@click.option('--use_mol2_charge', default=False)
@click.option('--multiplicity', default=1, help='spin multiplicity for qm simulation')
@click.option('--wfn', default=True, help='return a wfn file', type=bool)
@click.option('--population', default='npa', help='population analysis')
@click.option('--basis', default='6-311++G(d,p)', help='basis set')
@click.option('--method', default='MP2', help='qm method')
@click.option('--nprocs', default=4)
@click.option('--program', default='g09', help='orca or g09')
@click.argument('name')
def prepare_qm(name, charge, multiplicity, wfn, population, basis, method, nprocs, program, use_mol2_charge):
    """

    Creates inputs for QM programs like Gaussian09 and ORCA from MOl2 files

    Example:

        a2mdutils.py prepare-qm --charge=0 --basis=6-31G --wfn=True --method=B3LYP --program=g09 benzene.mol2

    will generate a benzene.g09.input file that can be run in Gaussian09 to obtain the benzene electron density.

    """
    __prepare_qm(name, charge, multiplicity, wfn, population, basis, method, nprocs, program, use_mol2_charge)


@click.command()
@click.option('--charge', default=0, help='charge for qm simulation')
@click.option('--use_mol2_charge', default=False)
@click.option('--multiplicity', default=1, help='spin multiplicity for qm simulation')
@click.option('--wfn', default=True, help='return a wfn file', type=bool)
@click.option('--population', default='npa', help='population analysis')
@click.option('--basis', default='6-311++G(d,p)', help='basis set')
@click.option('--method', default='MP2', help='qm method')
@click.option('--nprocs', default=4)
@click.option('--program', default='g09', help='orca or g09')
@click.argument('name')
def many_prepare_qm(name, charge, multiplicity, wfn, population, basis, method, nprocs, program, use_mol2_charge):
    """

    Creates inputs for QM programs like Gaussian09 and ORCA from a list of MOL2 files

    Example:

        a2mdutils.py many-prepare-qm --charge=0 --basis=6-31G --wfn=True --method=B3LYP --program=g09 peptides.txt

    will generate a set of inputs for each file name in peptide

    """
    pqm = lambda x: __prepare_qm(x, charge, multiplicity, wfn, population, basis, method, nprocs, program,
                                 use_mol2_charge)
    do_many(name, 'txt', pqm)


@click.command()
@click.option('--charges', default='npa', help='either MK or NPA')
@click.argument('name')
@click.argument('wfn')
@click.argument('gaussian_log')
@click.argument('output')
def update_mol2(name, wfn, gaussian_log, output, charges):
    """

    Generates new mol2 files by including charge information from Gaussian09 outputs and from
    wavefunction file coordinates.

    Example:
        update_mol2 --charges=MK benzene.mol2 benzene.wfn benzene.g09.out benzene.n.mol2

    Will update the Mol2 file and will include the new charges calculated in G09.

    """
    __update_mol2(name, wfn, gaussian_log, output, charges)


@click.command()
@click.option('--charges', default='npa', help='either MK or NPA')
@click.option('--suffix', default='.n.mol2', help='updated files will have name + suffix')
@click.argument('name')
@click.argument('wfn_suffix')
@click.argument('g09_suffix')
def many_update_mol2(name, wfn_suffix, g09_suffix, charges, suffix):
    """

    Generates new mol2 files by including charge information from Gaussian09 outputs and from
    wavefunction file coordinates. It generates one molecule for each entry of a text file

    Example:
        update_mol2 --charges=MK mol_list .wfn .g09.out .n.mol2

    Will read mol_list, change the suffix, read the wfn and g09 output files, and generate new
    mol2 with the MK charges calculated in G09.

    """
    __update_many_mol2(
        name, suffix,
        wfn_suffix=wfn_suffix,
        g09_suffix=g09_suffix, input_type='txt',
        charges=charges
    )


@click.command()
@click.option('--input_type', default='csv', help='either csv or npy')
@click.option('--output_type', default='npy', help="either csv or npy")
@click.option('--random', default=False, help="permutates sample")
@click.option('--npoints', default=None, help="chooses the first N points")
@click.argument('name')
def convert_sample(name, random, input_type, output_type, npoints):
    """

    Converts a .csv file into an .npy file, and viceversa

    Example:
        convert-sample --input_type=csv --output_type=csv foo.csv

    Will just convert formats.
    IMPORTANTE NOTE: Remove previously any commas and headers using sed

    """
    __convert_sample(name, random, input_type, output_type, npoints)


@click.command()
@click.option('--input_type', default='csv', help='either csv or npy')
@click.option('--output_type', default='npy', help="either csv or npy")
@click.option('--random', default=False, help="permutates sample")
@click.option('--npoints', default=None, help="chooses the first N points")
@click.argument('name')
def many_convert_sample(name, random, input_type, output_type, npoints):
    """

    Converts many .csv files into an .npy file, and viceversa

    Example:
        convert-sample --input_type=csv --output_type=csv file_list

    Will just convert formats of all entries within file_list.
    IMPORTANTE NOTE: Remove previously any commas and headers using sed

    """
    cs = lambda x: __convert_sample(x, random, input_type, output_type, npoints)
    do_many(name, 'txt', cs)


@click.command()
@click.option('--outfmt', default=None, help='')
@click.argument('--name', default=None)
@click.argument('--symmetry', default=False)
@click.argument('g09log')
@click.argument('output')
def extract_forces(name, g09log, output, outfmt):
    """

    Extracts the values of forces from a G09 output (Force keyword)

    """
    __extract_forces(name, g09log, output, outfmt)


@click.command()
@click.option('--symmetry', default=False)
@click.argument('g09log')
def many_extract_forces(g09log, symmetry):
    """

    Extracts the values of forces from many G09 outputs (Force keyword)

    """
    ef = lambda x: __extract_forces(
        x, x.replace('.g09.output', '.ff.csv'), name=None, output_format='csv', symmetry=symmetry
    )
    do_many(g09log, 'txt', ef)


@click.command()
@click.argument('name')
@click.argument('out')
def random_rotation(name, out):
    """

    Rotates a mol2 file

    """
    __random_rotation(name, out)


@click.command()
@click.argument('name')
@click.argument('out')
@click.argument('reference')
def relabel_mol2(name, out, reference):
    """

    Applies a relabelling of the atoms of a Mol2. Use only for special cases.

    """
    __rename_mol2_as(name, out, reference)


@click.command()
@click.argument('name')
@click.argument('reference')
def many_relabel_mol2(name, reference):
    """

    Applies a relabelling of the atoms of many Mol2. Use only for special cases.

    """
    rm = lambda x: __rename_mol2_as(x, x.replace('.mol2', '.r.mol2'), reference)
    do_many(name, 'txt', rm)


@click.command()
@click.option('--save_dm', default=True, help='keep precalculated density matrix')
@click.option('--save_coeff', default=False, help='keep molecular orbital coefficients')
@click.option('--program', default='g09', help='either orca or g09')
@click.option('--input_type', default='wfn', type=click.Choice(['wfn', 'wfx']), help='wfn/wfx')
@click.argument('name')
@click.argument('out')
def compile_wfn(name, out, save_dm, save_coeff, program, input_type):
    """

    Generates an HDF5 file contaning all the information of a Wavefunction

    Example:
        compile-wfn --save_dm=True --save_coeff=False benzene.wfn benzene.wfn.hdf5

    Creates an HDF5 file containing the density matrix but not the molecular orbital coefficientes
    """
    __compile_wfn(name, out, save_dm, save_coeff, program, input_type)


@click.command()
@click.option('--save_dm', default=True, type=bool, help='keep precalculated density matrix')
@click.option('--save_coeff', default=False, type=bool, help='keep molecular orbital coefficients')
@click.option('--program', default='g09', help='either orca or g09')
@click.option('--input_type', default='wfn', type=click.Choice(['wfn', 'wfx']), help='wfn/wfx')
@click.argument('name')
@click.argument('out')
def many_compile_wfn(name, out, save_dm, save_coeff, program, input_type):
    """

    Generates an HDF5 file contaning the information of many Wavefunctions

    Example:
        many-compile-wfn --save_dm=True --save_coeff=False wfn_list.txt benzene.wfn.hdf5

    Creates an HDF5 file containing the density matrix of all the wfn contained in wfn_list
    but not the molecular orbital coefficientes
    """
    __many_compile_wfn(name, out, save_dm, save_coeff, program, input_type)


@click.command()
@click.argument('name')
@click.argument('out')
def merge_sources(name, out):
    """

    Generates and HDF5 containing the information of all the hdf5 files present in NAME through symbolic links.

    Example:
        many-compile-wfn --save_dm=True --save_coeff=False wfn_list1.txt set1.wfn.hdf5
        many-compile-wfn --save_dm=True --save_coeff=False wfn_list2.txt set2.wfn.hdf5
        ls set?.wfn.hdf5 > source
        merge-sources source final.wfn.hdf5


    """
    __merge_h5(name, out)


@click.command()
@click.argument('name')
@click.argument('out')
def compile_mol2(name, out):
    """

    Generates a json file containing molecular properties of a mol2 file

    """
    __compile_mol2(name, out)


@click.command()
@click.argument('name')
@click.argument('out')
def many_compile_mol2(name, out):
    """

    Generates a json file containing molecular properties of a list of mol2 files

    """
    __many_compile_mol2(name, out)


@click.command()
def say_hello():
    print("just wanted to say hello!")


cli.add_command(prepare_qm)
cli.add_command(update_mol2)
cli.add_command(convert_sample)
cli.add_command(extract_forces)
cli.add_command(random_rotation)
cli.add_command(relabel_mol2)
cli.add_command(compile_wfn)
cli.add_command(compile_mol2)
cli.add_command(many_prepare_qm)
cli.add_command(many_update_mol2)
cli.add_command(many_convert_sample)
cli.add_command(many_extract_forces)
cli.add_command(many_relabel_mol2)
cli.add_command(many_compile_wfn)
cli.add_command(many_compile_mol2)
cli.add_command(merge_sources)

if __name__ == "__main__":
    cli()
