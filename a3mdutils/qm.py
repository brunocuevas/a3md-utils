import numpy as np
import re
from a3mdutils.volumes import Volume
import h5py
from a3mdutils import WFN_SYMMETRY_INDEX
from a3mdutils.wfx import from_wfx


class A2MDlibQM:
    def __init__(self, name, verbose):
        self.__name = name
        if verbose in [True, False]:
            self.__verbose = verbose
        else:
            self.__verbose = True

    def log(self, mssg):
        if self.__verbose:
            print("[{:s}] {:s}".format(self.__name, mssg))


symmetry_index = np.array(WFN_SYMMETRY_INDEX)

atom_names = list(
    ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
)


def parse_fortran_scientific(fortran_number):
    """
    This functions turns a fortran scienfitic notation into a C scientific notation.
    For instance:
    8.45612D-05 -> 8.45612E-05
    so it can be converted to python float

    :param fortran_number: string containing a number in fortran scientific notations
    :return: floating number
    """
    c_number = fortran_number.replace('D', 'E')
    return float(c_number)


class WaveFunction(A2MDlibQM):

    def __init__(
        self, coeff, dm, occ, exponents, syms, centers, coords, types, charges, primitives,
        molecular_orbitals, nuclei, verbose=False, prefetch_dm=False
    ):
        A2MDlibQM.__init__(self, verbose=verbose, name='wavefunction handler')
        self.coefficients = coeff
        self.occupancies = occ
        self.exponents = exponents
        self.symmetry_indices = syms
        self.primitive_centers = centers
        self.coordinates = coords
        self.atomic_symbols = types
        self.charges = charges
        self.number_primitives = primitives
        self.number_orbitals = molecular_orbitals
        self.number_centers = nuclei
        self.density_matrix = dm
        self.total_charge = self.get_charge()
        if prefetch_dm:
            self.density_matrix = self.calculate_density_matrix()

    def __call__(self, x):
        return self.eval(x)

    @staticmethod
    def __gaussian(r, s, exp):
        """

        Calculates a gaussian function (e^(-(x^2))) with a given
        harmonic, expressed in the function s. For instance, if
        s = [0,0,2], then the function calculates:

        f(x) = e^(-x^"2) * (z^2)

        :param r: radial distance to center
        :type r: np.ndarray
        :param s: symmetry type
        :type s: np.ndarray
        :param exp: exponent of the gassian
        :type exp: np.ndarray
        :return: gaussian function value
        :rtype: np.ndarray
        """
        r2 = np.sum(r * r, axis=1)
        smx = r[:, 0] ** s[0]
        smx *= r[:, 1] ** s[1]
        smx *= r[:, 2] ** s[2]
        return smx * np.exp(-r2 * exp)

    @staticmethod
    def __parse_centers(wfn, primitives):
        """

        In charge of parsing the distribution of centers for the basis set

        :param wfn:
        :param primitives:
        :return:
        """
        centers = np.zeros(primitives, dtype='int64')
        line = wfn[0]
        i = 0
        j = 0
        while line[:6] == "CENTRE":
            sp = line.split()
            sp = sp[2:]
            for centre in sp:
                centers[i] = int(centre) - 1
                i += 1
            j += 1
            line = wfn[j]
        return centers, wfn[j:]

    @staticmethod
    def __parse_coordinates(wfn_instance, nuclei_number):
        """

        In charge of parsing the geometry of the molecule

        :param wfn_instance:
        :param nuclei_number:
        :return:
        """
        coordinates_lines = wfn_instance[:nuclei_number]
        atom_types = [None] * nuclei_number
        coordinates = np.zeros((nuclei_number, 3))
        charge = np.zeros(nuclei_number)
        i = 0
        for line in coordinates_lines:
            splitted_line = line.split()
            atom_types[i] = splitted_line[0].encode('ascii', 'ignore')
            coordinates[i, 0] = float(splitted_line[4])
            coordinates[i, 1] = float(splitted_line[5])
            coordinates[i, 2] = float(splitted_line[6])
            charge[i] = float(splitted_line[-1])
            i += 1
        return coordinates, atom_types, charge, wfn_instance[nuclei_number:]

    @staticmethod
    def __parse_exponents(wfn, primitives):
        """

        In charge of parsing the exponents of the basisi set

        :param wfn:
        :param primitives:
        :return:
        """
        exponents = np.zeros(primitives, dtype='float64')
        line = wfn[0]
        i = 0
        j = 0
        while line[:9] == "EXPONENTS":
            sp = line.split()
            sp = sp[1:]
            for exp in sp:
                exponents[i] = parse_fortran_scientific(exp)
                i += 1
            j += 1
            line = wfn[j]
        return exponents, wfn[j:]

    @staticmethod
    def __parse_head(wfn_instance):
        """

        In charge of parsing the head

        :param wfn_instance:
        :return:
        """
        head_line = wfn_instance[0]
        head_terms = head_line.split()
        head = dict(
            molecular_orbitals=int(head_terms[1]),
            primitives=int(head_terms[4]),
            nuclei=int(head_terms[6])
        )
        return head, wfn_instance[1:]

    @staticmethod
    def __parse_orbitals(wfn, primitives, orbitals, program):
        """

        In charge of parsing the coefficients of the wave function for each molecular orbital

        :param wfn:
        :param primitives:
        :param orbitals:
        :return:
        """
        orbital_text = ''.join(wfn)
        orbital_text = re.split('END DATA', orbital_text)[0]
        if program == 'g09':
            orbital_text = re.split(r'MO\s*\d{1,3}\s*MO\s\d\.\d\s*OCC\s*', orbital_text)
            orbital_text = orbital_text[1:]
        elif program == 'orca':
            orbital_text = re.split('MO', orbital_text)
            orbital_text = orbital_text[1:]
        coeff = np.zeros((orbitals, primitives))
        occ = np.zeros(orbitals)
        i = 0
        j = 0
        for orb in orbital_text:
            orb = re.split('NO =', orb)[1]
            orb_lines = orb.split('\n')
            header = orb_lines[0].split()
            occ[i] = float(header[0])
            orb_lines = orb_lines[1:]
            for line in orb_lines:
                sp = line.split()
                for item in sp:
                    coeff[i, j] = parse_fortran_scientific(item)
                    j += 1
            j = 0
            i += 1
        return coeff, occ

    @staticmethod
    def __parse_symmetry(wfn, primitives):
        """

        In charge of parsing the symmetry coefficients of the basis set

        :param wfn:
        :param primitives:
        :return:
        """
        symetry = np.zeros(primitives, dtype='int64')
        line = wfn[0]
        i = 0
        j = 0
        while line[:4] == "TYPE":
            sp = line.split()
            sp = sp[2:]
            for sym in sp:
                symetry[i] = int(sym) - 1
                i += 1
            j += 1
            line = wfn[j]
        return symetry, wfn[j:]

    # PUBLIC METHODS

    def eval(self, coordinates):
        """

        Performs a calculation of electron density for the given coordinates by applying:

        rho = sum_i sum_j D_ij Xi_i Xi_j = Xi.T D Xi

        where D is the value of the density matrix (obtained by multiplying the coefficents i and j
        and the occupation numbers), and Xi_i and Xi_j are gaussian functions

        :param coordinates: coordinates upon which to provide the electron density value
        :type coordinates: np.ndarray
        :return: electron density
        :rtype: np.ndarray
        """
        if self.density_matrix is None:
            raise IOError("density matrix has not been set")

        # rho = np.zeros(coordinates.shape[0])
        n = coordinates.shape[0]
        x = np.zeros((self.number_primitives, n))
        for p in range(self.number_primitives):
            cntr = self.coordinates[self.primitive_centers[p], :]
            d = coordinates - cntr
            x[p, :] = self.__gaussian(
                d,
                symmetry_index[self.symmetry_indices[p], :],
                self.exponents[p]
            )

        rho = x * (self.density_matrix @ x)
        return rho.sum(0)

    def calculate_density_matrix(self):
        """

        Calculates the density matrix

        :return:
        """
        dm = np.zeros((self.number_primitives, self.number_primitives))
        for i in range(self.number_orbitals):
            if np.isclose(self.occupancies[i], 0.0):
                continue
            for p in range(self.number_primitives):
                dm[p, :] += self.occupancies[i] * self.coefficients[i, p] * self.coefficients[i, :]
        return dm

    def get_atom_labels(self):
        """

        returns a list of the labels of the atoms ['C','H','O','C',...]

        :return: list of atomic labels
        :rtype: list
        """
        atom_labels = [atom_names[int(i) - 1] for i in self.charges]
        return atom_labels

    def get_atomic_numbers(self):
        """

        returns a list of atomic numbers in format np.ndarry

        :return: atomic numbers
        :rtype: np.ndarray
        """
        return np.array([int(i) for i in self.charges])

    def get_coordinates(self):
        """

        returns coordinates in Atomic Units

        1AU = 0.5292 Angstroms

        :return:
        """
        return self.coordinates.copy()

    def get_number_molecular_orbitals(self):
        """

        returns the number of molecular orbitals

        :return: number of molecular orbitals
        """
        return self.number_orbitals

    def get_number_primitives(self):
        """

        returns the number of primitive functions

        :return: number of primitive functions
        """
        return self.number_primitives

    def get_charge(self):
        """

        returns the total charge of the wfn

        """
        n_electrons = self.occupancies.sum()
        n_protons = self.charges.sum()
        return n_protons - n_electrons

    @staticmethod
    def from_file(filename: str, program: str, prefetch_dm=True):
        """

        reads the wave function

        """
        with open(filename) as fh:
            wfn = fh.readlines()
        wfn = wfn[1:]
        head, wfn = WaveFunction.__parse_head(wfn)
        coords, types, charges, wfn = WaveFunction.__parse_coordinates(wfn, head['nuclei'])
        centers, wfn = WaveFunction.__parse_centers(wfn, head['primitives'])
        syms, wfn = WaveFunction.__parse_symmetry(wfn, head['primitives'])
        exponents, wfn = WaveFunction.__parse_exponents(wfn, head['primitives'])
        coeff, occ = WaveFunction.__parse_orbitals(
            wfn, head['primitives'], head['molecular_orbitals'], program=program
        )

        return WaveFunction(
            coeff=coeff, dm=None, occ=occ, exponents=exponents,
            syms=syms, centers=centers, coords=coords, types=types, charges=charges,
            primitives=head['primitives'], molecular_orbitals=head['molecular_orbitals'],
            nuclei=head['nuclei'], verbose=False, prefetch_dm=prefetch_dm
        )

    @staticmethod
    def from_hdf5(cwfn: h5py.Group):
        prefetch = True
        if cwfn.attrs['contains_coefficients']:
            coeff = cwfn['coefficients'][:, :]
        else:
            coeff = None
        if cwfn.attrs['contains_density_matrix']:
            dm = cwfn['density_matrix'][:, :]
            prefetch = False
        else:
            dm = None
        occ = cwfn['occupancies'][:]
        exponents = cwfn['exponents'][:]
        syms = cwfn['symmetry_indices'][:]
        centers = cwfn['primitive_centers'][:]
        coords = cwfn['coordinates'][:, :]
        types = cwfn['atomic_symbols'][:]
        types = [i.decode('UTF-8', 'ignore') for i in types]
        charges = cwfn['charges'][:]
        primitives = cwfn.attrs['number_primitives']
        molecular_orbitals = cwfn.attrs['number_orbitals']
        nuclei = cwfn.attrs['number_centers']

        return WaveFunction(
            coeff=coeff, dm=dm, occ=occ, exponents=exponents, syms=syms, centers=centers,
            coords=coords, types=types, charges=charges, primitives=primitives,
            molecular_orbitals=molecular_orbitals, nuclei=nuclei, verbose=False, prefetch_dm=prefetch
        )

    @staticmethod
    def from_wfx(filename, program=None, prefetch_dm=True):
        return WaveFunction(**from_wfx(filename), prefetch_dm=prefetch_dm)


class ElectronDensity(Volume):
    def __init__(self, verbose=False, file=None):
        Volume.__init__(self, verbose=verbose, filename=file)

    def sample(self, number_points):
        # SAMPLING THROUGH PERMUTATION TO AVOID REPETITIONS
        size = self.shape[0] * self.shape[1] * self.shape[2]
        if number_points > size:
            raise ArithmeticError("can't sample more data than there is")
        sampled = np.arange(size)
        sampled = np.random.permutation(sampled)[:number_points]
        tnsor = self.get_volume()
        ##
        cx, cy, cz = np.unravel_index(sampled, self.shape)
        itr = np.arange(number_points)
        sampled_density = np.zeros(number_points)
        for i, x, y, z in zip(itr, cx, cy, cz):
            sampled_density[i] = tnsor[x, y, z]
        r0 = self.get_r0()
        basis = self.get_basis()
        x = np.array([cx, cy, cz], dtype='float64').T
        x += r0
        x = x.dot(basis)
        return x, sampled_density


class GaussianLog(A2MDlibQM):

    def __init__(self, file, method, charges, ep=False, verbose=True):
        """

        :param file:
        :param verbose:
        """
        from a3mdutils.parsers import std_coordinates
        from a3mdutils.parsers import npa_charges, mk_charges
        from a3mdutils.parsers import energy_decomposition, dipole
        from a3mdutils.parsers import hf_energy, mp2_energy, dft_energy
        from a3mdutils.parsers import electrostatic_potential
        A2MDlibQM.__init__(self, verbose=verbose, name='gaussianLog')
        self.fname = file
        tokkens = [
            std_coordinates, dipole, energy_decomposition
        ]
        tokkens_labels = ['coordinates', 'dipole', 'energy_decomposition']
        if method == 'MP2':
            tokkens.append(mp2_energy)
            tokkens_labels.append('energy')
        elif method == 'HF':
            tokkens.append(hf_energy)
            tokkens_labels.append('energy')
        elif method[:3] == 'dft':
            tokkens.append(lambda x: dft_energy(x, functional=method[4:]))
            tokkens_labels.append('energy')

        if charges == 'NPA':
            tokkens.append(npa_charges)
            tokkens_labels.append('charges')
        elif charges == 'MK':
            tokkens.append(mk_charges)
            tokkens_labels.append('charges')

        if ep:
            tokkens.append(electrostatic_potential)
            tokkens_labels.append('ep')

        self.tokkens = tokkens
        self.tokken_labels = tokkens_labels

    def read(self):
        """

        :return:
        """
        output_dict = dict()
        for label, tk in zip(self.tokken_labels, self.tokkens):
            try:
                output_dict[label] = self.seek(tk)
            except RuntimeError:
                self.log("missing tokken {:s}".format(label))
                output_dict[label] = None
        return output_dict

    def seek(self, fun):
        """

        :param fun:
        :return:
        """
        with open(self.fname) as f:
            return fun(f.readlines())


class CubeFile(A2MDlibQM):
    def __init__(self, file, verbose=False):
        """

        CubeFile allows to read the cube representation of electron density and store
        it in a numpy tensor.

        :param file:
        :param verbose:
        """
        A2MDlibQM.__init__(self, name='CubeFile', verbose=verbose)
        self.file = file
        self.cube_tensor = None
        self.origin = None
        self.basis = None
        self.read()

    def read(self):
        """
        Reads a cube file
        :return:
        """

        x_ori = None
        y_ori = None
        z_ori = None
        n_atoms = 1000
        x_n_values = None
        y_n_values = None
        z_n_values = None

        unit_cell = np.zeros((3, 3), dtype='float64')
        row_counter = 0
        row_buffer = None

        cube_tensor = None
        i = 0
        j = 0

        with open(self.file) as f:
            line = f.readline()
            cnt = 1
            while line:
                line = f.readline()
                cnt += 1

                if cnt == 3:
                    n_atoms, x_ori, y_ori, z_ori = line.split()
                    n_atoms = int(n_atoms)
                    x_ori = float(x_ori)
                    y_ori = float(y_ori)
                    z_ori = float(z_ori)

                    self.log(
                        "origin at : {:12.6f} {:12.6f} {:12.6f} (Angstroms)".format(
                            x_ori, y_ori, z_ori
                        )
                    )

                if cnt == 4:
                    x_n_values, xi, yi, zi = line.split()
                    x_n_values = int(x_n_values)
                    unit_cell[0, :] = float(xi), float(yi), float(zi)
                if cnt == 5:
                    y_n_values, xj, yj, zj = line.split()
                    y_n_values = int(y_n_values)
                    unit_cell[1, :] = float(xj), float(yj), float(zj)
                if cnt == 6:
                    z_n_values, xk, yk, zk = line.split()
                    z_n_values = int(z_n_values)
                    unit_cell[2, :] = float(xk), float(yk), float(zk)
                    n_values = z_n_values * x_n_values * y_n_values
                    row_buffer = np.zeros(z_n_values, dtype='float64')
                    cube_tensor = np.zeros((x_n_values, y_n_values, z_n_values), dtype='float32')
                    self.log("total points : {:12d}".format(n_values))
                if cnt > (6 + n_atoms):
                    row_length = len(line.split())
                    row_buffer[row_counter:row_counter + row_length] = list(map(float, line.split()))
                    row_counter += row_length
                    if row_counter == z_n_values:
                        cube_tensor[i, j, :] = row_buffer
                        row_counter -= row_counter
                        row_buffer -= row_buffer
                        j += 1
                        if j == y_n_values:
                            i += 1
                            j -= j

        self.cube_tensor = cube_tensor
        self.origin = np.array((x_ori, y_ori, z_ori), dtype='float64')
        self.basis = unit_cell

    def get_plane(self, value, axis='z'):
        """

        gives back the nearest layer of the tensor to the given value. Useful
        for representation of contours.

        :param value:
        :param axis:
        :return:
        """

        if axis not in ['x', 'y', 'z']:
            raise IOError("not found axis {:s}".format(axis))

        dim_idx = ['x', 'y', 'z'].index(axis)

        tensor_idx = int((value - self.origin[dim_idx]) / self.basis[dim_idx, dim_idx])

        if dim_idx == 0:
            return self.cube_tensor[tensor_idx, :, :]
        elif dim_idx == 1:
            return self.cube_tensor[:, tensor_idx, :]
        elif dim_idx == 2:
            return self.cube_tensor[:, :, tensor_idx]


class WaveFunctionHDF5:
    def __init__(self, filename, wfn_init=WaveFunction.from_hdf5, mode='r'):
        self.data = h5py.File(filename, mode)
        self.wfn_init = wfn_init

    def add(self, key: str, wfn: WaveFunction, save_dm=True, save_coeff=False):
        """

        :param key:
        :param wfn:
        :param save_dm:
        :param save_coeff:
        :return:
        """

        try:
            cwfn = self.data.create_group(key)
        except ValueError:
            print('WARNING : {:s} key already exists. Skipping.'.format(key))
            return

        cwfn.create_dataset('atomic_symbols', data=wfn.atomic_symbols)
        cwfn.create_dataset('coordinates', data=wfn.coordinates)
        cwfn.create_dataset('charges', data=wfn.charges)

        cwfn.attrs.create('number_centers', data=wfn.number_centers)
        cwfn.attrs.create('number_primitives', data=wfn.number_primitives)
        cwfn.attrs.create('number_orbitals', data=wfn.number_orbitals)
        cwfn.attrs.create('total_charge', data=wfn.total_charge)

        cwfn.create_dataset('symmetry_indices', data=wfn.symmetry_indices)
        cwfn.create_dataset('primitive_centers', data=wfn.primitive_centers)
        cwfn.create_dataset('exponents', data=wfn.exponents)
        cwfn.create_dataset('occupancies', data=wfn.occupancies)

        if save_dm:
            cwfn.attrs.create('contains_density_matrix', data=True)
            cwfn.create_dataset('density_matrix', data=wfn.density_matrix)
        else:
            cwfn.attrs.create('contains_density_matrix', data=False)

        if save_coeff:
            cwfn.attrs.create('contains_coefficients', data=True)
            cwfn.create_dataset('coefficients', data=wfn.coefficients)
        else:
            cwfn.attrs['contains_coefficients'] = False

    def iterall(self):
        for key, item in self.data.items():
            yield key, self.wfn_init(cwfn=item)

    def __getitem__(self, key):
        try:
            cwfn = self.data[key]
        except KeyError:
            raise IOError("wfn {:s} not found within hdf5 file".format(key))
        return key, self.wfn_init(cwfn=cwfn)

    def close(self):
        self.data.close()


if __name__ == '__main__':
    pass
