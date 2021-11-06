import numpy as np


def parse_scalar_text(contents):
    return contents[0].strip()


def parse_scalar_int_number(contents):
    return int(contents[0].strip())


def parse_vector_int_number(contents):
    values = []
    for line in contents:
        values += line.strip().split()
    values = np.array([int(i) - 1 for i in values], dtype='int32')
    return values


def parse_vector_float_number(contents):
    values = []
    for line in contents:
        values += line.strip().split()
    values = np.array([float(i) for i in values], dtype='float64')
    return values


def parse_vector_text(contents):
    values = []
    for line in contents:
        values.append(line.strip()[0].encode('ascii', 'ignore'))
    return values


def parse_orbital_coefficients(contents):
    coefficients = []
    coefficients_split = '\n'.join(contents).split('<MO Number>')[1:]
    for i, block in enumerate(coefficients_split):
        block = block.split('\n')[3:]
        x = parse_vector_float_number(block)
        coefficients.append(x)
    coefficients = np.array(coefficients, dtype='float64')
    return coefficients


def parse_coordinates(contents):
    coordinates = []
    for line in contents:
        coordinates.append([float(i) for i in line.split()])
    coordinates = np.array(coordinates, dtype='float64')
    return coordinates

fields = dict(
    coeff=dict(label='Molecular Orbital Primitive Coefficients', fun=parse_orbital_coefficients),
    occ=dict(label='Molecular Orbital Occupation Numbers', fun=parse_vector_float_number),
    exponents=dict(label='Primitive Exponents', fun=parse_vector_float_number),
    syms=dict(label='Primitive Types', fun=parse_vector_int_number),
    centers=dict(label='Primitive Centers', fun=parse_vector_int_number),
    coords=dict(label='Nuclear Cartesian Coordinates', fun=parse_coordinates),
    types=dict(label='Nuclear Names', fun=parse_vector_text),
    charges=dict(label='Nuclear Charges', fun=parse_vector_float_number),
    primitives=dict(label='Number of Primitives', fun=parse_scalar_int_number),
    molecular_orbitals=dict(label='Number of Occupied Molecular Orbitals', fun=parse_scalar_int_number),
    nuclei=dict(label='Number of Nuclei', fun=parse_scalar_int_number),

    # title=dict(label='Title', fun=parse_scalar_text),
    # nelec=dict(label='Number of Electrons', fun=parse_scalar_int_number),
    # naelec=dict(label='Number of Alpha Electrons', fun=parse_scalar_int_number),
    # nbelec=dict(label='Number of Beta Electrons', fun=parse_scalar_int_number),
    # mult=dict(label='Electron Spin Multiplicity', fun=parse_scalar_int_number),
    # ncelec=dict(label='Number of Core Electrons', fun=parse_scalar_int_number),
    # energies=dict(label='Molecualr Orbital Energies', fun=parse_vector_float_number),
    # spins=dict(label='Molecular Orbital Spin Types', fun=parse_vector_text)
)


def split_by_labels(contents):
    flag = False
    block = None
    label = None
    for i, line in enumerate(contents):
        line = line.strip()
        if line[0] == '<' and line[1] != '/' and flag == False:
            flag = True
            block = []
            label = line.strip().replace('<', '').replace('>', '')
        elif line == '</{:s}>'.format(label) and flag == True:
            flag = False
            yield label, block
        elif flag:
            block.append(line)


def match_block(label, block, fields_dict):
    for key, item in fields_dict.items():
        if item['label'] == label:
            return key, item['fun'](block)
    return None, None


def from_wfx(file):
    with open(file) as f:
        contents = f.readlines()

    contents = [i.strip() for i in contents]
    contents = [i for i in contents if len(i) > 0]
    contents = [i for i in contents if i[:7] != 'Warning']
    contents = [i for i in contents if i[0] != '#']
    wavefunction = dict()
    for label, block in split_by_labels(contents):
        l, u = match_block(label, block, fields)
        if u is not None:
            wavefunction[l] = u
    wavefunction['dm'] = None
    return wavefunction
