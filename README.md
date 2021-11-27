# A3MD - utils

**Important**: I am building this repository from pieces of a previous repository. This means
*I still have to run some tests to check I didn't mess up*.

## Summary

A3MDutils provides some useful classes and scripts that I developped
for my main project, A3MD. You can employ it to:
- generate inputs for ORCA and Gaussian
- compile QM electron density information in H5 files 
- rotate coordinates of MOL2 files
- extract forces from QM files

## Install

I am only providing manual build

    python3 -m build

This generates some files in dist/ that you can install with pip.

## Usage

To interact with these scripts, you can just call the a3mdutils script.

    >>> a3mdutils 
    Usage: a3mdutils [OPTIONS] COMMAND [ARGS]...

    a set of scripts to work with molecular representations and their electron densities


    Options:
    --help  Show this message and exit.

    Commands:
    compile-mol2         Generates a json file containing molecular...
    compile-wfn          Generates an HDF5 file contaning all the...
    convert-sample       Converts a .csv file into an .npy file, and viceversa
    extract-forces       Extracts the values of forces from a G09 output...
    many-compile-mol2    Generates a json file containing molecular...
    many-compile-wfn     Generates an HDF5 file contaning the information...
    many-convert-sample  Converts many .csv files into an .npy file, and...
    many-extract-forces  Extracts the values of forces from many G09...
    many-prepare-qm      Creates inputs for QM programs like Gaussian09 and...
    many-relabel-mol2    Applies a relabelling of the atoms of many Mol2.
    many-update-mol2     Generates new mol2 files by including charge...
    merge-sources        Generates and HDF5 containing the information of...
    prepare-qm           Creates inputs for QM programs like Gaussian09 and...
    random-rotation      Rotates a mol2 file
    relabel-mol2         Applies a relabelling of the atoms of a Mol2.
    update-mol2          Generates new mol2 files by including charge...

## Support

Right now, I cannot support this repository but from time to time. But don't
hesistate to contact me if you find something that does not work at 
brunocuevaszuviria at gmail.com
