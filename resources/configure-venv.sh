#!/bin/bash

set -e

# -----------------------------------------------------------------------------
# Script to configure Python virtual environment.
# Usage: configure-venv.sh
# - set up a virtual environment at "${HOME}/.python_venv"
# - install all the necessary packages from "pip"
# - create an iPython kernel that works with Jupyter from virtual environment
# -----------------------------------------------------------------------------

# Cancel installation if run as a root.
if [ "$EUID" -eq 0 ]; then
    echo "Do NOT run this script as a root!"
    exit
fi

# Set up a virtual environment for Python.
PYDIR="${HOME}/.python_venv"
PIP="${PYDIR}/bin/pip"
PYTHON="${PYDIR}/bin/python"
mkdir -p ${PYDIR}/cache
cat > ${PYDIR}/pip.conf <<EOF
    [global]
    cache-dir=${PYDIR}/cache
EOF
# Activate virtual environment.
python -m venv ${PYDIR}

# Update pip.
${PIP} install --upgrade pip --require-virtualenv

# Install packages.
${PIP} install --require-virtualenv \
    numpy scipy matplotlib sympy pandas h5py torch ipykernel

# Create an iPython kernel.
${PYTHON} -m ipykernel install --user --name kernel_venv --display-name "Python (venv)"