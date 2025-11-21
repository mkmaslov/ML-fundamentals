#!/bin/bash

set -e

# -----------------------------------------------------------------------------
# Script for downloading Kaggle data.
# Usage: kaggle.sh <competition_handle>
# - check pre-requsites: virtual env, pip and kaggle packages
# - download project to "../data/<competition_handle>/"
# -----------------------------------------------------------------------------

# Highlight the output.
YELLOW="\e[1;33m" ; RED="\e[1;31m" ; GREEN="\e[1;32m" ; COLOR_OFF="\e[0m"
cprint() { echo -ne "${1}${2}${COLOR_OFF}\n"; }
error() { cprint ${RED} "Error: ${1}"; }

# Instructions for running the script.
[ "$1" = "--help" ] \
	&& cprint "Usage: kaggle.sh <competition_handle>" && exit 0

# Path to virtual environment.
PYTHON_VENV="${HOME}/.python_venv"

# Verify if virtual environment exists and pip is installed.
[ ! -d "${PYTHON_VENV}" ] && error "\"${PYTHON_VENV}\" not found!" && exit 1
! command -v "${PYTHON_VENV}/bin/pip" >/dev/null 2>&1 \
	&& error "\"${PYTHON_VENV}/bin/pip\" not found!" && exit 1

# Install Kaggle package if not installed.
! command -v "${PYTHON_VENV}/bin/kaggle" >/dev/null 2>&1 \
	&& "${PYTHON_VENV}/bin/pip" --require-virtualenv install kaggle

# Path to repository's root directory, i.e., parent directory of this script.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Verify if Kaggle config exists.
export KAGGLE_CONFIG_DIR="${ROOT_DIR}/.kaggle/"
[ ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ] \
	&& error "\"${KAGGLE_CONFIG_DIR}/kaggle.json\" not found!" && exit 1

# Download competition data.
DOWNLOAD_DIR="${ROOT_DIR}/data/$1"
if [ -d "${DOWNLOAD_DIR}" ]; then
	error "folder \"${DOWNLOAD_DIR}\" already exists!"
else
	mkdir -p "${DOWNLOAD_DIR}"
	"${PYTHON_VENV}/bin/kaggle" competitions download -c "$1" -p "${DOWNLOAD_DIR}"
	unzip -q "${DOWNLOAD_DIR}/$1.zip" -d "${DOWNLOAD_DIR}"
	rm "${DOWNLOAD_DIR}/$1.zip"
fi