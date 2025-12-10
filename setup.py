"""Setup repo"""
import logging
# import os
from pathlib import Path
from setuptools import setup
from typing import List


# LOGGER
logger = logging.getLogger(__name__)


# DEFAULT
CLONE_URL = r'https://github.com/steeve-boucherie/pygp-sandbox.git'


# UTILS
def get_base() -> Path:
    """
    Return path to the base folder of the repository

    Returns
    -------
        Path
    """
    return Path(__file__).resolve().parent


def long_description() -> str:
    """
    Read the package description from the README file, \
        if it exists.

    Returns
    -------
        str
    """
    path = get_base() / 'README.md'
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            desc = f.read()

    else:
        desc = ''

    return desc


def env_template() -> List[str]:
    """
    Read the template environment file.

    Returns
    -------
        List[str]

    Raises
    ------
        FileNotFoundError
    """
    path = get_base() / '.config' / 'env_template'
    if not path.exists():
        msg = f'Cannot find template file "{path.name}" in config folder: ' \
              f'{str(path.parent)}. Your local repo might be broken.\n' \
              f'Clone the repo from {CLONE_URL}, and start the installation process again.'
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(path, 'r') as f:
        lines = f.readlines()

    return lines


def update_env(lines: List[str]) -> List[str]:
    """
    Update the environment file with the right paths.

    Notes
    -----
    Updates the path to the "base" folder.

    Parameters
    ----------
    lines: List[str]
        A list of strings, the content of the template env file.

    Returns
    -------
        List[str]
    """
    pattern = 'BASE='
    ind = [
        n for n, line in enumerate(lines)
        if pattern in line
    ]
    if len(ind) != 1:
        msg = 'Something is wrong with you env file template...\n' \
              f'Clone the repo from {CLONE_URL}, and rerun the installation process.'
        logger.error(msg)
        raise ValueError(msg)

    ind = ind[0]
    lines[ind] = pattern + str(get_base())

    return lines


def create_env() -> None:
    """Create the .env file in the base folder using the templaten \
        and updating the paths."""
    # Get the template
    lines = env_template()

    # Update
    lines = update_env(lines)

    # Write to file
    with open(get_base() / '.env', 'w') as f:
        f.writelines(lines)


# SETUP
setup()


# Create env file
create_env()
