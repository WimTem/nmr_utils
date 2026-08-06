from pathlib import Path
from ase.io import read


def iter_snapshots(root):
    """
    Yield every .magres structure found under root.

    Directory structure:

        root/
            frame0001/
                CONTCAR
                calc.magres
    """

    root = Path(root)

    for directory in sorted(root.iterdir()):

        if not directory.is_dir():
            continue

        magres = list(directory.glob("*.magres"))

        if len(magres) != 1:
            continue

        yield read(magres[0])