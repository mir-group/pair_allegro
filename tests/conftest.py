import pytest

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import yaml
import warnings

import torch

from nequip.utils import Config
from nequip.data import dataset_from_config

TESTS_DIR = Path(__file__).resolve().parent
LAMMPS = os.environ.get("LAMMPS", "lmp")
# Allow user to specify prefix to set up environment before mpirun. For example,
# using `LAMMPS_ENV_PREFIX="conda run -n whatever"` to run LAMMPS in a different
# conda environment.
LAMMPS_ENV_PREFIX = os.environ.get("LAMMPS_ENV_PREFIX", "")
_lmp_help = subprocess.run(
    " ".join([LAMMPS_ENV_PREFIX, "mpirun", LAMMPS, "-h"]),
    shell="True",
    stdout=subprocess.PIPE,
    check=True,
).stdout
HAS_KOKKOS: bool = b"allegro/kk" in _lmp_help
HAS_KOKKOS_CUDA: bool = b"KOKKOS package API: CUDA" in _lmp_help
HAS_OPENMP: bool = b"OPENMP" in _lmp_help

if not HAS_KOKKOS:
    warnings.warn("Not testing pair_allegro with Kokkos since it wasn't built with it")
if HAS_KOKKOS and torch.cuda.is_available() and not HAS_KOKKOS_CUDA:
    warnings.warn("Kokkos not built with CUDA even though CUDA is available")
if not HAS_OPENMP:
    warnings.warn(
        "Not testing pair_allegro with OpenMP since LAMMPS wasn't built with the OPENMP package"
    )


@pytest.fixture(
    params=[
        ("CuPd-cubic-big.xyz", "CuPd", ["Cu", "Pd"], 5.1, n_rank)
        for n_rank in (1, 4, 8)
    ]
    + [("Cu-cubic.xyz", "Cu", ["Cu"], 4.5, n_rank) for n_rank in (1, 2)]
    + [
        ("aspirin.xyz", "aspirin", ["C", "H", "O"], 4.0, 1),
        ("aspirin.xyz", "aspirin", ["C", "H", "O"], 15.0, 1),
        ("Cu2AgO4.xyz", "mp-1225882", ["Cu", "Ag", "O"], 4.9, 1),
        ("Cu-cubic.xyz", "Cu", ["Cu"], 15.5, 1),
    ],
    scope="session",
)
def dataset_options(request):
    out = dict(
        zip(
            ["dataset_file_name", "run_name", "chemical_symbols", "r_max"],
            request.param,
        )
    )
    out["dataset_file_name"] = TESTS_DIR / ("test_data/" + out["dataset_file_name"])
    return out, request.param[-1]


@pytest.fixture(
    params=[
        187382,
        109109,
    ],
    scope="session",
)
def model_seed(request):
    return request.param


def _check_and_print(retcode):
    __tracebackhide__ = True
    if retcode.returncode:
        if len(retcode.stdout) > 0:
            print(retcode.stdout.decode("ascii"))
        if len(retcode.stderr) > 0:
            print(retcode.stderr.decode("ascii"), file=sys.stderr)
        retcode.check_returncode()


@pytest.fixture(scope="session")
def deployed_model(model_seed, dataset_options):
    dataset_options, n_rank = dataset_options
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config.from_file(str(TESTS_DIR / "test_data/test_repro.yaml"))
        config.update(dataset_options)
        config["seed"] = model_seed
        config["root"] = tmpdir + "/root"
        configpath = tmpdir + "/config.yaml"
        with open(configpath, "w") as f:
            yaml.dump(dict(config), f)
        # run a nequip-train command
        retcode = subprocess.run(
            ["nequip-train", configpath],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)
        # run nequip-deploy
        deployed_path = tmpdir + "/deployed.pth"
        retcode = subprocess.run(
            [
                "nequip-deploy",
                "build",
                "--train-dir",
                config["root"] + "/" + config["run_name"],
                deployed_path,
            ],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        _check_and_print(retcode)
        # load structures to test on
        d = dataset_from_config(config)
        # take some frames
        structures = [d[i].to_ase(type_mapper=d.type_mapper) for i in range(5)]
        # give them cells even if nonperiodic
        if not all(structures[0].pbc):
            L = 50.0
            for struct in structures:
                struct.cell = L * np.eye(3)
                struct.center()
        for s in structures:
            s.rattle(stdev=0.2)
            s.wrap()
        structures = structures[:1]
        yield deployed_path, structures, config, n_rank
