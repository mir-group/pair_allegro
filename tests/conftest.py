import pytest

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import warnings

import torch

from nequip.data import to_ase
from nequip.utils.global_dtype import _GLOBAL_DTYPE
from nequip.ase import NequIPCalculator
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

from typing import Dict, Final

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

COMPILE_MODES: Final[Dict[str, str]] = {
    "torchscript": "test.nequip.pth",
    "aotinductor": "test.nequip.pt2",
}


@pytest.fixture(
    params=[
        ("CuPd-cubic-big.xyz", ["Cu", "Pd"], 5.0),
        ("aspirin.xyz", ["C", "H", "O"], 5.0),
        ("aspirin.xyz", ["C", "H", "O"], 15.0),
        ("Cu2AgO4.xyz", ["Cu", "Ag", "O"], 5.0),
        ("Cu-cubic.xyz", ["Cu"], 5.0),
        ("Cu-cubic.xyz", ["Cu"], 15.0),
    ],
    scope="session",
)
def dataset_options(request):
    out = dict(
        zip(
            ["dataset_file_name", "chemical_symbols", "cutoff_radius"],
            request.param,
        )
    )
    out["dataset_file_name"] = TESTS_DIR / ("test_data/" + out["dataset_file_name"])
    return out


@pytest.fixture(
    params=["float32", "float64"],
    scope="session",
)
def model_dtype(request):
    return request.param


def _check_and_print(retcode, encoding="ascii"):
    __tracebackhide__ = True
    if retcode.returncode:
        if len(retcode.stdout) > 0:
            print(retcode.stdout.decode(encoding, errors="replace"))
        if len(retcode.stderr) > 0:
            print(retcode.stderr.decode(encoding, errors="replace"), file=sys.stderr)
        retcode.check_returncode()


@pytest.fixture(scope="session")
def deployed_allegro_model(model_dtype, dataset_options):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield deployed_model("allegro", tmpdir, model_dtype, dataset_options)


@pytest.fixture(scope="session")
def deployed_nequip_model(model_dtype, dataset_options):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield deployed_model("nequip", tmpdir, model_dtype, dataset_options)


def deployed_model(nequip_or_allegro, tmpdir, dtype, dataset_options):

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices += ["cuda"]

    # === set up options and tolerances ===
    tol = {"float32": 5e-4, "float64": 1e-8}[dtype]
    # aspirin (sGDML) data is in kcal/mol and 1 eV = 23 kcal/mol
    if dataset_options["dataset_file_name"] == "aspirin.xyz":
        tol *= 23

    # === setup config from template ===
    config = OmegaConf.load(
        str(TESTS_DIR / f"test_data/test_repro_{nequip_or_allegro}.yaml")
    )
    with open_dict(config):
        # the checkpoint file `last.ckpt` will be located in the hydra runtime directory
        # so we set it to the tmpdir
        config["hydra"] = {"run": {"dir": tmpdir}}
        config["model_dtype"] = dtype
        config.update(dataset_options)
    config = OmegaConf.create(config)
    configpath = tmpdir + "/config.yaml"
    OmegaConf.save(config=config, f=configpath)
    # === run `nequip-train` to get checkpoint ===
    retcode = subprocess.run(
        ["nequip-train", "-cn", "config"],
        cwd=tmpdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    checkpoint_path = tmpdir + "/last.ckpt"

    # === run `nequip-compile` for both `.pth` and `.pt2` ===
    # one training run for both torchscript and aotinductor tests
    for mode, filename in COMPILE_MODES.items():
        for device in devices:
            command = [
                "nequip-compile",
                "--input-path",
                checkpoint_path,
                "--output-path",
                tmpdir + "/" + f"{device}_" + filename,
                "--mode",
                mode,
                "--device",
                device,
                # target accepted as argument for both modes, but unused for torchscript mode
                "--target",
                f"pair_{nequip_or_allegro}",
            ]
            print(command)
            retcode = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(
                    os.environ,
                    NEQUIP_FLOAT32_MODEL_TOL=str(tol),
                    NEQUIP_FLOAT64_MODEL_TOL=str(tol),
                ),
            )
            _check_and_print(retcode, "utf-8")

    # === get the `test` dataset (5 frames) ===
    torch.set_default_dtype(_GLOBAL_DTYPE)
    datamodule = instantiate(config.data, _recursive_=False)
    datamodule.prepare_data()
    datamodule.setup("test")
    dloader = datamodule.test_dataloader()[0]

    structures = []
    for data in dloader:
        # `to_ase` returns a List because data from datamodule's dataloader is batched (trivially with batch size 1)
        structures += to_ase(data)

    # give them cells even if nonperiodic
    if not all(structures[0].pbc):
        L = 50.0
        for struct in structures:
            struct.cell = L * np.eye(3)
            struct.center()
    for s in structures:
        # wrapping is extremely important for the nequip tests
        s.wrap()
    structures = structures[:1]

    calc = NequIPCalculator.from_checkpoint_model(
        checkpoint_path,
        set_global_options=False,
        species_to_type_name={s: s for s in config["chemical_symbols"]},
    )
    return tmpdir, calc, structures, config, tol
