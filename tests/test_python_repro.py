import pytest

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import yaml
import textwrap
import warnings
from io import StringIO
from collections import Counter

import ase
import ase.build
import ase.io

import torch

from nequip.ase import NequIPCalculator
from nequip.utils import Config
from nequip.data import dataset_from_config, AtomicData, AtomicDataDict

TESTS_DIR = Path(__file__).resolve().parent
LAMMPS = os.environ.get("LAMMPS", "lmp")
_lmp_help = subprocess.run([LAMMPS, "-h"], stdout=subprocess.PIPE, check=True).stdout
HAS_KOKKOS: bool = b"allegro/kk" in _lmp_help
HAS_OPENMP: bool = b"OPENMP" in _lmp_help

if not HAS_KOKKOS:
    warnings.warn("Not testing pair_allegro with Kokkos since it wasn't built with it")
if not HAS_OPENMP:
    warnings.warn(
        "Not testing pair_allegro with OpenMP since LAMMPS wasn't built with the OPENMP package"
    )


@pytest.fixture(
    params=[
        ("CuPd-cubic-big.xyz", "CuPd", ["Cu", "Pd"], 5.1, n_rank)
        for n_rank in (1, 2, 4)
    ]
    + [
        ("aspirin.xyz", "aspirin", ["C", "H", "O"], 4.0, 1),
        ("aspirin.xyz", "aspirin", ["C", "H", "O"], 15.0, 1),
        ("Cu2AgO4.xyz", "mp-1225882", ["Cu", "Ag", "O"], 4.9, 1),
        ("Cu-cubic.xyz", "Cu", ["Cu"], 4.5, 1),
        ("Cu-cubic.xyz", "Cu", ["Cu"], 15.5, 1),
    ],
    scope="module",
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
    scope="module",
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


@pytest.fixture(scope="module")
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


@pytest.mark.parametrize(
    "kokkos,openmp",
    [(False, False)]
    + ([(False, True)] if HAS_OPENMP else [])
    + ([(True, False)] if HAS_KOKKOS else []),
)
def test_repro(deployed_model, kokkos: bool, openmp: bool):
    structure: ase.Atoms
    deployed_model: str
    deployed_model, structures, config, n_rank = deployed_model
    num_types = len(config["chemical_symbols"])

    calc = NequIPCalculator.from_deployed_model(
        deployed_model,
        set_global_options=True,
        species_to_type_name={s: s for s in config["chemical_symbols"]},
    )

    newline = "\n"
    periodic = all(structures[0].pbc)
    PRECISION_CONST: float = 1e6
    lmp_in = textwrap.dedent(
        f"""
        units		metal
        atom_style	atomic
        newton on
        thermo 1

        # get a box defined before pair_coeff
        {'boundary p p p' if periodic else 'boundary s s s'}

        read_data structure.data

        pair_style	allegro
        # note that ASE outputs lammps types in alphabetical order of chemical symbols
        # since we use chem symbols in this test, just put the same
        pair_coeff	* * {deployed_model} {' '.join(sorted(set(config["chemical_symbols"])))}
{newline.join('        mass  %i 1.0' % i for i in range(1, num_types + 1))}

        neighbor	1.0 bin
        neigh_modify    delay 0 every 1 check no

        fix		1 all nve

        timestep	0.001

        compute atomicenergies all pe/atom
        compute totalatomicenergy all reduce sum c_atomicenergies

        thermo_style custom step time temp pe c_totalatomicenergy etotal press spcpu cpuremain
        run 0
        print $({PRECISION_CONST} * pe) file pe.dat
        print $({PRECISION_CONST} * c_totalatomicenergy) file totalatomicenergy.dat
        write_dump all custom output.dump id type x y z fx fy fz c_atomicenergies modify format float %20.15g
        """
    )

    # for each model,structure pair
    # build a LAMMPS input using that structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # save out the LAMMPS input:
        infile_path = tmpdir + "/test_repro.in"
        with open(infile_path, "w") as f:
            f.write(lmp_in)
        # environment variables
        env = dict(os.environ)
        env["ALLEGRO_DEBUG"] = "true"
        # save out the structure
        for i, structure in enumerate(structures):
            ase.io.write(
                tmpdir + f"/structure.data",
                structure,
                format="lammps-data",
            )

            # run LAMMPS
            OMP_NUM_THREADS = 4  # just some choice
            retcode = subprocess.run(
                # MPI options if MPI
                (["mpirun", "-np", str(n_rank)] if n_rank > 1 else [])
                # LAMMPS exec
                + [LAMMPS]
                # Kokkos options if Kokkos
                + (
                    [
                        "-sf",
                        "kk",
                        "-k",
                        "on",
                        ("g" if torch.cuda.is_available() else "t"),
                        str(
                            max(torch.cuda.device_count() // n_rank, 1)
                            if torch.cuda.is_available()
                            else OMP_NUM_THREADS
                        ),
                        "-pk",
                        "kokkos newton on neigh full",
                    ]
                    if kokkos
                    else []
                )
                # OpenMP options if openmp
                + (["-sf", "omp", "-pk", "omp", str(OMP_NUM_THREADS)] if openmp else [])
                # input
                + ["-in", infile_path],
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _check_and_print(retcode)

            # Check the inputs:
            if n_rank == 1:
                # TODO: this will only make sense with one rank
                # load debug data:
                mi = None
                lammps_stdout = iter(retcode.stdout.decode("utf-8").splitlines())
                line = next(lammps_stdout, None)
                while line is not None:
                    if line.startswith("Allegro edges: i j rij"):
                        edges = []
                        while not line.startswith("end Allegro edges"):
                            line = next(lammps_stdout)
                            edges.append(line)
                        edges = np.loadtxt(StringIO("\n".join(edges[:-1])))
                        mi = edges
                        break
                    line = next(lammps_stdout)
                mi = {
                    "i": mi[:, 0:1].astype(int),
                    "j": mi[:, 1:2].astype(int),
                    "rij": mi[:, 2:],
                }

                # first, check the model INPUTS
                structure_data = AtomicData.to_AtomicDataDict(
                    AtomicData.from_ase(structure, r_max=float(config["r_max"]))
                )
                structure_data = AtomicDataDict.with_edge_vectors(
                    structure_data, with_lengths=True
                )
                lammps_edge_tuples = [
                    tuple(e)
                    for e in np.hstack(
                        (
                            mi["i"],
                            mi["j"],
                        )
                    )
                ]
                nq_edge_tuples = [
                    tuple(e.tolist())
                    for e in structure_data[AtomicDataDict.EDGE_INDEX_KEY].t()
                ]
                # same num edges
                assert len(lammps_edge_tuples) == len(nq_edge_tuples)
                if kokkos:
                    # In the kokkos version, the atom ij are not tags
                    # so the order can't be compared to nequip
                    # so we just check overall set quantities instead
                    # this is slightly less stringent but should still catch problems
                    # check counters of per-atom num edges are same
                    assert Counter(
                        np.bincount(mi["i"].reshape(-1)).tolist()
                    ) == Counter(
                        torch.bincount(
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][0]
                        ).tolist()
                    )
                    # check OVERALL "set" of pairwise distance is good
                    nq_rij = structure_data[AtomicDataDict.EDGE_LENGTH_KEY].clone()
                    nq_rij, _ = nq_rij.sort()
                    lammps_rij = mi["rij"].copy().squeeze(-1)
                    lammps_rij.sort()
                    assert np.allclose(nq_rij, lammps_rij)
                else:
                    # check same number of i,j edges across both
                    assert Counter(e[:2] for e in lammps_edge_tuples) == Counter(
                        e[:2] for e in nq_edge_tuples
                    )
                    # finally, check for each ij whether the the "sets" of edge lengths match
                    nq_ijr = np.core.records.fromarrays(
                        (
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][0],
                            structure_data[AtomicDataDict.EDGE_INDEX_KEY][1],
                            structure_data[AtomicDataDict.EDGE_LENGTH_KEY],
                        ),
                        names="i,j,rij",
                    )
                    # we can do "set" comparisons by sorting into groups by ij,
                    # and then sorting the rij _within_ each ij pair---
                    # this is what `order` does for us with the record array
                    nq_ijr.sort(order=["i", "j", "rij"])
                    lammps_ijr = np.core.records.fromarrays(
                        (
                            mi["i"].reshape(-1),
                            mi["j"].reshape(-1),
                            mi["rij"].reshape(-1),
                        ),
                        names="i,j,rij",
                    )
                    lammps_ijr.sort(order=["i", "j", "rij"])
                    assert np.allclose(nq_ijr["rij"], lammps_ijr["rij"])

            # load dumped data
            lammps_result = ase.io.read(
                tmpdir + f"/output.dump", format="lammps-dump-text"
            )

            # --- now check the OUTPUTS ---
            structure.calc = calc

            # check output atomic quantities
            print(
                f"Max force error: {np.abs(structure.get_forces() - lammps_result.get_forces()).max()}"
            )
            assert np.allclose(
                structure.get_forces(),
                lammps_result.get_forces(),
                atol=1e-4,
            )
            print(
                f"Max atomic energy error: {np.abs(structure.get_potential_energies() - lammps_result.arrays['c_atomicenergies'].reshape(-1)).max()}"
            )
            assert np.allclose(
                structure.get_potential_energies(),
                lammps_result.arrays["c_atomicenergies"].reshape(-1),
                atol=5e-5,
            )

            # check system quantities
            lammps_pe = float(Path(tmpdir + f"/pe.dat").read_text()) / PRECISION_CONST
            lammps_totalatomicenergy = (
                float(Path(tmpdir + f"/totalatomicenergy.dat").read_text())
                / PRECISION_CONST
            )
            assert np.allclose(lammps_pe, lammps_totalatomicenergy)
            assert np.allclose(
                structure.get_potential_energy(),
                lammps_pe,
                atol=1e-6,
            )
