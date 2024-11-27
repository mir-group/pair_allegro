import pytest

import os
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import textwrap
from io import StringIO
from collections import Counter

import ase
import ase.build
import ase.io

import torch

from nequip.data import AtomicDataDict, from_ase, compute_neighborlist_
from nequip.nn import with_edge_vectors_


from conftest import (
    _check_and_print,
    LAMMPS,
    LAMMPS_ENV_PREFIX,
    HAS_KOKKOS,
    HAS_KOKKOS_CUDA,
    HAS_OPENMP,
    COMPILE_MODES,
)


@pytest.mark.parametrize(
    "kokkos,openmp",
    [(False, False)]
    + ([(False, True)] if HAS_OPENMP else [])
    + ([(True, False)] if HAS_KOKKOS else []),
)
@pytest.mark.parametrize(
    "compile_mode",
    # i.e. torchscript or aotinductor
    list(COMPILE_MODES.keys()),
)
@pytest.mark.parametrize(
    "n_rank",
    [1, 2, 4],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_repro(
    deployed_allegro_model,
    kokkos: bool,
    openmp: bool,
    compile_mode: str,
    n_rank: int,
    device: str,
):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not detected, skipping `devive=cuda` tests")
    structure: ase.Atoms
    model_tmpdir, calc, structures, config, tol = deployed_allegro_model
    model_file_path = model_tmpdir + f"/{device}_" + COMPILE_MODES[compile_mode]

    # TODO: how to run CPU tests with GPU enabled Kokkos?
    if kokkos:
        if HAS_KOKKOS_CUDA and device == "cpu":
            pytest.skip(
                "Kokkos compiled with GPU-enabled backend, skipping CPU + Kokkos tests"
            )

    # decide which tests to use `n_rank` > 1
    if n_rank > 1:
        data_name = config["dataset_file_name"]
        r_max = float(config["cutoff_radius"])
        if not (data_name in ["CuPd-cubic-big.xyz", "Cu-cubic.xyz"] and r_max < 8.0):
            pytest.skip(
                f"skipping `n_rank={n_rank}` Allegro test for {data_name} and `r_max={r_max}`"
            )

    num_types = len(config["chemical_symbols"])

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
        pair_coeff	* * {model_file_path} {' '.join(sorted(set(config["chemical_symbols"])))}
{newline.join('        mass  %i 1.0' % i for i in range(1, num_types + 1))}

        neighbor	1.0 bin
        neigh_modify    delay 0 every 1 check no

        fix		1 all nve

        timestep	0.001

        compute atomicenergies all pe/atom
        compute totalatomicenergy all reduce sum c_atomicenergies
        compute stress all pressure NULL virial  # NULL means without temperature contribution

        thermo_style custom step time temp pe c_totalatomicenergy etotal press spcpu cpuremain c_stress[*]
        run 0
        print "$({PRECISION_CONST} * c_stress[1]) $({PRECISION_CONST} * c_stress[2]) $({PRECISION_CONST} * c_stress[3]) $({PRECISION_CONST} * c_stress[4]) $({PRECISION_CONST} * c_stress[5]) $({PRECISION_CONST} * c_stress[6])" file stress.dat
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
        env["_NEQUIP_LOG_LEVEL"] = "DEBUG"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

        # save out the structure
        for i, structure in enumerate(structures):
            ase.io.write(
                tmpdir + "/structure.data",
                structure,
                format="lammps-data",
            )

            # run LAMMPS
            OMP_NUM_THREADS = 4  # just some choice
            retcode = subprocess.run(
                " ".join(
                    # Allow user to specify prefix to set up environment before mpirun. For example,
                    # using `LAMMPS_ENV_PREFIX="conda run -n whatever"` to run LAMMPS in a different
                    # conda environment.
                    [LAMMPS_ENV_PREFIX]
                    +
                    # MPI options if MPI
                    # --oversubscribe necessary for GitHub Actions since it only gives 2 slots
                    # > Alternatively, you can use the --oversubscribe option to ignore the
                    # > number of available slots when deciding the number of processes to
                    # > launch.
                    ["mpirun", "--oversubscribe", "-np", str(n_rank), LAMMPS]
                    # Kokkos options if Kokkos
                    + (
                        [
                            "-sf",
                            "kk",
                            "-k",
                            "on",
                            ("g" if HAS_KOKKOS_CUDA else "t"),
                            str(
                                max(torch.cuda.device_count() // n_rank, 1)
                                if HAS_KOKKOS_CUDA
                                else OMP_NUM_THREADS
                            ),
                            "-pk",
                            "kokkos newton on neigh half",
                        ]
                        if kokkos
                        else []
                    )
                    # OpenMP options if openmp
                    + (
                        ["-sf", "omp", "-pk", "omp", str(OMP_NUM_THREADS)]
                        if openmp
                        else []
                    )
                    # input
                    + ["-in", infile_path]
                ),
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )

            # uncomment to view LAMMPS output
            _check_and_print(retcode)

            # Check the inputs:
            if n_rank == 1:
                # this will only make sense with one rank
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
                structure_data = from_ase(structure)
                structure_data = compute_neighborlist_(
                    structure_data, r_max=float(config.cutoff_radius)
                )
                structure_data = with_edge_vectors_(structure_data, with_lengths=True)
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
                tmpdir + "/output.dump", format="lammps-dump-text"
            )

            # --- now check the OUTPUTS ---
            structure.calc = calc

            # check output atomic quantities
            max_force_err = np.max(
                np.abs(structure.get_forces() - lammps_result.get_forces())
            )
            max_force_comp = np.max(np.abs(structure.get_forces()))
            force_rms = np.sqrt(np.mean(np.square(structure.get_forces())))
            assert np.allclose(
                structure.get_forces(),
                lammps_result.get_forces(),
                atol=tol,
                rtol=tol,
            ), f"Force max abs err: {max_force_err:.8g} (atol/rtol={tol:.3g}). Max force component: {max_force_comp}, Force RMS: {force_rms}"
            assert np.allclose(
                structure.get_potential_energies(),
                lammps_result.arrays["c_atomicenergies"].reshape(-1),
                atol=tol,
                rtol=tol,
            ), f"Max atomic energy error: {np.abs(structure.get_potential_energies() - lammps_result.arrays['c_atomicenergies'].reshape(-1)).max()}"

            # check system quantities
            lammps_pe = float(Path(tmpdir + "/pe.dat").read_text()) / PRECISION_CONST
            lammps_totalatomicenergy = (
                float(Path(tmpdir + "/totalatomicenergy.dat").read_text())
                / PRECISION_CONST
            )
            assert np.allclose(lammps_pe, lammps_totalatomicenergy)
            assert np.allclose(
                structure.get_potential_energy(),
                lammps_pe,
                atol=tol,
                rtol=tol,
            )
            # in `metal` units, pressure/stress has units bars
            # so need to convert
            lammps_stress = np.fromstring(
                Path(tmpdir + "/stress.dat").read_text(), sep=" ", dtype=np.float64
            ) * (ase.units.bar / PRECISION_CONST)
            # https://docs.lammps.org/compute_pressure.html
            # > The ordering of values in the symmetric pressure tensor is as follows: pxx, pyy, pzz, pxy, pxz, pyz.
            lammps_stress = np.array(
                [
                    [lammps_stress[0], lammps_stress[3], lammps_stress[4]],
                    [lammps_stress[3], lammps_stress[1], lammps_stress[5]],
                    [lammps_stress[4], lammps_stress[5], lammps_stress[2]],
                ]
            )
            if periodic:
                # In LAMMPS, the convention is that the stress tensor, and thus the pressure, is related to the virial
                # WITHOUT a sign change.  In `nequip`, we chose currently to follow the virial = -stress x volume
                # convention => stress = -1/V * virial.  ASE does not change the sign of the virial, so we have
                # to flip the sign from ASE for the comparison.
                ase_stress = -structure.get_stress(voigt=False)
                stress_err = np.max(np.abs(ase_stress - lammps_stress))
                stol = tol * 20
                assert np.allclose(
                    ase_stress,
                    lammps_stress,
                    atol=stol,
                    rtol=stol,
                ), f"Stress max abs err: {stress_err:.8g} (tol={stol:.3g})\nASE stress: {ase_stress.flatten().tolist()}\nLAMMPS stress: {lammps_stress.flatten().tolist()}"
