/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include "compute_allegro.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair_nequip_allegro.h"
#include "update.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>

using namespace LAMMPS_NS;

template<int peratom>
ComputeAllegro<peratom>::ComputeAllegro(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg)
{

  if constexpr (!peratom) {
    // compute 1 all allegro quantity length
    if (narg != 5) error->all(FLERR, "Incorrect args for compute allegro");
  } else {
    // compute 1 all allegro/atom quantity length newton(1/0)
    if (narg != 6) error->all(FLERR, "Incorrect args for compute allegro/atom");
  }

  if (strcmp(arg[1], "all") != 0)
    error->all(FLERR, "compute allegro can only operate on group 'all'");

  quantity = arg[3];
  if constexpr (peratom) {
    peratom_flag = 1;
    nperatom = std::atoi(arg[4]);
    newton = std::atoi(arg[5]);
    if (newton) comm_reverse = nperatom;
    size_peratom_cols = nperatom==1 ? 0 : nperatom;
    nmax = -12;
    if (comm->me == 0)
      error->message(FLERR, "compute allegro/atom will evaluate the quantity {} of length {} with newton {}", quantity,
                     size_peratom_cols, newton);
  } else {
    vector_flag = 1;
    //As stated in the README, we assume vector properties are extensive
    extvector = 1;
    size_vector = std::atoi(arg[4]);
    if (size_vector <= 0) error->all(FLERR, "Incorrect vector length!");
    memory->create(vector, size_vector, "ComputeAllegro:vector");
    if (comm->me == 0)
      error->message(FLERR, "compute allegro will evaluate the quantity {} of length {}", quantity,
                     size_vector);
  }

  if (force->pair == nullptr) {
    error->all(FLERR, "no pair style; compute allegro must be defined after pair style");
  }

  ((PairNequIPAllegro<0> *) force->pair)->add_custom_output(quantity);
}

template<int peratom>
void ComputeAllegro<peratom>::init()
{
  ;
}

template<int peratom>
ComputeAllegro<peratom>::~ComputeAllegro()
{
  if (copymode) return;
  if constexpr (peratom) {
    memory->destroy(vector_atom);
  } else {
    memory->destroy(vector);
  }
}

template<int peratom>
void ComputeAllegro<peratom>::compute_vector()
{
  invoked_vector = update->ntimestep;

  // empty domain, pair style won't store tensor
  // note: assumes nlocal == inum
  if (atom->nlocal == 0) {
    for (int i = 0; i < size_vector; i++) {
      vector[i] = 0.0;
    }
  } else {
    const torch::Tensor &quantity_tensor =
        ((PairNequIPAllegro<0> *) force->pair)->custom_output.at(quantity).cpu().ravel();

    auto quantity = quantity_tensor.data_ptr<double>();

    if (quantity_tensor.size(0) != size_vector) {
      error->one(FLERR, "size {} of quantity tensor {} does not match expected {} on rank {}",
                 quantity_tensor.size(0), this->quantity, size_vector, comm->me);
    }

    for (int i = 0; i < size_vector; i++) { vector[i] = quantity[i]; }
  }

  // even if empty domain
  MPI_Allreduce(MPI_IN_PLACE, vector, size_vector, MPI_DOUBLE, MPI_SUM, world);
}

template<int peratom>
void ComputeAllegro<peratom>::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->destroy(array_atom);
    memory->create(array_atom, nmax, nperatom, "allegro/atom:array");
    if (nperatom==1) vector_atom = &array_atom[0][0];
  }

  // guard against empty domain (pair style won't store tensor)
  if (atom->nlocal > 0) {
    const torch::Tensor &quantity_tensor =
        ((PairNequIPAllegro<0> *) force->pair)->custom_output.at(quantity).cpu().contiguous().reshape({-1,nperatom});

    auto quantity = quantity_tensor.accessor<double,2>();
    quantityptr = quantity_tensor.data_ptr<double>();

    int nlocal = atom->nlocal;
    for (int i = 0; i < nlocal; i++) {
      for (int j = 0; j < nperatom; j++) {
        array_atom[i][j] = quantity[i][j];
      }
    }
  }

  // even if empty domain
  if (newton) comm->reverse_comm(this);
}

template<int peratom>
int ComputeAllegro<peratom>::pack_reverse_comm(int n, int first, double *buf)
{
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int j = 0; j < nperatom; j++) {
      buf[m++] = quantityptr[i*nperatom + j];
    }
  }
  return m;
}

template<int peratom>
void ComputeAllegro<peratom>::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int k = 0; k < nperatom; k++) {
      array_atom[j][k] += buf[m++];
    }
  }
}


namespace LAMMPS_NS {
  template class ComputeAllegro<0>;
  template class ComputeAllegro<1>;
}
