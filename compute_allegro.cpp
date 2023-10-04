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
#include "pair_allegro.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>


using namespace LAMMPS_NS;

ComputeAllegro::ComputeAllegro(LAMMPS *lmp, int narg, char **arg) : Compute(lmp, narg, arg) {

  // compute 1 all allegro quantity length
  if (narg != 5)
    error->all(FLERR, "Incorrect args for compute nequip");

  if (strcmp(arg[1], "all") != 0)
    error->all(FLERR, "compute nequip can only operate on group 'all'");

  quantity = arg[3];
  vector_flag = 1;
  size_vector = std::atoi(arg[4]);

  if (comm->me==0) error->message(FLERR, "compute allegro will evaluate the quantity {} of length {}", quantity, size_vector);

  if(size_vector<=0)
    error->all(FLERR, "Incorrect vector length!");
  memory->create(vector, size_vector, "ComputeAllegro:vector");
}

void ComputeAllegro::init(){
  ;
}

ComputeAllegro::~ComputeAllegro(){
  if (!copymode) {
    memory->destroy(vector);
  }
}

// Force and energy computation
void ComputeAllegro::compute_vector(){
  invoked_peratom = update->ntimestep;

  const torch::Tensor &quantity_tensor = ((PairAllegro<lowhigh>*) force->pair)->custom_output.at(quantity).cpu();
  auto quantity = quantity_tensor.data_ptr<double>();

  for(int i = 0; i < size_vector; i++){
    vector[i] = quantity[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, vector, size_vector, MPI_DOUBLE, MPI_SUM, world);
}

