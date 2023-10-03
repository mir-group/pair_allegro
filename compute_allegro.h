/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(allegro,ComputeAllegro)

#else

#ifndef LMP_COMPUTE_ALLEGRO_H
#define LMP_COMPUTE_ALLEGRO_H

#include "compute.h"
#include "pair_allegro.h"

#include <torch/torch.h>
#include <string>

namespace LAMMPS_NS {

class ComputeAllegro : public Compute {
 public:
  ComputeAllegro(class LAMMPS *, int, char**);
  ~ComputeAllegro();
  void compute_vector() override;
  void init() override;

 protected:
  std::string quantity;

};

}

#endif
#endif

