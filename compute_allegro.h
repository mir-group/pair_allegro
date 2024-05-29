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

ComputeStyle(allegro,ComputeAllegro<0>)
ComputeStyle(allegro/atom,ComputeAllegro<1>)

#else

#ifndef LMP_COMPUTE_ALLEGRO_H
#define LMP_COMPUTE_ALLEGRO_H

#include "compute.h"
#include "pair_allegro.h"

#include <string>

namespace LAMMPS_NS {

template<int peratom>
class ComputeAllegro : public Compute {
 public:
  ComputeAllegro(class LAMMPS *, int, char**);
  ~ComputeAllegro();
  void compute_vector() override;
  void compute_peratom() override;
  void init() override;

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  std::string quantity;
  double *quantityptr;
  int newton;
  int nperatom;
  int nmax;

};

}

#endif
#endif

