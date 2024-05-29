/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(addbornforce,FixAddBornForce);
// clang-format on
#else

#ifndef LMP_FIX_ADDBORNFORCE_H
#define LMP_FIX_ADDBORNFORCE_H

#include "fix.h"
#include <torch/torch.h>

namespace LAMMPS_NS {

class FixAddBornForce : public Fix {
public:
  FixAddBornForce(class LAMMPS *, int, char **);
  ~FixAddBornForce() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void min_post_force(int) override;
  double compute_vector(int) override;
  double memory_usage() override;

  // this is used as the energy contribution
  double compute_scalar() override;

  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

private:
  double xvalue, yvalue, zvalue;
  int varflag;
  char *xstr, *ystr, *zstr;
  char *idregion;
  class Region *region;
  int xvar, yvar, zvar, xstyle, ystyle, zstyle;
  int reduced_flag;
  double extrapolarization[3], extraenergy;

  int maxatom;
  double **efieldatom;
  torch::Tensor born_tensor;
};

} // namespace LAMMPS_NS

#endif
#endif
