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


#ifdef PAIR_CLASS

PairStyle(allegro,PairAllegro<lowlow>)
PairStyle(allegro3232,PairAllegro<lowlow>)
PairStyle(allegro6464,PairAllegro<highhigh>)
PairStyle(allegro3264,PairAllegro<lowhigh>)
PairStyle(allegro6432,PairAllegro<highlow>)

#else

#ifndef LMP_PAIR_ALLEGRO_H
#define LMP_PAIR_ALLEGRO_H

#include "pair.h"

#include <torch/torch.h>
#include <vector>
#include <type_traits>
enum Precision {lowlow, highhigh, lowhigh, highlow};

namespace LAMMPS_NS {

template<Precision precision>
class PairAllegro : public Pair {
 public:
  PairAllegro(class LAMMPS *);
  virtual ~PairAllegro();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  double cutoff;
  torch::jit::Module model;
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;

  int batch_size = -1;

  typedef typename std::conditional_t<precision==lowlow || precision==lowhigh, float, double> inputtype;
  typedef typename std::conditional_t<precision==lowlow || precision==highlow, float, double> outputtype;

  torch::ScalarType inputtorchtype = torch::CppTypeToScalarType<inputtype>();
  torch::ScalarType outputtorchtype = torch::CppTypeToScalarType<outputtype>();

 protected:
  int debug_mode = 0;

};

}

#endif
#endif

