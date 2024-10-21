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

PairStyle(nequip,PairNequIPAllegro<1>)
PairStyle(allegro,PairNequIPAllegro<0>)

#else

#ifndef LMP_PAIR_NEQUIP_ALLEGRO_H
#define LMP_PAIR_NEQUIP_ALLEGRO_H

#include "pair.h"

#include <torch/torch.h>
#include <vector>
#include <type_traits>
#include <map>
#include <string>


namespace LAMMPS_NS {

template<int nequip_mode>
class PairNequIPAllegro : public Pair {
 public:
  PairNequIPAllegro(class LAMMPS *);
  virtual ~PairNequIPAllegro();
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

  typedef float inputtype;
  typedef double outputtype;

  torch::ScalarType inputtorchtype = torch::CppTypeToScalarType<inputtype>();
  torch::ScalarType outputtorchtype = torch::CppTypeToScalarType<outputtype>();

  std::vector<std::string> custom_output_names;
  std::map<std::string, torch::Tensor> custom_output;
  void add_custom_output(std::string);

 protected:
  int debug_mode = 0;

  double** cutoff_matrix;

  c10::Dict<std::string, torch::Tensor> preprocess();

  torch::Tensor get_cell();
  void get_tag2i(std::vector<int>&);
};

}

#endif
#endif

