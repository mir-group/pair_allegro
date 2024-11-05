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

PairStyle(nequip,PairNequIPAllegro<true>)
PairStyle(allegro,PairNequIPAllegro<false>)

#else

#ifndef LMP_PAIR_NEQUIP_ALLEGRO_H
#define LMP_PAIR_NEQUIP_ALLEGRO_H

#include "pair.h"

#include <torch/torch.h>
#ifdef NEQUIP_AOT_COMPILE
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#endif

#include <vector>
#include <type_traits>
#include <map>
#include <string>


namespace LAMMPS_NS {

template<bool nequip_mode>
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
  torch::Device device = torch::kCPU;
  std::vector<int> type_mapper;

  std::string model_path;

  // `use_aot` bool present even if NEQUIP_AOT_COMPILE not used at compile-time
  // cleaner logic where we always have an outer condition `use_aot` and an inner `ifndef` condition on whether NEQUIP_AOT_COMPILE was used at compile-time
  // errors out if `use_aot` is true, but NEQUIP_AOT_COMPILE wasn't used at compile-time
  bool use_aot;

  // keep separate `torchscript_model` and `aot_model` declarations since which is used is only determined at runtime
  torch::jit::Module torchscript_model;

#ifdef NEQUIP_AOT_COMPILE
  // both torchscript or AOT model can be used
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aot_model;
  std::vector<std::string> model_input_order;
  std::vector<std::string> model_output_order;
#endif

  // In nequip >= 0.7.0, input and output are always F64
  typedef double inputtype;
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
  c10::Dict<std::string, torch::Tensor> call(c10::Dict<std::string, torch::Tensor>);

  torch::Tensor get_cell();
  void get_tag2i(std::vector<int>&);
};

}

#endif
#endif

