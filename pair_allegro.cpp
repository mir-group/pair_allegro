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

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "output.h"
#include "potential_file_reader.h"
#include "tokenizer.h"
#include "update.h"
#include <pair_allegro.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

// TODO: Only if MPI is available
#include <mpi.h>

// Freezing is broken from C++ in <=1.10; so we've dropped support.
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#error "PyTorch version < 1.11 is not supported"
#endif

using namespace LAMMPS_NS;

template <Precision precision> PairAllegro<precision>::PairAllegro(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  if (comm->me == 0)
    std::cout << "Allegro is using input precision " << typeid(inputtype).name()
              << " and output precision " << typeid(outputtype).name() << std::endl;
  ;

  if (const char *env_p = std::getenv("ALLEGRO_DEBUG")) {
    std::cout << "PairAllegro is in DEBUG mode, since ALLEGRO_DEBUG is in env\n";
    debug_mode = 1;
  }

  if (torch::cuda::is_available()) {
    int deviceidx = -1;
    if (comm->nprocs > 1) {
      MPI_Comm shmcomm;
      MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
      int shmrank;
      MPI_Comm_rank(shmcomm, &shmrank);
      deviceidx = shmrank;
    }
    if (deviceidx >= 0) {
      int devicecount = torch::cuda::device_count();
      if (deviceidx >= devicecount) {
        if (debug_mode) {
          // To allow testing multi-rank calls, we need to support multiple ranks with one GPU
          std::cerr << "WARNING (Allegro): my rank (" << deviceidx
                    << ") is bigger than the number of visible devices (" << devicecount
                    << "), wrapping around to use device " << deviceidx % devicecount
                    << " again!!!";
          deviceidx = deviceidx % devicecount;
        } else {
          // Otherwise, more ranks than GPUs is an error
          std::cerr << "ERROR (Allegro): my rank (" << deviceidx
                    << ") is bigger than the number of visible devices (" << devicecount << ")!!!";
          error->all(FLERR,
                     "pair_allegro: mismatch between number of ranks and number of available GPUs");
        }
      }
    }
    device = c10::Device(torch::kCUDA, deviceidx);
  } else {
    device = torch::kCPU;
  }
  if (debug_mode) std::cout << "Allegro is using device " << device << "\n";
}

template <Precision precision> PairAllegro<precision>::~PairAllegro()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutoff_matrix);
  }
}

template <Precision precision> void PairAllegro<precision>::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style Allegro requires atom IDs");

  // Request a full neighbor list.
  if (lmp->kokkos) {
    // Only request full to avoid a Kokkos bug; pair_allegro_kokkos.cpp doesn't need GHOST anyway
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    // Non-kokkos needs ghost to avoid segfaults
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  }

  if (force->newton_pair == 0) error->all(FLERR, "Pair style Allegro requires newton pair on");
}

template <Precision precision> double PairAllegro<precision>::init_one(int i, int j)
{
  return cutoff;
}

template <Precision precision> void PairAllegro<precision>::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cutoff_matrix, n, n, "pair:cutoff_matrix");
}

template <Precision precision> void PairAllegro<precision>::settings(int narg, char ** /*arg*/)
{
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0) error->all(FLERR, "Illegal pair_style command, too many arguments");
}

template <Precision precision> void PairAllegro<precision>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3 + ntypes))
    error->all(FLERR,
               "Incorrect args for pair coefficients, should be * * <model>.pth <type1> <type2> "
               "... <typen>");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 0;

  std::vector<std::string> elements(ntypes);
  for (int i = 0; i < ntypes; i++) { elements[i] = arg[i + 1]; }

  if (comm->me == 0) std::cout << "Allegro: Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {{"config", ""},
                                                           {"nequip_version", ""},
                                                           {"r_max", ""},
                                                           {"n_species", ""},
                                                           {"type_names", ""},
                                                           {"_jit_bailout_depth", ""},
                                                           {"_jit_fusion_strategy", ""},
                                                           {"allow_tf32", ""},
                                                           {"per_edge_type_cutoff", ""}};
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  // Check if model is a NequIP model
  if (metadata["nequip_version"].empty()) {
    error->all(FLERR,
               "The indicated TorchScript file does not appear to be a deployed NequIP model; did "
               "you forget to run `nequip-deploy`?");
  }

  // If the model is not already frozen, we should freeze it:
  // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
  if (model.hasattr("training")) {
    if (comm->me == 0) std::cout << "Allegro: Freezing TorchScript model...\n";
    model = torch::jit::freeze(model);
  }

  // In PyTorch >=1.11, this is now set_fusion_strategy
  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  //strategy = {{torch::jit::FusionBehavior::STATIC, 100}, {torch::jit::FusionBehavior::DYNAMIC, 1}};

  //if (metadata["_jit_fusion_strategy"].empty()) { //TODO: respect model
  //  // This is the default used in the Python code
  //  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
  //} else {
  //  std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
  //  std::string fusion_type, fusion_depth;
  //  while(std::getline(strat_stream, fusion_type, ',')) {
  //    std::getline(strat_stream, fusion_depth, ';');
  //    strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
  //  }
  //}
  torch::jit::setFusionStrategy(strategy);

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  if (debug_mode) {
    std::cout << "Allegro: Information from model: " << metadata.size() << " key-value pairs\n";
    for (const auto &n : metadata) {
      if (n.first == "type_names")
        std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    }
  }

  cutoff = std::stod(metadata["r_max"]);

  //TODO: This
  type_mapper.resize(ntypes, -1);
  std::stringstream ss;
  int n_species = std::stod(metadata["n_species"]);
  ss << metadata["type_names"];
  if (comm->me == 0)
    std::cout << "Type mapping:"
              << "\n";
  if (comm->me == 0)
    std::cout << "Allegro type | Allegro name | LAMMPS type | LAMMPS name"
              << "\n";
  for (int i = 0; i < n_species; i++) {
    std::string ele;
    ss >> ele;
    for (int itype = 1; itype <= ntypes; itype++) {
      if (ele.compare(arg[itype + 3 - 1]) == 0) {
        type_mapper[itype - 1] = i;
        if (comm->me == 0)
          std::cout << i << " | " << ele << " | " << itype << " | " << arg[itype + 3 - 1] << "\n";
      }
    }
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++) {
    for (int j = i; j <= ntypes; j++) {
      if ((type_mapper[i - 1] >= 0) && (type_mapper[j - 1] >= 0)) { setflag[i][j] = 1; }
    }
  }

  if (!metadata["per_edge_type_cutoff"].empty()) {
    std::stringstream matrix_string;
    matrix_string << metadata["per_edge_type_cutoff"];
    std::vector<int> reverse_type_mapper(n_species, -1);

    for (int i = 0; i < ntypes; i++) { reverse_type_mapper[type_mapper[i]] = i; }

    for (int i = 0; i < n_species; i++) {
      for (int j = 0; j < n_species; j++) {
        double cutij;
        matrix_string >> cutij;
        if (reverse_type_mapper[i] >= 0 && reverse_type_mapper[j] >= 0) {
          if (comm->me == 0) {
            printf("%s %s si=%d sj=%d ti=%d tj=%d cut=%.2f\n", arg[reverse_type_mapper[i] + 3],
                   arg[reverse_type_mapper[j] + 3], i, j, reverse_type_mapper[i],
                   reverse_type_mapper[j], cutij);
          }
          cutoff_matrix[reverse_type_mapper[i]][reverse_type_mapper[j]] = cutij;    //TODO
        }
      }
    }
  } else {
    for (int i = 0; i < ntypes; i++) {
      for (int j = 0; j < ntypes; j++) { cutoff_matrix[i][j] = cutoff; }
    }
  }
}

// Force and energy computation
template <Precision precision> void PairAllegro<precision>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  // Should be on.
  int newton_pair = force->newton_pair;

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum == nlocal);    // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Skip calculation if empty domain
  if (inum==0) return;

  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;

  // Number of bonds per atom
  std::vector<int> neigh_per_atom(nlocal, 0);
  int ntypes = atom->ntypes;

#pragma omp parallel for reduction(+ : nedges)
  for (int ii = 0; ii < nlocal; ii++) {
    int i = ilist[ii];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;

      double cutij =
          cutoff_matrix[type[i] - 1]
                       [type[j] - 1];    // cutoff_matrix[(type[i]-1)*ntypes + type[j]-1];
      //printf("i=%5d j=%5d ti=%d tj=%d cut=%.2f\n", i, j, type[i], type[j], cutij);
      if (rsq <= cutij * cutij) {
        neigh_per_atom[ii]++;
        nedges++;
      }
    }
  }

  // Cumulative sum of neighbors, for knowing where to fill in the edges tensor
  std::vector<int> cumsum_neigh_per_atom(nlocal);

  for (int ii = 1; ii < nlocal; ii++) {
    cumsum_neigh_per_atom[ii] = cumsum_neigh_per_atom[ii - 1] + neigh_per_atom[ii - 1];
  }

  torch::Tensor pos_tensor =
      torch::zeros({ntotal, 3}, torch::TensorOptions().dtype(inputtorchtype));
  torch::Tensor edges_tensor =
      torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor ij2type_tensor =
      torch::zeros({ntotal}, torch::TensorOptions().dtype(torch::kInt64));

  auto pos = pos_tensor.accessor<inputtype, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode) printf("Allegro edges: i j rij\n");
#pragma omp parallel for
  for (int ii = 0; ii < ntotal; ii++) {
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    ij2type[i] = type_mapper[itype - 1];

    pos[i][0] = x[i][0];
    pos[i][1] = x[i][1];
    pos[i][2] = x[i][2];

    if (ii >= nlocal) { continue; }

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];

    int edge_counter = cumsum_neigh_per_atom[ii];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx * dx + dy * dy + dz * dz;

      double cutij =
          cutoff_matrix[itype - 1][jtype - 1];    //cutoff_matrix[(type[i]-1)*ntypes + type[j]-1];
      if (rsq > cutij * cutij) { continue; }

      // TODO: double check order
      edges[0][edge_counter] = i;
      edges[1][edge_counter] = j;

      edge_counter++;

      if (debug_mode) printf("%d %d %.10g\n", itag - 1, jtag - 1, sqrt(rsq));
    }
  }
  if (debug_mode) printf("end Allegro edges\n");

  torch::Tensor compute_custom_tensor =
      torch::full({1}, false, torch::TensorOptions().dtype(torch::kBool));
  if (update->ntimestep == output->next && custom_output_names.size() > 0) {
    // error->message(FLERR, "computing custom output");
    auto tmp = compute_custom_tensor.accessor<bool, 1>();
    tmp[0] = true;
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("atom_types", ij2type_tensor.to(device));
  input.insert("compute_custom_output", compute_custom_tensor);
  std::vector<torch::IValue> input_vector(1, input);

  auto output = model.forward(input_vector).toGenericDict();

  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<outputtype, 2>();

  //torch::Tensor total_energy_tensor = output.at("total_energy").toTensor().cpu(); WRONG WITH MPI

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor().cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<outputtype, 2>();
  outputtype atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<outputtype>()[0];

  //std::cout << "atomic energy sum: " << atomic_energy_sum << std::endl;
  //std::cout << "Total energy: " << total_energy_tensor << "\n";
  //std::cout << "atomic energy shape: " << atomic_energy_tensor.sizes()[0] << "," << atomic_energy_tensor.sizes()[1] << std::endl;
  //std::cout << "atomic energies: " << atomic_energy_tensor << std::endl;

  // Write forces and per-atom energies (0-based tags here)
  eng_vdwl = 0.0;
#pragma omp parallel for reduction(+ : eng_vdwl)
  for (int ii = 0; ii < ntotal; ii++) {
    int i = ilist[ii];

    f[i][0] += forces[i][0];
    f[i][1] += forces[i][1];
    f[i][2] += forces[i][2];
    if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i][0];
    if (ii < inum) eng_vdwl += atomic_energies[i][0];
  }

  if (vflag) {
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<outputtype, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    virial[0] = v[0][0][0];
    virial[1] = v[0][1][1];
    virial[2] = v[0][2][2];
    virial[3] = v[0][0][1];
    virial[4] = v[0][0][2];
    virial[5] = v[0][1][2];
  }
  if (vflag_atom) { error->all(FLERR, "Pair style Allegro does not support per-atom virial"); }

  // TODO: Figure out reliable solution
  // if (update->ntimestep == this->output->next || update->ntimestep==0) {
    if (debug_mode) {
      std::cout << "ALLEGRO CUSTOM OUTPUT" << std::endl;
      for (const auto &elem : output) {
        std::cout << elem.key() << "\n" << elem.value() << std::endl;
      }
    }

    for (const std::string &output_name : custom_output_names) {
      if (!output.contains(output_name)) error->all(FLERR, "missing {}", output_name);
      // printf("pair_allegro inserting %s\n", output_name.data()); fflush(stdout);
      custom_output.insert_or_assign(output_name, output.at(output_name).toTensor().detach());
    }
  // }
}

template <Precision precision> void PairAllegro<precision>::add_custom_output(std::string name)
{
  custom_output_names.push_back(name);
}

namespace LAMMPS_NS {
template class PairAllegro<lowlow>;
template class PairAllegro<highhigh>;
template class PairAllegro<lowhigh>;
template class PairAllegro<highlow>;
}    // namespace LAMMPS_NS
