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
#include <pair_nequip_allegro.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <filesystem>
#include <sstream>
#include <string>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include <mpi.h>

// Freezing is broken from C++ in <=1.10; so we've dropped support.
#if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
#error "PyTorch version < 1.11 is not supported"
#endif

#ifdef NEQUIP_AOT_COMPILE
// torch 2.6 required for AOT Inductor
#if (TORCH_VERSION_MAJOR < 2 && TORCH_VERSION_MINOR <= 6)
#error "NEQUIP_AOT_COMPILE requires PyTorch >= 2.6"
#endif
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#endif

using namespace LAMMPS_NS;

template <bool nequip_mode> PairNequIPAllegro<nequip_mode>::PairNequIPAllegro(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;

  if (comm->me == 0)
    std::cout << "NequIP/Allegro is using input precision " << typeid(inputtype).name()
              << " and output precision " << typeid(outputtype).name() << std::endl;
  ;

  // === set variables based on environment variables ===
  // debug mode
  if (const char *env_p = std::getenv("_NEQUIP_LOG_LEVEL")) {
    if (std::string(env_p) == "DEBUG") {
        std::cout << "Debug mode enabled, since _NEQUIP_LOG_LEVEL is set to DEBUG\n";
        debug_mode = 1;
    }
  }

  // error out if more than one rank but in NequIP mode
  if (nequip_mode && comm->nprocs > 1) {
    error->all(FLERR,
               "pair_nequip only works with a single MPI rank but more than one detected");
  }
  
  // === set device ===
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
  if (debug_mode) std::cout << "NequIP/Allegro is using device " << device << "\n";
}

template <bool nequip_mode> PairNequIPAllegro<nequip_mode>::~PairNequIPAllegro()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutoff_matrix);
  }
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style Allegro requires atom IDs");

  // Request a full neighbor list.
  if (lmp->kokkos) {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    // Non-kokkos needs ghost to avoid segfaults
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  }

  if (!nequip_mode && force->newton_pair == 0) error->all(FLERR, "Pair style allegro requires newton pair on");
  if (nequip_mode && force->newton_pair) error->all(FLERR, "Pair style nequip requires newton pair off");
}

template <bool nequip_mode> double PairNequIPAllegro<nequip_mode>::init_one(int i, int j)
{
  return cutoff;
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cutoff_matrix, n, n, "pair:cutoff_matrix");
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::settings(int narg, char ** /*arg*/)
{
  // "allegro" should be the only word after "pair_style" in the input file.
  if (narg > 0) error->all(FLERR, "Illegal pair_style command, too many arguments");
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  int ntypes = atom->ntypes;

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 0;

  // === parse arg ===
  // should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3 + ntypes)) {
    error->all(FLERR,
	        "Incorrect args for pair coefficients, should be * * <model>.nequip.pth/pt2 <type1> <type2> ... <typen>");
  }

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // set model path
  model_path = std::string(arg[2]);
  // condition `use_aot` on extension `.nequip.pth` vs `.nequip.pt2`
  std::string ts_ext = ".nequip.pth";
  std::string aot_ext = ".nequip.pt2";
  if (model_path.size() >= ts_ext.size() && model_path.compare(model_path.size() - ts_ext.size(), ts_ext.size(), ts_ext) == 0) {
    use_aot = false;
  } else if (model_path.size() >= aot_ext.size() && model_path.compare(model_path.size() - aot_ext.size(), aot_ext.size(), aot_ext) == 0) {
	use_aot = true;
  }
  else {
    throw std::runtime_error("Only accepts model paths with extension `.nequip.pth` or `.nequip.pt2`, but found" + model_path);
  }

  // set up metadata dict
  std::unordered_map<std::string, std::string> metadata;

  // load model and metadata depending on torchscript vs aot
  if (comm->me == 0) std::cout << "NequIP/Allegro: Loading model from " << model_path << "\n";
  if (!use_aot) {
    metadata = {
	  {"r_max", ""},
	  {"per_edge_type_cutoff", ""},
	  {"type_names", ""},
	  {"num_types", ""},
	  {"allow_tf32", ""}
    };
    // === TorchScript ===
    torchscript_model = torch::jit::load(model_path, device, metadata);
    // TODO:  Python will do this, but we probably should still do it here?
	torchscript_model.eval();
    // TODO : think about whether to move freezing into Python in nequip-compile
    // If the model is not already frozen, we should freeze it:
    // This is the check used by PyTorch: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/api/module.cpp#L476
    if (torchscript_model.hasattr("training")) {
      // TODO (general):  use LAMMPS logging tools more consistently?
      if (comm->me == 0) std::cout << "NequIP/Allegro: Freezing TorchScript model...\n";
      torchscript_model = torch::jit::freeze(torchscript_model);
    }
  } else {
#ifndef NEQUIP_AOT_COMPILE
    throw std::runtime_error("AOT Inductor compiled model (`.nequip.pt2` extension) found but pair style not compiled with `NEQUIP_AOT_COMPILE`");
#else
    // === AOT ===
    aot_model = std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path);
    metadata = aot_model->get_metadata();

    // set up input and output order
    model_input_order = {"pos", "edge_index", "atom_types"};
    if (nequip_mode) {
      std::vector<std::string> additional_items = {"cell", "edge_cell_shift"};
      model_input_order.insert(model_input_order.end(), additional_items.begin(), additional_items.end());
    }
    model_output_order = {"atomic_energy", "forces", "virial"};
#endif
  }

  // === process metadata information ===
  if (debug_mode) {
    std::cout << "NequIP/Allegro: Information from model: " << metadata.size() << " key-value pairs\n";
    for (const auto &n : metadata) {
      std::cout << "Key:[" << n.first << "] Value:[" << n.second << "]\n";
    }
  }

  if (!use_aot) {
    // TorchScript -- hardcode DYNAMIC, 10 (same is true on Python side)
    torch::jit::FusionStrategy strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
    torch::jit::setFusionStrategy(strategy);
  }

  // TODO: should TF32 be set before loading the model? does it matter?
  // set whether to allow TF32 -- it gets saved as "0" or "1"
  bool allow_tf32 = std::stoi(metadata["allow_tf32"]);
  // see https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  cutoff = std::stod(metadata["r_max"]);

  type_mapper.resize(ntypes, -1);
  std::stringstream ss;
  int num_model_types = std::stod(metadata["num_types"]);
  ss << metadata["type_names"];
  if (comm->me == 0)
    std::cout << "Type mapping:"
              << "\n";
  if (comm->me == 0)
    std::cout << "NequIP/Allegro type | NequIP/Allegro name | LAMMPS type | LAMMPS name"
              << "\n";
  for (int i = 0; i < num_model_types; i++) {
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
    std::vector<int> reverse_type_mapper(num_model_types, -1);

    for (int i = 0; i < ntypes; i++) { reverse_type_mapper[type_mapper[i]] = i; }

    for (int i = 0; i < num_model_types; i++) {
      for (int j = 0; j < num_model_types; j++) {
        double cutij;
        matrix_string >> cutij;
        if (reverse_type_mapper[i] >= 0 && reverse_type_mapper[j] >= 0) {
          if (comm->me == 0) {
            printf("%s %s si=%d sj=%d ti=%d tj=%d cut=%.2f\n", arg[reverse_type_mapper[i] + 3],
                   arg[reverse_type_mapper[j] + 3], i, j, reverse_type_mapper[i],
                   reverse_type_mapper[j], cutij);
          }
          cutoff_matrix[reverse_type_mapper[i]][reverse_type_mapper[j]] = cutij;
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
template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Atom forces
  double **f = atom->f;

  int inum = list->inum;
  if (inum==0) return;

  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;

  // Mapping from neigh list ordering to x/f ordering
  // (in case we want to support pair_coeff other than * * in the future)
  int *ilist = list->ilist;

  // create input to model (positions etc)
  auto input = preprocess();
  // evaluate model
  auto output = call(input);

  // extract forces and energies as CPU tensors
  torch::Tensor forces_tensor = output.at("forces").cpu();
  auto forces = forces_tensor.accessor<outputtype, 2>();

  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<outputtype, 2>();
  outputtype atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<outputtype>()[0];

  // store forces and energy
  // energy = sum of local atomic energies to avoid energy shift issues for ghost atoms
  // NequIP only produces forces on local atoms,
  // Allegro also on ghost atoms, these are reverse communicated w/Newton
  eng_vdwl = 0.0;
  int nforces = nequip_mode ? inum : ntotal;
#pragma omp parallel for reduction(+ : eng_vdwl)
  for (int ii = 0; ii < nforces; ii++) {
    int i = ilist[ii];

    f[i][0] += forces[i][0];
    f[i][1] += forces[i][1];
    f[i][2] += forces[i][2];
    if (eflag_atom && ii < inum) eatom[i] = atomic_energies[i][0];
    if (ii < inum) eng_vdwl += atomic_energies[i][0];
  }

  if (vflag) {
    torch::Tensor v_tensor = output.at("virial").cpu();
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
  if (vflag_atom) { error->all(FLERR, "Pair styles nequip and allegro do not support per-atom virial"); }

  if (debug_mode && 2<1) {
    std::cout << "ALLEGRO CUSTOM OUTPUT" << std::endl;
    for (const auto &elem : output) {
      std::cout << elem.key() << "\n" << elem.value() << std::endl;
    }
  }

  for (const std::string &output_name : custom_output_names) {
    if (!output.contains(output_name)) error->all(FLERR, "missing {}", output_name);
    custom_output.insert_or_assign(output_name, output.at(output_name).detach());
  }
}

template <bool nequip_mode> c10::Dict<std::string, torch::Tensor> PairNequIPAllegro<nequip_mode>::call(c10::Dict<std::string, torch::Tensor> input) {
  // This function takes an "AtomicDataDict", calls the model, and returns an "AtomicDataDict"
  // Note that this function does NOT deal with devices, it assumes `input` is already on the right device and returns whatever device the model returns.
  // Moving to device is the responsibility of the calling code, since what device the original tensors are on varies between Kokkos and non-Kokkos anyway.

  // call the model depending on compilation mode
  c10::Dict<std::string, torch::Tensor> output;

  if (!use_aot) {
    // === TorchScript ===
    std::vector<torch::IValue> input_vector(1, input);

    // can't just do
    // ```output = model.forward(input_vector).toGenericDict();```
    // because of type mismatch, i.e.
    // ```: c10::Dict<std::string, at::Tensor> = c10::Dict<c10::IValue, c10::IValue>```
    auto generic_output = torchscript_model.forward(input_vector).toGenericDict();
    for (const auto& item : generic_output) {
      std::string key = item.key().toStringRef();
      torch::Tensor value = item.value().toTensor();
      output.insert(key, value);
    }
  } else {
#ifndef NEQUIP_AOT_COMPILE	  
	// should have errored out at load time in `coeff` but have error code here just in case
    throw std::runtime_error("AOT Inductor compiled model (`.nequip.pt2` extension) found but pair style not compiled with `NEQUIP_AOT_COMPILE`");
#else
    // === AOT ===
    // `input` dict -> `input_vector` according to `model_input_order`
    std::vector<torch::Tensor> input_vector;
    for (const std::string &input_field : model_input_order) {
      input_vector.push_back(input.at(input_field));
    }
    // run the model
    // TODO: do we need to run with cuda stream and get the cuda stream from Kokkos??
	// see https://github.com/pytorch/pytorch/blob/39ede99a33b0631330a3966e567d3c07d93aca17/torch/csrc/inductor/aoti_runner/model_container_runner.h#L27
	std::vector<torch::Tensor> output_vector = aot_model->run(input_vector);
    // `output_vector` -> `output` dict according to `model_output_order`
    std::vector<torch::Tensor>::iterator tensor_it = output_vector.begin();
    for (const std::string& key : model_output_order) {
      output.insert(key, *tensor_it++);
    }
#endif
  }
  return output;
}


template <bool nequip_mode> c10::Dict<std::string, torch::Tensor> PairNequIPAllegro<nequip_mode>::preprocess() {
  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;

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

  // Total number of bonds (sum of number of neighbors)
  int nedges = 0;

  // Number of bonds per atom
  std::vector<int> neigh_per_atom(nlocal, 0);

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
                       [type[j] - 1];
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

  // NequIP only needs positions and types of local atoms,
  // Allegro also needs ghost atom info

  torch::Tensor pos_tensor =
      torch::zeros({nequip_mode ? inum : ntotal, 3}, torch::TensorOptions().dtype(inputtorchtype));
  torch::Tensor edges_tensor =
      torch::zeros({2, nedges}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor ij2type_tensor =
      torch::zeros({nequip_mode ? inum : ntotal}, torch::TensorOptions().dtype(torch::kInt64));

  auto pos = pos_tensor.accessor<inputtype, 2>();
  auto edges = edges_tensor.accessor<long, 2>();
  auto ij2type = ij2type_tensor.accessor<long, 1>();

  std::vector<int> tag2i(nequip_mode ? inum+1 : 0);
  torch::Tensor cell_tensor, cell_inv_tensor;
  torch::Tensor edge_cell_shifts_tensor;
  inputtype* edge_cell_shifts, *cell_inv;
  inputtype periodic_shift[3];
  if (nequip_mode) {
    // cell for NequIP
    cell_tensor = get_cell();
    cell_inv_tensor = cell_tensor.inverse().transpose(0,1);
    cell_inv = cell_inv_tensor.data_ptr<inputtype>();

    // edge cell shifts for NequIP;
    // the integer number of lattice vectors to add to x[j]-x[i]
    // for each edge
    // (note: x[j] and x[i] always within box, local atoms)
    edge_cell_shifts_tensor = torch::zeros({nedges,3}, torch::TensorOptions().dtype(inputtorchtype));
    edge_cell_shifts = edge_cell_shifts_tensor.data_ptr<inputtype>();

    // get tag mapping for NequIP
    // note: from 1 to nlocal (inclusive) since tags are 1-based
    get_tag2i(tag2i);
  }

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  if (debug_mode) {
    if (nequip_mode) printf("NEQUIP edges: i j xi[:] xj[:] cell_shift[:] rij\n");
    else printf("Allegro edges: i j rij\n");
  }
#pragma omp parallel for if(!debug_mode)
  for (int ii = 0; ii < ntotal; ii++) {
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    if (!nequip_mode || ii<inum) {
      pos[i][0] = x[i][0];
      pos[i][1] = x[i][1];
      pos[i][2] = x[i][2];
      ij2type[i] = type_mapper[itype - 1];
    }

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
          cutoff_matrix[itype - 1][jtype - 1];
      if (rsq > cutij * cutij) { continue; }

      edges[0][edge_counter] = i;
      edges[1][edge_counter] = nequip_mode ? tag2i[jtag] : j; // remap to local for NequIP

      // compute cell shift for NequIP as matrix product of inverse cell matrix
      // and edge vector
      inputtype *e_vec = &edge_cell_shifts[3*edge_counter];
      if constexpr (nequip_mode) {
        for (int d = 0; d < 3; d++)
          periodic_shift[d] = x[j][d] - x[tag2i[jtag]][d];

        // edge_cell_shift[e] = round(cell_inv.matmul(periodic_shift))
        for (int d = 0; d < 3; d++) {
          inputtype tmp = 0;
          for (int k = 0; k < 3; k++)
            tmp += cell_inv[3*d+k] * periodic_shift[k];

          e_vec[d] = std::round(tmp);
        }
      }
      if (debug_mode) {
        if (nequip_mode)
          printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", itag-1, jtag-1,
              x[i][0],x[i][1],x[i][2],x[tag2i[jtag]][0],x[tag2i[jtag]][1],x[tag2i[jtag]][2],
              e_vec[0],e_vec[1],e_vec[2],sqrt(rsq));
        else printf("%d %d %.10g\n", itag - 1, jtag - 1, sqrt(rsq));
      }
      edge_counter++;
    }
  }
  if (debug_mode) {
    if (nequip_mode) printf("end NEQUIP edges\n");
    else printf("end Allegro edges\n");
  }

  if (debug_mode && nequip_mode) std::cout << "cell:\n" << cell_tensor << "\n";

  // create dictionary that goes into the model
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor.to(device));
  input.insert("edge_index", edges_tensor.to(device));
  input.insert("atom_types", ij2type_tensor.to(device));
  if (nequip_mode) {
    input.insert("edge_cell_shift", edge_cell_shifts_tensor.to(device));
	// reshape cell to (1, 3, 3), i.e. with batch dims for consistency with nequip conventions
    cell_tensor = cell_tensor.unsqueeze(0);
    input.insert("cell", cell_tensor.to(device));
  }

  return input;
}

template <bool nequip_mode> torch::Tensor PairNequIPAllegro<nequip_mode>::get_cell(){
  torch::Tensor cell_tensor = torch::zeros({3,3}, torch::TensorOptions().dtype(inputtorchtype));
  auto cell = cell_tensor.accessor<inputtype,2>();

  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];

  return cell_tensor;
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::get_tag2i(std::vector<int> &tag2i){
  int inum = list->inum;
  int *ilist = list->ilist;
  tagint *tag = atom->tag;
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int itag = tag[i];

    // Inverse mapping from tag to x/f atom index
    tag2i[itag] = i;
  }
}

template <bool nequip_mode> void PairNequIPAllegro<nequip_mode>::add_custom_output(std::string name)
{
  custom_output_names.push_back(name);
}

namespace LAMMPS_NS {
template class PairNequIPAllegro<false>;
template class PairNequIPAllegro<true>;
}    // namespace LAMMPS_NS
