/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
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

#include <cmath>
#include "kokkos.h"
#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "error.h"
#include "atom_masks.h"
#include "math_const.h"

#include <pair_allegro_kokkos.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>

using namespace LAMMPS_NS;
using namespace MathConst;
namespace Kokkos {
  template <>
  struct reduction_identity<s_FEV_FLOAT> {
    KOKKOS_FORCEINLINE_FUNCTION static s_FEV_FLOAT sum() {
      return s_FEV_FLOAT();
    }
  };
}

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

template<Precision precision>
PairAllegroKokkos<precision>::PairAllegroKokkos(LAMMPS *lmp) : PairAllegro<precision>(lmp)
{
  this->respa_enable = 0;


  this->atomKK = (AtomKokkos *) this->atom;
  this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  this->datamask_read = X_MASK | F_MASK | TAG_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  this->datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<Precision precision>
PairAllegroKokkos<precision>::~PairAllegroKokkos()
{
  if (!this->copymode) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->eatom = NULL;
    this->vatom = NULL;
  }
}

/* ---------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) this->no_virial_fdotr_compute = 1;

  this->ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (this->eflag_atom) {
    this->memoryKK->destroy_kokkos(k_eatom,this->eatom);
    this->memoryKK->create_kokkos(k_eatom,this->eatom,this->maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (this->vflag_atom) {
    this->memoryKK->destroy_kokkos(k_vatom,this->vatom);
    this->memoryKK->create_kokkos(k_vatom,this->vatom,this->maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  this->atomKK->sync(this->execution_space,this->datamask_read);
  if (eflag || vflag) this->atomKK->modified(this->execution_space,this->datamask_modify);
  else this->atomKK->modified(this->execution_space,F_MASK);

  x = this->atomKK->k_x.template view<DeviceType>();
  f = this->atomKK->k_f.template view<DeviceType>();
  tag = this->atomKK->k_tag.template view<DeviceType>();
  type = this->atomKK->k_type.template view<DeviceType>();
  nlocal = this->atom->nlocal;
  newton_pair = this->force->newton_pair;
  nall = this->atom->nlocal + this->atom->nghost;

  const int inum = this->list->inum;
  const int ignum = inum + this->atom->nghost;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(this->list);
  d_ilist = k_list->d_ilist;
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;

  if (inum==0) return; // empty domain

  this->copymode = 1;


  // build short neighbor list

  const int max_neighs = d_neighbors.extent(1);
  // TODO: check inum/ignum here
  const int n_atoms = neighflag == FULL ? inum : inum;



  if(d_numneigh_short.extent(0) < inum){
    d_numneigh_short = decltype(d_numneigh_short)();
    d_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::numneighs_short") ,inum);
    d_cumsum_numneigh_short = decltype(d_cumsum_numneigh_short)();
    d_cumsum_numneigh_short = Kokkos::View<int*,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("Allegro::cumsum_numneighs_short") ,inum);
  }
  if(d_neighbors_short.extent(0) < inum || d_neighbors_short.extent(1) < max_neighs){
    d_neighbors_short = decltype(d_neighbors_short)();
    d_neighbors_short = Kokkos::View<int**,DeviceType>(Kokkos::ViewAllocateWithoutInitializing("FLARE::neighbors_short") ,inum,max_neighs);
    //c10::cuda::CUDACachingAllocator::emptyCache();
  }

  // compute short neighbor list
  auto d_numneigh_short = this->d_numneigh_short;
  auto d_neighbors_short = this->d_neighbors_short;
  auto d_cumsum_numneigh_short = this->d_cumsum_numneigh_short;
  double cutoff = this->cutoff;
  auto x = this->x;
  auto d_type = this->type;
  auto d_ilist = this->d_ilist;
  auto d_numneigh = this->d_numneigh;
  auto d_neighbors = this->d_neighbors;
  auto f = this->f;
  auto d_eatom = this->d_eatom;
  auto d_type_mapper = this->d_type_mapper;
  auto d_cutoff_matrix = this->d_cutoff_matrix;

  Kokkos::parallel_for("Allegro: Short neighlist", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii){
      const int i = d_ilist[ii];
      const X_FLOAT xtmp = x(i,0);
      const X_FLOAT ytmp = x(i,1);
      const X_FLOAT ztmp = x(i,2);

      const int si = d_type[i] - 1;

      const int jnum = d_numneigh[i];
      int inside = 0;
      for (int jj = 0; jj < jnum; jj++) {
        int j = d_neighbors(i,jj);
        j &= NEIGHMASK;

        const int sj = d_type[j] - 1;
        const double ijcut = d_cutoff_matrix(si, sj); //TODO

        //printf("i=%3d j=%3d ti=%d tj=%d cut=%.2f\n", i, j, d_type[i], d_type[j], ijcut);

        const X_FLOAT delx = xtmp - x(j,0);
        const X_FLOAT dely = ytmp - x(j,1);
        const X_FLOAT delz = ztmp - x(j,2);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < ijcut*ijcut) {
          d_neighbors_short(ii,inside) = j;
          inside++;
        }
      }
      d_numneigh_short(ii) = inside;
  });
  Kokkos::deep_copy(d_cumsum_numneigh_short, d_numneigh_short);

  Kokkos::parallel_scan("Allegro: cumsum shortneighs", Kokkos::RangePolicy<DeviceType>(0,inum), KOKKOS_LAMBDA(const int ii, int& update, const bool is_final){
      const int curr_val = d_cumsum_numneigh_short(ii);
      update += curr_val;
      if(is_final) d_cumsum_numneigh_short(ii) = update;
  });
  int nedges = 0;
  Kokkos::View<int*, Kokkos::HostSpace> nedges_view("Allegro: nedges",1);
  Kokkos::deep_copy(nedges_view, Kokkos::subview(d_cumsum_numneigh_short, Kokkos::make_pair(inum-1, inum)));
  nedges = nedges_view(0);

  //auto nn = Kokkos::create_mirror_view(d_numneigh_short);
  //Kokkos::deep_copy(nn, d_numneigh_short);
  //auto cs = Kokkos::create_mirror_view(d_cumsum_numneigh_short);
  //Kokkos::deep_copy(cs, d_cumsum_numneigh_short);
  //printf("INUM=%d, GNUM=%d, IGNUM=%d\n", inum, list->gnum, ignum);
  //printf("NEDGES: %d\nnumneigh_short cumsum\n",nedges);
  //for(int i = 0; i < inum; i++){
  //  printf("%d %d\n", nn(i), cs(i));
  //}

  double padding_factor = 1.05;

  if(d_edges.extent(1) < nedges || nedges*padding_factor*padding_factor < d_edges.extent(1)){
    d_edges = decltype(d_edges)();
    d_edges = decltype(d_edges)("Allegro: edges", 2, padding_factor*nedges);
  }
  if(d_ij2type.extent(0) < ignum+2 || (ignum+2)*padding_factor*padding_factor < d_ij2type.extent(0)){
    d_ij2type = decltype(d_ij2type)();
    d_ij2type = decltype(d_ij2type)("Allegro: ij2type", padding_factor*ignum+2);
    d_xfloat = decltype(d_xfloat)();
    d_xfloat = decltype(d_xfloat)("Allegro: xfloat", padding_factor*ignum+2, 3);
  }

  auto d_edges = this->d_edges;
  auto d_ij2type = this->d_ij2type;
  auto d_xfloat = this->d_xfloat;

  Kokkos::parallel_for("Allegro: store type mask and x", Kokkos::RangePolicy<DeviceType>(0, ignum), KOKKOS_LAMBDA(const int i){
      d_ij2type(i) = d_type_mapper(d_type(i)-1);
      d_xfloat(i,0) = x(i,0);
      d_xfloat(i,1) = x(i,1);
      d_xfloat(i,2) = x(i,2);
  });

  int max_atoms = d_ij2type.extent(0);
  Kokkos::parallel_for("Allegro: store fake atoms", Kokkos::RangePolicy<DeviceType>(ignum, max_atoms), KOKKOS_LAMBDA(const int i){
      d_ij2type(i) = d_type_mapper(0);
      d_xfloat(i,0) = i==max_atoms-1 ? 100.0 : 0.0;
      d_xfloat(i,1) = 0.0;
      d_xfloat(i,2) = 0.0;
  });

  Kokkos::parallel_for("Allegro: create edges", Kokkos::TeamPolicy<DeviceType>(inum, Kokkos::AUTO()), KOKKOS_LAMBDA(const MemberType team_member){
      const int ii = team_member.league_rank();
      const int i = d_ilist(ii);
      const int startedge = ii==0 ? 0 : d_cumsum_numneigh_short(ii-1);
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team_member, d_numneigh_short(ii)), [&] (const int jj){
          d_edges(0, startedge + jj) = i;
          d_edges(1, startedge + jj) = d_neighbors_short(ii,jj);
      });
  });

  int max_edges = d_edges.extent(1);
  Kokkos::parallel_for("Allegro: store fake edges", Kokkos::RangePolicy<DeviceType>(nedges, max_edges), KOKKOS_LAMBDA(const int i){
      d_edges(0, i) = max_atoms-2;
      d_edges(1, i) = max_atoms-1;
  });

  torch::Tensor ij2type_tensor = torch::from_blob(d_ij2type.data(), {max_atoms}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor edges_tensor = torch::from_blob(d_edges.data(), {2,max_edges}, {(long) d_edges.extent(1),1}, torch::TensorOptions().dtype(torch::kInt64).device(this->device));
  torch::Tensor pos_tensor = torch::from_blob(d_xfloat.data(), {max_atoms,3}, {3,1}, torch::TensorOptions().device(this->device).dtype(this->inputtorchtype));

  if (this->debug_mode) {
    printf("Allegro edges: i j rij\n");
    for (long i = 0; i < nedges; i++) {
      printf(
        "%ld %ld %.10g\n",
        edges_tensor.index({0, i}).item<long>(),
        edges_tensor.index({1, i}).item<long>(),
        (pos_tensor[edges_tensor.index({0, i}).item<long>()] - pos_tensor[edges_tensor.index({1, i}).item<long>()]).square().sum().sqrt().item<inputtype>()
      );
    }
    printf("end Allegro edges\n");
  }

  c10::Dict<std::string, torch::Tensor> input;
  input.insert("pos", pos_tensor);
  input.insert("edge_index", edges_tensor);
  input.insert("atom_types", ij2type_tensor);
  std::vector<torch::IValue> input_vector(1, input);
  //std::cout << "NequIP model input:\n";
  //std::cout << "pos:\n" << pos_tensor.cpu() << "\n";
  //std::cout << "edge_index:\n" << edges_tensor.cpu() << "\n";
  //std::cout << "atom_types:\n" << ij2type_tensor.cpu() << "\n";

  auto output = this->model.forward(input_vector).toGenericDict();
  torch::Tensor forces_tensor = output.at("forces").toTensor();
  torch::Tensor atomic_energy_tensor = output.at("atomic_energy").toTensor();

  UnmanagedFloatView1D d_atomic_energy(atomic_energy_tensor.data_ptr<outputtype>(), inum);
  UnmanagedFloatView2D d_forces(forces_tensor.data_ptr<outputtype>(), ignum, 3);

  //std::cout << "NequIP model output:\n";
  //std::cout << "forces:\n" << forces_tensor.cpu() << "\n";
  //std::cout << "atomic_energy:\n" << atomic_energy_tensor.cpu() << "\n";

  this->eng_vdwl = 0.0;
  auto eflag_atom = this->eflag_atom;
  Kokkos::parallel_reduce("Allegro: store forces",
      Kokkos::RangePolicy<DeviceType>(0, ignum),
      KOKKOS_LAMBDA(const int i, double &eng_vdwl){
        f(i,0) = d_forces(i,0);
        f(i,1) = d_forces(i,1);
        f(i,2) = d_forces(i,2);
        if(eflag_atom && i < inum){
          d_eatom(i) = d_atomic_energy(i);
        }
        if(i < inum){
          eng_vdwl += d_atomic_energy(i);
        }
      },
      this->eng_vdwl
      );

  if (eflag_atom) {
    // if (need_dup)
    //   Kokkos::Experimental::contribute(d_eatom, dup_eatom);
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if(vflag){
    torch::Tensor v_tensor = output.at("virial").toTensor().cpu();
    auto v = v_tensor.accessor<outputtype, 3>();
    // Convert from 3x3 symmetric tensor format, which NequIP outputs, to the flattened form LAMMPS expects
    // First [0] index on v is batch
    this->virial[0] = v[0][0][0];
    this->virial[1] = v[0][1][1];
    this->virial[2] = v[0][2][2];
    this->virial[3] = v[0][0][1];
    this->virial[4] = v[0][0][2];
    this->virial[5] = v[0][1][2];
  }
  if(this->vflag_atom) {
    this->error->all(FLERR,"Pair style Allegro does not support per-atom virial");
  }

  if (this->vflag_fdotr) pair_virial_fdotr_compute(this);

  this->copymode = 0;

}






/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::coeff(int narg, char **arg)
{
  super::coeff(narg,arg);
  int ntypes = this->atom->ntypes;

  d_type_mapper = IntView1D("Allegro: type_mapper", this->type_mapper.size());
  auto h_type_mapper = Kokkos::create_mirror_view(d_type_mapper);
  for(int i = 0; i < this->type_mapper.size(); i++){
    h_type_mapper(i) = this->type_mapper[i];
  }
  Kokkos::deep_copy(d_type_mapper, h_type_mapper);

  d_cutoff_matrix = View2D("Allegro: cutoff_matrix", ntypes, ntypes);
  auto h_cutoff_matrix = Kokkos::create_mirror_view(d_cutoff_matrix);
  for(int i = 0; i < ntypes; i++){
    for(int j = 0; j < ntypes; j++){
      h_cutoff_matrix(i,j) = this->cutoff_matrix[i][j];
      if (this->comm->me==0) printf("ti=%d tj=%d cut=%.2f\n", i, j, h_cutoff_matrix(i,j));
    }
  }
  Kokkos::deep_copy(d_cutoff_matrix, h_cutoff_matrix); // TODO: Check
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<Precision precision>
void PairAllegroKokkos<precision>::init_style()
{
  super::init_style();

  auto request = this->neighbor->find_request(this);
  request->set_kokkos_host(std::is_same<DeviceType,LMPHostType>::value &&
      !std::is_same<DeviceType,LMPDeviceType>::value);
  request->set_kokkos_device(std::is_same<DeviceType,LMPDeviceType>::value);

  neighflag = this->lmp->kokkos->neighflag;
  if (neighflag != FULL) {
    this->error->all(FLERR,"Needs full neighbor list style with pair_allegro/kk");
  }
}



namespace LAMMPS_NS {
template class PairAllegroKokkos<lowlow>;
template class PairAllegroKokkos<highhigh>;
template class PairAllegroKokkos<lowhigh>;
template class PairAllegroKokkos<highlow>;
}

