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

PairStyle(dice/kk,PairDICEKokkos<LMPDeviceType>)
PairStyle(dice/kk/device,PairDICEKokkos<LMPDeviceType>)
PairStyle(dice/kk/host,PairDICEKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_DICE_KOKKOS_H
#define LMP_PAIR_DICE_KOKKOS_H

#include "pair_dice.h"
#include <pair_kokkos.h>

struct TagSingleBond{};
struct TagB2{};
struct TagBetaB2{};
struct TagNorm2{};
struct Tagw{};
struct Tagu{};
struct TagF{};
struct TagStoreF{};

namespace LAMMPS_NS {

template<class DeviceType>
class PairDICEKokkos : public PairDICE {
 public:
  using MemberType = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairDICEKokkos(class LAMMPS *);
  virtual ~PairDICEKokkos();
  virtual void compute(int, int);
  virtual void coeff(int, char **);
  virtual void init_style();

  KOKKOS_INLINE_FUNCTION
  void v_tally(E_FLOAT (&v)[6], const int &i, const int &j,
      const F_FLOAT &fx, const F_FLOAT &fy, const F_FLOAT &fz, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const;

  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
 protected:
  typedef Kokkos::DualView<int***,DeviceType> tdual_int_3d;
  typedef typename tdual_int_3d::t_dev_const_randomread t_int_3d_randomread;
  typedef typename tdual_int_3d::t_host t_host_int_3d;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_tagint_1d tag;
  typename AT::t_int_1d_randomread type;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;

  using IntView1D = Kokkos::View<int*, Kokkos::LayoutRight, DeviceType>;
  using IntView2D = Kokkos::View<int**, Kokkos::LayoutRight, DeviceType>;
  using LongView1D = Kokkos::View<long*, Kokkos::LayoutRight, DeviceType>;
  using LongView2D = Kokkos::View<long**, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView1D = Kokkos::View<float*, Kokkos::LayoutRight, DeviceType>;
  using UnmanagedFloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;
  using View1D = Kokkos::View<F_FLOAT*, Kokkos::LayoutRight, DeviceType>;
  using View2D = Kokkos::View<F_FLOAT**, Kokkos::LayoutRight, DeviceType>;
  using FloatView2D = Kokkos::View<float**, Kokkos::LayoutRight, DeviceType>;



  IntView1D d_type_mapper;
  LongView1D d_ij2type;
  LongView2D d_edges;
  FloatView2D d_xfloat;



  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  //NeighListKokkos<DeviceType> k_list;

  int neighflag,newton_pair;
  int nlocal,nall,eflag,vflag;

  Kokkos::View<int**,DeviceType> d_neighbors_short;
  Kokkos::View<int*,DeviceType> d_numneigh_short, d_cumsum_numneigh_short;


  friend void pair_virial_fdotr_compute<PairDICEKokkos>(PairDICEKokkos*);
};
}

#endif
#endif

/* ERROR/WARNING messages:

E: Cannot use chosen neighbor list style with pair dice/kk

Self-explanatory.

*/
