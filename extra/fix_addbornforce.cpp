/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_addbornforce.h"

#include "atom.h"
#include "atom_masks.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "memory.h"
#include "force.h"
#include "modify.h"
#include "region.h"
#include "update.h"
#include "variable.h"
#include "pair_allegro.h"
#include <torch/torch.h>
#include <iostream>

#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum { NONE, CONSTANT, EQUAL, ATOM };

/* ---------------------------------------------------------------------- */
/*
 * fix ID group-ID addbornforce ex ey ez keyword value
 */

FixAddBornForce::FixAddBornForce(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg), xstr(nullptr), ystr(nullptr), zstr(nullptr),
      idregion(nullptr), region(nullptr), efieldatom(nullptr) {
  if (narg < 6)
    utils::missing_cmd_args(FLERR, "fix addbornforce", error);

  dynamic_group_allow = 1;
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 0;

  if (utils::strmatch(arg[3], "^v_")) {
    xstr = utils::strdup(arg[3] + 2);
  } else {
    xvalue = utils::numeric(FLERR, arg[3], false, lmp);
    xstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4], "^v_")) {
    ystr = utils::strdup(arg[4] + 2);
  } else {
    yvalue = utils::numeric(FLERR, arg[4], false, lmp);
    ystyle = CONSTANT;
  }
  if (utils::strmatch(arg[5], "^v_")) {
    zstr = utils::strdup(arg[5] + 2);
  } else {
    zvalue = utils::numeric(FLERR, arg[5], false, lmp);
    zstyle = CONSTANT;
  }

  // always called, makes no sense to have nevery > 1
  nevery = 1;

  int iarg = 6;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix addbornforce region", error);
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region)
        error->all(FLERR, "Region {} for fix addbornforce does not exist",
                   arg[iarg + 1]);
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown fix addbornforce keyword: {}", arg[iarg]);
  }

  reduced_flag = 0;

  maxatom = 1;
  memory->create(efieldatom, maxatom, 3, "addbornforce:efieldatom");

  // KOKKOS package

  datamask_read = X_MASK | F_MASK | MASK_MASK | IMAGE_MASK;
  datamask_modify = F_MASK;

  ((PairAllegro<lowhigh> *)force->pair)->add_custom_output("born_charge");
  ((PairAllegro<lowhigh> *)force->pair)->add_custom_output("polarization");
  ((PairAllegro<lowhigh> *)force->pair)->add_custom_output("polarizability");
}

/* ---------------------------------------------------------------------- */

FixAddBornForce::~FixAddBornForce() {
  delete[] xstr;
  delete[] ystr;
  delete[] zstr;
  delete[] idregion;
  memory->destroy(efieldatom);
}

/* ---------------------------------------------------------------------- */

int FixAddBornForce::setmask() {
  int mask = 0;
  mask |= POST_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAddBornForce::init() {
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR, "Variable {} for fix addbornforce does not exist",
                 xstr);
    if (input->variable->equalstyle(xvar))
      xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar))
      xstyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addbornforce is invalid style",
                 xstr);
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR, "Variable {} for fix addbornforce does not exist",
                 ystr);
    if (input->variable->equalstyle(yvar))
      ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar))
      ystyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addbornforce is invalid style",
                 ystr);
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR, "Variable {} for fix addbornforce does not exist",
                 zstr);
    if (input->variable->equalstyle(zvar))
      zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar))
      zstyle = ATOM;
    else
      error->all(FLERR, "Variable {} for fix addbornforce is invalid style",
                 zstr);
  }

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region)
      error->all(FLERR, "Region {} for fix addbornforce does not exist",
                 idregion);
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else
    varflag = CONSTANT;

  if (varflag != CONSTANT)
    error->warning(FLERR, "non-constant efield is experimental, strongly distrust scalar/vector output");
}

/* ---------------------------------------------------------------------- */

void FixAddBornForce::setup(int vflag) {
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    error->all(FLERR, "unsupported integration");
  }
}

/* ---------------------------------------------------------------------- */

void FixAddBornForce::min_setup(int vflag) { post_force(vflag); }

/* ---------------------------------------------------------------------- */

void FixAddBornForce::post_force(int vflag) {
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  imageint *image = atom->image;
  double v[6];
  int nlocal = atom->nlocal;

  if (update->ntimestep % nevery)
    return;

  // virial setup

  v_init(vflag);

  // update region if necessary

  if (region)
    region->prematch();

  // reallocate efieldatom array if necessary

  if (varflag == ATOM && atom->nmax > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(efieldatom);
    memory->create(efieldatom, maxatom, 3, "addbornforce:efieldatom");
  }

  reduced_flag = 0;
  extraenergy = extrapolarization[0] = extrapolarization[1] = extrapolarization[2] = 0.0;

  torch::Tensor born_tensor =
    ((PairAllegro<lowhigh> *) force->pair)->custom_output.at("born_charge").cpu();
  this->born_tensor = born_tensor;

  auto born = born_tensor.accessor<double,3>();

  // reverse comm and add up Born charges, Newton-style
  comm->reverse_comm(this, 9);

  // constant force
  // potential energy = - x dot f in unwrapped coords

  if (varflag == CONSTANT) {
    double unwrap[3];
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0], x[i][1], x[i][2]))
          continue;
        domain->unmap(x[i], image[i], unwrap);
        // row-vector efield * Born charge matrix
        // since Zij = dFj/dei = dPi/drj
        for (int j = 0; j < 3; j++) {
          f[i][j] += xvalue * born[i][0][j] + yvalue*born[i][1][j] + zvalue*born[i][2][j];
        }
      }

    // variable force, wrap with clear/add
    // potential energy = evar if defined, else 0.0
    // wrap with clear/add

  } else {
    // TODO: Figure out extra energy/polarization in this case
    double unwrap[3];

    modify->clearstep_compute();

    if (xstyle == EQUAL)
      xvalue = input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar, igroup, &efieldatom[0][0], 3, 0);
    if (ystyle == EQUAL)
      yvalue = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar, igroup, &efieldatom[0][1], 3, 0);
    if (zstyle == EQUAL)
      zvalue = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar, igroup, &efieldatom[0][2], 3, 0);

    modify->addstep_compute(update->ntimestep + 1);

    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) {
        if (region && !region->match(x[i][0], x[i][1], x[i][2]))
          continue;
        domain->unmap(x[i], image[i], unwrap);
        if (xstyle == ATOM)
          xvalue = efieldatom[i][0];
        if (ystyle == ATOM)
          yvalue = efieldatom[i][1];
        if (zstyle == ATOM)
          zvalue = efieldatom[i][2];

        // row-vector efield * Born charge matrix
        // since Zij = dFj/dei = dPi/drj
        for (int j = 0; j < 3; j++) {
          f[i][j] += xvalue * born[i][0][j] + yvalue*born[i][1][j] + zvalue*born[i][2][j];
        }
      }
    }
  }


  torch::Tensor polarization_tensor =
    ((PairAllegro<lowhigh> *) force->pair)->custom_output.at("polarization").cpu();
  torch::Tensor polarizability_tensor =
    ((PairAllegro<lowhigh> *) force->pair)->custom_output.at("polarizability").cpu();

  // std::cout << "polarization: " << polarization_tensor << std::endl;
  // std::cout << "polarizability: " << polarizability_tensor << std::endl;

  auto polarization = polarization_tensor.accessor<double, 2>();
  auto polarizability = polarizability_tensor.accessor<double, 3>();

  for (int i = 0; i <3; i++) {
    extrapolarization[i] = xvalue*polarizability[0][0][i] + yvalue*polarizability[0][1][i] + zvalue*polarizability[0][2][i];
  }

  extraenergy = -(xvalue * (polarization[0][0]+extrapolarization[0])
                + yvalue * (polarization[0][1]+extrapolarization[1])
                + zvalue * (polarization[0][2]+extrapolarization[2]));

}

/* ---------------------------------------------------------------------- */

void FixAddBornForce::min_post_force(int vflag) { post_force(vflag); }

int FixAddBornForce::pack_reverse_comm(int n, int first, double *buf){
  auto born = born_tensor.accessor<double,3>();
  int m = 0;
  int last = first + n;

  for (int i = first; i < last; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        buf[m++] = born[i][j][k];
      }
    }
  }
  return m;
}

void FixAddBornForce::unpack_reverse_comm(int n, int *list, double *buf){
  auto born = born_tensor.accessor<double,3>();
  int m = 0;

  for (int i = 0; i < n; i++) {
    int ii = list[i];
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        born[ii][j][k] += buf[m++];
      }
    }
  }
}

/* ----------------------------------------------------------------------
   potential energy of added force
------------------------------------------------------------------------- */

double FixAddBornForce::compute_scalar() {
  // only sum across procs one time

  double tmp = 0.0;
  MPI_Allreduce(&extraenergy, &tmp, 1, MPI_DOUBLE, MPI_SUM, world);
  return tmp;
}

/* ----------------------------------------------------------------------
   return components of total force on fix group before force was changed
------------------------------------------------------------------------- */

double FixAddBornForce::compute_vector(int n) {
  // only sum across procs one time

  if (reduced_flag == 0) {
    MPI_Allreduce(MPI_IN_PLACE, extrapolarization, 4, MPI_DOUBLE, MPI_SUM, world);
    reduced_flag = 1;
  }
  return extrapolarization[n];
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixAddBornForce::memory_usage() {
  double bytes = 0.0;
  if (varflag == ATOM)
    bytes = maxatom * 3 * sizeof(double);
  return bytes;
}
