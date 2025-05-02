# `pair_nequip` and `pair_allegro`: LAMMPS pair styles

These pair styles allow you to use models from the [NequIP framework](https://github.com/mir-group/nequip) in LAMMPS simulations. This repository provides two pair styles: `pair_nequip` is for the NequIP message-passing GNN model, which is limited to one MPI rank; `pair_allegro` is for the strictly local Allegro model, which supports parallel execution and MPI in LAMMPS.

 - [Usage](#usage)
 - [Installation](#installation)
 - [References & citing](#references--citing)
 - [FAQ](#faq)
 - [Community, contact, questions, and contributing](#community-contact-questions-and-contributing)


> [!IMPORTANT]
> A [major backwards-incompatible update](./docs/guide/upgrading.md) to the `nequip` framework was released on April 23rd 2025 as version v0.7.0 including the updated pair styles here. The previous version of `pair_allegro` for use with older models can be found in this repository as version v0.6.0. The previous version of `pair_nequip` can be found in the [`pair_nequip` repository](https://github.com/mir-group/pair_nequip) as version v0.6.0.

## Usage
Before you can use a model in LAMMPS, you must compile it using `nequip-compile`. For more information, please see the [NequIP framework documentation](https://nequip.readthedocs.io). The output of `nequip-compile` should be a `.nequip.pth` or a `.nequip.pt2` file for the TorchScript and AOTI compilers, respectively.

In your LAMMPS script, first define the pair style
```
pair_style	nequip
```
for NequIP models and
```
pair_style	allegro
```
for Allegro models. Then specify the model file to use with
```
pair_coeff	* * my-compiled-model.nequip.pth/pt2 <model type name for LAMMPS type 1> <model type name for LAMMPS type 2> ...
```
where `my-compiled-model.nequip.pth/pt2` is the filename of your trained and **compiled** model, output from `nequip-compile`.

The names after the model file name indicate, in order, the names of the model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the NequIP/Allegro model!).
The given names must be consistent with the model's type names that were specified in its training YAML file in the `type_names` option (under `training_module.model`). Typically, this will be the chemical symbol for each LAMMPS type.

### Running with Kokkos
To run with Kokkos (only supported for Allegro models), please see the [LAMMPS Kokkos documentation](https://docs.lammps.org/Speed_kokkos.html#running-on-gpus). Example:
```bash
mpirun -np 8 lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in in.script
```
to run on 2 nodes with 4 GPUs *each*.

## Installation

### Download LAMMPS
```bash
git clone --depth=1 https://github.com/lammps/lammps
```
or your preferred method (`--depth=1` prevents the entire history of the LAMMPS repository from being downloaded).

### Download this repository
```bash
git clone --depth=1 https://github.com/mir-group/pair_nequip_allegro
```
or by downloading a ZIP of the source.

### Patch LAMMPS
From the `pair_nequip_allegro` directory, run:
```bash
./patch_lammps.sh /path/to/lammps/
```

### Configure LAMMPS with CMake

For general information on building LAMMPS with CMake, see [the LAMMPS documentation](https://docs.lammps.org/Build_cmake.html).

In your LAMMPS source directory, you will run something like:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake [options]
```
The following subsections discuss options to include that are specific to `pair_nequip_allegro`. You may need to try to configure and build LAMMPS a number of times while revisiting the sections below.

#### AOTI Compilation (recommended, significant performance gains)
To use PyTorch 2 Ahead-of-Time Inductor (AOTI) compilation (described in [our paper](https://arxiv.org/abs/2504.16068)), you must use **at least PyTorch 2.6.0** (and/or corresponding `libtorch`) and configure an additional compile-time flag:
```
-DNEQUIP_AOT_COMPILE=ON
```
Look out for the following in the CMake output to confirm:
```
-- << NEQUIP flags >>
-- NEQUIP_AOT_COMPILE is enabled/disabled.
```
These steps are necessary to run the pair styles with AOTI compiled models (those with the `.nequip.pt2` extension).

#### `libtorch` (required)

##### without Kokkos
If you have PyTorch installed in your Python environment:
```bash
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

If you don't have PyTorch installed and will not use AOTI, you can download `libtorch` from the [PyTorch download page](https://pytorch.org/get-started/locally/). Unzip the downloaded file, then configure LAMMPS:
```bash
-DCMAKE_PREFIX_PATH=/path/to/unzipped/libtorch
```

##### with Kokkos
If you have PyTorch installed, run the following command:
```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```
If it returns `True`, and use
```bash
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```

If it returns `False`, first try to use the `libtorch` from the [PyTorch download page](https://pytorch.org/get-started/locally/). **Ensure that you download a `cxx11 abi` version.** Unzip the downloaded file, and use:
```bash
-DCMAKE_PREFIX_PATH=/path/to/unzipped/libtorch
```

If you are using AOTI compilation, the pre-built `libtorch` may fail to work. In this case, try installing and building all of PyTorch from source in a new Python environment **using the ABI11 flags**. Then only use that Python environment to run `nequip-compile` and to build LAMMPS using:
```bash
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```
The command at the top of this section should return `True` in this new Python environment.

#### MKL
PyTorch's CMake will look for MKL automatically for no reason. If it cannot find it (`MKL_INCLUDE_DIR` is not found), you can set it to some existing path, e.g.
```
-DMKL_INCLUDE_DIR=/tmp
```

#### CUDA
CMake will look for CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`).

Note that the CUDA that comes with PyTorch when installed with `conda` is insufficient and you may have to install full CUDA seperately. A minor version mismatch between the available CUDA version and PyTorch's CUDA version is usually *not* a problem, as long as the system CUDA's minor version is the same or newer. cuDNN is also required by PyTorch.

#### Kokkos (recommended, best GPU performance and most reliable)
`pair_allegro` supports the use of Kokkos to accelerate the pair style on the GPU and avoid host-GPU transfers.
`pair_allegro` supports two setups for Kokkos: pair_style and model both on CPU, or both on GPU. Please ensure you build LAMMPS with the appropriate Kokkos backends enabled for your usecase. For example, to use CUDA GPUs, add:
```
-DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON
```
to your `cmake` command. See the [LAMMPS documentation](https://docs.lammps.org/Speed_kokkos.html) for more build options and how to correctly run LAMMPS with Kokkos.

Kokkos support is currently only available for `pair_style allegro`.

#### OpenMP (optional, better performance, mutually exclusive with Kokkos)
`pair_allegro` supports the use of OpenMP to accelerate certain parts of the pair style, by setting `OMP_NUM_THREADS` and using the [LAMMPS OpenMP package](https://docs.lammps.org/Speed_omp.html).
OpenMP and Kokkos are mutually exclusive.

OpenMP supports both `pair_style nequip` and `pair_style allegro`.

### Building LAMMPS
```bash
make -j$(nproc)
```
This produces an executable `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. The [LAMMPS documentation](https://docs.lammps.org/Build_cmake.html) has more details.


## References & citing

**Any and all use of this software, in whole or in part, should clearly acknowledge and link to this repository.**

Please see the [`nequip`](https://github.com/mir-group/nequip?tab=readme-ov-file#references--citing) and [`allegro`](https://github.com/mir-group/allegro?tab=readme-ov-file#references--citing) repositories for relevant citations.

## FAQ

1. Q: My simulation is immediately or bizzarely unstable

   A: Please ensure that your mapping from LAMMPS atom types to NequIP framework atom types, specified in the `pair_coeff` line, is correct, and that the units are consistent between your training data and your LAMMPS simulation.
2. Q: I get the following error:
   ```
    instance of 'c10::Error'
        what():  PytorchStreamReader failed locating file constants.pkl: file not found
   ```

   A: Make sure you intended to use TorchScript and that you correctly compiled your model to TorchScript with `nequip-compile`.


## Community, contact, questions, and contributing

If you find a bug or have a proposal for a feature, please post it in the [Issues](https://github.com/mir-group/pair_nequip_allegro/issues).
If you have a self-contained question or other discussion topic, try our [GitHub Discussions](https://github.com/mir-group/pair_nequip_allegro/discussions).

**If your post is related to the NequIP software framework in general or the `allegro` extension package, please post in the issues or discussions on those repositories.** Discussions on this repository should be specific to the pair styles.

Active users and interested developers are invited to join us on the NequIP community chat server, which is hosted on the excellent [Zulip](https://zulip.com/) software.
Zulip is organized a little bit differently than chat software like Slack or Discord that you may be familiar with: please review [their introduction](https://zulip.com/help/introduction-to-topics) before posting.
[Fill out the interest form for the NequIP community here](https://forms.gle/mEuonVCHdsgTtLXy7).

We can also be reached by email at allegro-nequip@g.harvard.edu.