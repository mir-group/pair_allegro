# `pair_allegro`: LAMMPS pair style for Allegro

This pair style allows you to use Allegro models from the [`allegro`](https://github.com/mir-group/allegro) package in LAMMPS simulations. Allegro is designed to enable parallelism, and so `pair_allegro` **supports MPI in LAMMPS**. It also supports OpenMP (better performance) or Kokkos (best performance) for accelerating the pair style.

For more details on Allegro itself, background, and the LAMMPS pair style please see the [`allegro`](https://github.com/mir-group/allegro) package and our paper:
> *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics* <br/>
> Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, Boris Kozinsky <br/>
> https://www.nature.com/articles/s41467-023-36329-y <br/>
and
> *Scaling the leading accuracy of deep equivariant models to biomolecular simulations of realistic size* <br/>
> Albert Musaelian, Anders Johansson, Simon Batzner, Boris Kozinsky <br/>
> https://arxiv.org/abs/2304.10061 <br/>

`pair_allegro` authors: **Anders Johansson**, Albert Musaelian.

## Pre-requisites

* PyTorch or LibTorch >= 1.11.0;  please note that at present we **only recommend 1.11** on CUDA systems.

## Usage in LAMMPS

```
pair_style	allegro
pair_coeff	* * deployed.pth <Allegro type name for LAMMPS type 1> <Allegro type name for LAMMPS type 2> ...
```
where `deployed.pth` is the filename of your trained, **deployed** model.

The names after the model path `deployed.pth` indicate, in order, the names of the Allegro model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the Allegro model!).
The given names must be consistent with the names specified in the Allegro training YAML in `chemical_symbol_to_type` or `type_names`. Typically, this will be the chemical symbol for each LAMMPS type.

To run with Kokkos, please see the [LAMMPS Kokkos documentation](https://docs.lammps.org/Speed_kokkos.html#running-on-gpus). Example:
```bash
mpirun -np 8 lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in in.script
```
to run on 2 nodes with 4 GPUs each.

### Compute
We provide an experimental "compute" that allows you to extract custom quantities from Allegro models, such as [polarization](https://arxiv.org/abs/2403.17207). You can extract either global or per-atom properties with syntax along the lines of
```
compute polarization all allegro polarization 3
compute polarizability all allegro polarizability 9
compute borncharges all allegro/atom born_charge 9 1
```

The name after `allegro[/atom]` is attempted extracted from the dictionary that the Allegro model returns. The following number is the number of elements after flattening the output. In the examples above, polarization is a 3-element global vector, while polarizability and Born charges are global and per-atom 3x3 matrices, respectively. For per-atom quantities, the second number is a flag indicating whether the properties should be reverse-communicated "Newton-style" like forces, which will depend on your property and the specifics of your implementation.

*Note: For extracting multiple quantities, simply use multiple commands. The properties will be extracted from the same dictionary, without any recomputation.*

*Note: The quantities will be attempted extracted at every timestep. In the future, we may add support for passing a flag to the model indicating that the "custom" output should be computed.*

*Note: The group flag shoul generally be `all`.*

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone --depth 1 https://github.com/lammps/lammps
```
or your preferred method.
(`--depth 1` prevents the entire history of the LAMMPS repository from being downloaded.)

### Download this repository
```bash
git clone https://github.com/mir-group/pair_allegro
```
or by downloading a ZIP of the source.

### Patch LAMMPS
From the `pair_allegro` directory, run:
```bash
./patch_lammps.sh /path/to/lammps/
```

### Libraries

#### Libtorch
If you have PyTorch installed and are **NOT** using Kokkos:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
```
If you don't have PyTorch installed **OR** are using Kokkos, you need to download LibTorch from the [PyTorch download page](https://pytorch.org/get-started/locally/). **Ensure you download the cxx11 ABI version if using Kokkos.** Unzip the downloaded file, then configure LAMMPS:
```bash
cd lammps
mkdir build
cd build
cmake ../cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

#### MKL
CMake will look for MKL automatically. If it cannot find it (`MKL_INCLUDE_DIR` is not found) and you are using a Python environment, a simple solution is to run `conda install mkl-include` or `pip install mkl-include` and append:
```
-DMKL_INCLUDE_DIR="$CONDA_PREFIX/include"
```
to the `cmake` command if using a `conda` environment, or
```
-DMKL_INCLUDE_DIR=`python -c "import sysconfig;from pathlib import Path;print(Path(sysconfig.get_paths()[\"include\"]).parent)"`
```
if using plain Python and `pip`.

#### CUDA
CMake will look for CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`).

Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) is usually insufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately. A minor version mismatch between the available full CUDA version and the version of `cudatoolkit` is usually *not* a problem, as long as the system CUDA is equal or newer. (For example, PyTorch's requested `cudatoolkit==11.3` with a system CUDA of 11.4 works, but a system CUDA 11.1 will likely fail.) cuDNN is also required by PyTorch.

#### With OpenMP (optional, better performance)
`pair_allegro` supports the use of OpenMP to accelerate certain parts of the pair style, by setting `OMP_NUM_THREADS` and using the [LAMMPS OpenMP package](https://docs.lammps.org/Speed_omp.html).

#### With Kokkos (GPU, optional, best performance)
`pair_allegro` supports the use of Kokkos to accelerate certain parts of the pair style on the GPU to avoid host-GPU transfers.
`pair_allegro` supports two setups for Kokkos: pair_style and model both on CPU, or both on GPU. Please ensure you build LAMMPS with the appropriate Kokkos backends enabled for your usecase. For example, to use CUDA GPUs, add:
```
-DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON
```
to your `cmake` command. See the [LAMMPS documentation](https://docs.lammps.org/Speed_kokkos.html) for more build options and how to correctly run LAMMPS with Kokkos.

### Building LAMMPS
```bash
make -j$(nproc)
```
This gives `lammps/build/lmp`, which can be run as usual with `/path/to/lmp -in in.script`. If you specify `-DCMAKE_INSTALL_PREFIX=/somewhere/in/$PATH` (the default is `$HOME/.local`), you can do `make install` and just run `lmp -in in.script`.

## FAQ

1. Q: My simulation is immediately or bizzarely unstable

   A: Please ensure that your mapping from LAMMPS atom types to NequIP atom types, specified in the `pair_coeff` line, is correct, and that the units are consistent between your training data and your LAMMPS simulation.
2. Q: I get the following error:
   ```
    instance of 'c10::Error'
        what():  PytorchStreamReader failed locating file constants.pkl: file not found
   ```

   A: Make sure you remembered to deploy (compile) your model using `nequip-deploy`, and that the path to the model given with `pair_coeff` points to a deployed model `.pth` file, **not** a file containing only weights like `best_model.pth`.
3. Q: I get the following error:
   ```
    instance of 'c10::Error'
        what():  isTuple()INTERNAL ASSERT FAILED
   ```

   A: We've seen this error occur when you try to load a TorchScript model deployed from PyTorch>1.11 in LAMMPS built against 1.11. Try redeploying your model (retraining not necessary) in a PyTorch 1.11 install.
4. Q: I get the following error:
    ```
    Exception: Argument passed to at() was not in the map
    ```

    A: We now require models to have been trained with stress support, which is achieved by replacing `ForceOutput` with `StressForceOutput` in the training configuration. Note that you do not need to train on stress (though it may improve your potential, assuming your stress data is correct and converged). If you desperately wish to keep using a model without stress output, you can remove lines that look like [these](https://github.com/mir-group/pair_allegro/blob/99036043e74376ac52993b5323f193dee3f4f401/pair_allegro_kokkos.cpp#L332-L343) in your version of `pair_allegro[_kokkos].cpp`.
