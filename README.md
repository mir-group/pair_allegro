# `pair_allegro`: LAMMPS pair style for NequIP and Allegro

This pair style allows you to use Allegro models from the [`nequip`](https://github.com/mir-group/nequip) and [`allegro`](https://github.com/mir-group/allegro) packages in LAMMPS simulations. NequIP is a message-passing graph neural network limited to one MPI rank, while Allegro is a local model and thus supports parallelism, and so `pair_allegro` **supports MPI in LAMMPS**. It also supports OpenMP (better performance) or Kokkos (best performance) for accelerating the pair style.

For more details on Allegro itself, background, and the LAMMPS pair style please see the [`allegro`](https://github.com/mir-group/allegro) package and our paper:
> *Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics* <br/>
> Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, Boris Kozinsky <br/>
> https://www.nature.com/articles/s41467-023-36329-y <br/>
and
> *Scaling the leading accuracy of deep equivariant models to biomolecular simulations of realistic size* <br/>
> Albert Musaelian, Anders Johansson, Simon Batzner, Boris Kozinsky <br/>
> https://doi.org/10.1145/3581784.3627041 <br/>

`pair_allegro` authors: **Anders Johansson**, Albert Musaelian.

## Pre-requisites

* PyTorch or LibTorch >= 1.11.0;  please note that at present we have only thoroughly tested 1.11 on NVIDIA GPUs (see [#311 for NequIP](https://github.com/mir-group/nequip/discussions/311#discussioncomment-5129513)) and 1.13 on AMD GPUs, but newer 2.x versions *may* also work. With newer versions, setting the environment variable `PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1` sometimes helps.

## Usage in LAMMPS

First define the pair style,
```
pair_style	nequip
```
for NequIP models and
```
pair_style	allegro
```
for Allegro models. Then specify the model with
```
pair_coeff	* * deployed.pth <NequIP/Allegro type name for LAMMPS type 1> <NequIP/Allegro type name for LAMMPS type 2> ...
```
where `deployed.pth` is the filename of your trained, **deployed** model.

The names after the model path `deployed.pth` indicate, in order, the names of the model's atom types to use for LAMMPS atom types 1, 2, and so on. The number of names given must be equal to the number of atom types in the LAMMPS configuration (not the NequIP/Allegro model!).
The given names must be consistent with the names specified in the training YAML file in `chemical_symbol_to_type` or `type_names`. Typically, this will be the chemical symbol for each LAMMPS type.

To run with Kokkos (only supported for Allegro models), please see the [LAMMPS Kokkos documentation](https://docs.lammps.org/Speed_kokkos.html#running-on-gpus). Example:
```bash
mpirun -np 8 lmp -sf kk -k on g 4 -pk kokkos newton on neigh full -in in.script
```
to run on 2 nodes with 4 GPUs *each*.

### Compute (currently only supported for Allegro models)
We provide an experimental "compute" that allows you to extract custom quantities from Allegro models, such as [polarization](https://arxiv.org/abs/2403.17207). You can extract either global or per-atom properties with syntax along the lines of
```
compute polarization all allegro polarization 3
compute polarizability all allegro polarizability 9
compute borncharges all allegro/atom born_charge 9 1
```

The name after `allegro[/atom]` is attempted extracted from the dictionary that the Allegro model returns. The following number is the number of elements after flattening the output. In the examples above, polarization is a 3-element global vector, while polarizability and Born charges are global and per-atom 3x3 matrices, respectively.

For per-atom quantities, the second number is a 1/0 flag indicating whether the properties should be reverse-communicated "Newton-style" like forces, which will depend on your property and the specifics of your implementation.


*Note: For extracting multiple quantities, simply use multiple commands. The properties will be extracted from the same dictionary, without any recomputation.*

*Note: The group flag should generally be `all`.*

*Note: Global quantities are assumed extensive and summed across MPI ranks. Keep ghost atoms in mind when trying to think of whether this works for your property; for example, it does not work for Allegro's global energy if there are non-zero energy shifts, as these are also applied to ghost atoms.*

## Building LAMMPS with this pair style

### Download LAMMPS
```bash
git clone --depth=1 https://github.com/lammps/lammps
```
or your preferred method.
(`--depth=1` prevents the entire history of the LAMMPS repository from being downloaded.)

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
PyTorch's CMake will look for MKL automatically for no reason. If it cannot find it (`MKL_INCLUDE_DIR` is not found), you can set it to some existing path, e.g.
```
-DMKL_INCLUDE_DIR=/tmp
```

#### CUDA
CMake will look for CUDA and cuDNN. You may have to explicitly provide the path for your CUDA installation (e.g. `-DCUDA_TOOLKIT_ROOT_DIR=/usr/lib/cuda/`).

Note that the CUDA that comes with PyTorch when installed with `conda` (the `cudatoolkit` package) is usually insufficient (see [here](https://github.com/pytorch/extension-cpp/issues/26), for example) and you may have to install full CUDA seperately. A minor version mismatch between the available full CUDA version and the version of `cudatoolkit` is usually *not* a problem, as long as the system CUDA is equal or newer. (For example, PyTorch's requested `cudatoolkit==11.3` with a system CUDA of 11.4 works, but a system CUDA 11.1 will likely fail.) cuDNN is also required by PyTorch.

#### With OpenMP (optional, better performance)
`pair_allegro` supports the use of OpenMP to accelerate certain parts of the pair style, by setting `OMP_NUM_THREADS` and using the [LAMMPS OpenMP package](https://docs.lammps.org/Speed_omp.html).

#### With Kokkos (GPU-resident, optional, best performance, most reliable)
`pair_allegro` supports the use of Kokkos to accelerate the pair style on the GPU and avoid host-GPU transfers.
`pair_allegro` supports two setups for Kokkos: pair_style and model both on CPU, or both on GPU. Please ensure you build LAMMPS with the appropriate Kokkos backends enabled for your usecase. For example, to use CUDA GPUs, add:
```
-DPKG_KOKKOS=ON -DKokkos_ENABLE_CUDA=ON
```
to your `cmake` command. See the [LAMMPS documentation](https://docs.lammps.org/Speed_kokkos.html) for more build options and how to correctly run LAMMPS with Kokkos.

*Note: Kokkos support is currently only available for Allegro models.*

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

    A: We now require models to have been trained with stress support, which is achieved by replacing `ForceOutput` with `StressForceOutput` in the training configuration. Note that you do not need to train on stress (though it may improve your potential, assuming your stress data is correct and converged). If you desperately wish to keep using a model without stress output, there are two options: 1) Remove lines that look like [these](https://github.com/mir-group/pair_allegro/blob/99036043e74376ac52993b5323f193dee3f4f401/pair_allegro_kokkos.cpp#L332-L343) in your version of `pair_allegro[_kokkos].cpp` 2) Redeploy the model with an updated config file, as described [here](https://github.com/mir-group/nequip/issues/69#issuecomment-1129273665).
