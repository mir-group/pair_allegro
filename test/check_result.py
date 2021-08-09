from ase.io import read, write
import numpy as np
import sys

reference = np.load("reference0.npz")
print(reference.files)
ref_forces = reference["forces"]
ref_energy = reference["energy"]
print(ref_energy)


# np.savetxt(    sys.stdout, np.concatenate((frame.get_positions(), ref_forces), axis=1), "%.5f")

result = read("output.dump", format="lammps-dump-text")
diff = np.abs(result.get_forces() - ref_forces)
print(np.mean(diff))
print(result.get_forces(), "\n", ref_forces)
