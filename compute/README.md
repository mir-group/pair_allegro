
### Compute

> [!IMPORTANT]
> `compute allegro` only supports Allegro models and `pair_allegro` right now.

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