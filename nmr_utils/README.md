# dyn-dipolar

Average dipolar tensors and chemical shifts over a molecular dynamics trajectory stored as
multiple VASP calculations.

Directory layout:

trajectory/

    frame0001/
        CONTCAR
        frame.magres

    frame0002/
        CONTCAR
        frame.magres

    ...

Usage

```bash
dyn-dipolar trajectory --atoms 5 12
```

The program computes the dipolar tensor for every snapshot using Soprano
and prints the trajectory average.