# sectioned.py

# Imports
import numpy as np

# parse_k
def parse_k(k):
    """parse_k(k) -> Parse the `k` vector which defines whether a rotation is fixed,
       independent, or defined by a basis. Complete specification of the instance, basis,
       and basis coefficients is possible."""
    inst_info, basis_info, coeff_info, k_lookup = {}, {}, {}, []
    l, i = 0, 0
    while i < k.shape[0]:
        k_lookup.append(i)

        if k[i] == 0:
            # fixed global rotation
            i += 1
        elif k[i] < 0:
            # instance global rotation
            inst_info.setdefault(k[i+1], []).append(l)
            i += 2
        else:
            # basis rotation
            n = k[i]
            for j in xrange(n):
                basis_info.setdefault(k[i+1+2*j], []).append(l)
                coeff_info.setdefault(k[i+1+2*j+1], []).append(l)
            i += 2*n + 1

        l += 1

    k_lookup = np.asarray(k_lookup, dtype=np.int32)

    return inst_info, basis_info, coeff_info, k_lookup


