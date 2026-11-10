"""Conservative removal of provably identical callback runs before rollback quarantine."""
import numpy as np

LEGACY = 'legacy_v1'
REVISION = 'exact_repeat_v1'


def retained_rows(clocks, values, rows, header_cuts):
    """Only remove exact prefixes replayed after a backward jump in one source.

    Compare complete timestamp groups, including internal header partitions.
    No tolerances, sorting, interpolation, cross-source or prefix-payload guessing.
    A failed proof leaves the whole jump for the original quarantine policy.
    """
    keep=np.ones(len(clocks),bool)
    edges=np.r_[0,np.flatnonzero(np.diff(clocks)<0)+1,len(clocks)]
    removed=0
    for i in range(1,len(edges)-1):
        a,b=int(edges[i]),int(edges[i+1])
        prior=np.flatnonzero(keep[:a])
        if not len(prior):continue
        pc=clocks[prior]
        if np.any(np.diff(pc)<0) or clocks[a]<pc[0]:continue
        upper=min(int(pc[-1]),int(clocks[b-1]))
        end=a+int(np.searchsorted(clocks[a:b],upper,side='right'))
        left=int(np.searchsorted(pc,clocks[a],side='left'))
        right=int(np.searchsorted(pc,upper,side='right'))
        old=prior[left:right]
        if end<=a or not np.array_equal(pc[left:right],clocks[a:end]):continue
        if not np.array_equal(values[old],values[a:end]):continue
        def internal_headers(rr):
            pos=np.searchsorted(rr,header_cuts)
            return pos[(pos>0)&(pos<len(rr))]
        if not np.array_equal(internal_headers(rows[old]),internal_headers(rows[a:end])):continue
        keep[a:end]=False
        removed+=end-a
    return keep,removed
