from collections import Counter
import numpy as np
def mismatchkernel(seq1,seq2,substring_len):
    def get_substrings(seq,substring_len):
        seqlength=len(seq)
        substrings=[]
        mismatchedset=[]
     
        for i in range(seqlength-substring_len+1):
            x=seq[i:i+substring_len]
            substrings.append(x)
            mismatchedset.append(x)
            for j in range(len(x)):
                b=x[0:j]+'*'+x[j+1:]
                mismatchedset.append(b)
        return substrings,mismatchedset

       
    seq1_substrings,seq1_mismatchedset=get_substrings(seq1,substring_len)
    seq2_substrings,seq2_mismatchedset=get_substrings(seq2,substring_len)
    seq1_substrings_dict=dict(Counter(seq1_substrings))
    seq2_substrings_dict=dict(Counter(seq2_substrings))
    intersection=list(set(seq1_substrings) & set(seq2_substrings))
    intersection_seq1_count=np.array([seq1_substrings_dict[x] for x in intersection])
    intersection_seq2_count=np.array([seq2_substrings_dict[x] for x in intersection])
    matched_count=intersection_seq1_count*intersection_seq2_count
    penalty=sum(matched_count)*substring_len
    mismatched_intersection=list(set(seq1_mismatchedset) & set(seq2_mismatchedset))
    seq1_dict=dict(Counter(seq1_mismatchedset))
    seq2_dict=dict(Counter(seq2_mismatchedset))
    seq1_mismatched_count=np.array([seq1_dict[x] for x in mismatched_intersection])
    seq2_mismatched_count=np.array([seq2_dict[x] for x in mismatched_intersection])
    mismatched_count=(seq1_mismatched_count)*(seq2_mismatched_count)
    kern=sum(mismatched_count)-penalty

    return kern
