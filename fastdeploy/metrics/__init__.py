from typing import List
"""
metrics
"""

def build_buckets(mantissa_lst: List[int], max_value: int) -> List[int]:
    """
    Generate a list of bucket boundaries using a set of mantissas scaled by powers of 10,
    stopping when the generated value exceeds the specified maximum value.
    """
    exponent = 0
    buckets: List[int] = []
    while True:
        for m in mantissa_lst:
            value = m * 10 ** exponent
            if value <= max_value:
                buckets.append(value)
            else:
                return buckets
        exponent += 1


def build_1_2_5_buckets(max_value: int) -> List[int]:
    """
    Generate a bucket list using the common [1, 2, 5] mantissa pattern,
    scaled by powers of 10 up to the specified maximum value.
    """
    return build_buckets([1, 2, 5], max_value)