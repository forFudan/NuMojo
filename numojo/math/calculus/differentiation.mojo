# ===----------------------------------------------------------------------=== #
# implements basic Integral functions
# Last updated: 2024-07-02
# ===----------------------------------------------------------------------=== #

import math
import .. math_funcs as _mf
import ...core as core
from ...core.ndarray import NDArray, NDArrayShape
from ...core.utility_funcs import is_inttype, is_floattype

"""TODO: 
1) add a Variant[NDArray, Scalar, ...] to include all possibilities
2) add edge_order
"""


fn gradient[
    in_dtype: DType, out_dtype: DType = DType.float32
](x: NDArray[in_dtype], spacing: Scalar[in_dtype]) raises -> NDArray[out_dtype]:
    """
    Compute the integral of y over x using the trapezoidal rule.

    Parameters:
        in_dtype: Input data type.
        out_dtype: Output data type, defaults to float32.

    Args:
        x: An array.
        spacing: An array of the same shape as x containing the spacing between adjacent elements.

    Constraints:
        `fdtype` must be a floating-point type if `idtype` is not a floating-point type.

    Returns:
        The integral of y over x using the trapezoidal rule.
    """
    # var result: NDArray[out_dtype] = NDArray[out_dtype](x.shape(), random=False)
    # var space: NDArray[out_dtype] =  NDArray[out_dtype](x.shape(), random=False)
    # if spacing.isa[NDArray[in_dtype]]():
    #     for i in range(x.num_elements()):
    #         space[i] = spacing._get_ptr[NDArray[in_dtype]]()[][i].cast[out_dtype]()

    # elif spacing.isa[Scalar[in_dtype]]():
    #     var int: Scalar[in_dtype] = spacing._get_ptr[Scalar[in_dtype]]()[]
    #     space = numojo.arange[in_dtype, out_dtype](1, x.num_elements(), step=int)

    var result: NDArray[out_dtype] = NDArray[out_dtype](x.shape(), random=False)
    var space: NDArray[out_dtype] = core.arange[in_dtype, out_dtype](
        1, x.num_elements() + 1, step=spacing.cast[in_dtype]()
    )
    var hu: Scalar[out_dtype] = space.get_scalar(1)
    var hd: Scalar[out_dtype] = space.get_scalar(0)
    result.store(0, (x.get_scalar(1).cast[out_dtype]() - x.get_scalar(0).cast[out_dtype]()) / (hu - hd))

    hu = space.get_scalar(x.num_elements() - 1)
    hd = space.get_scalar(x.num_elements() - 2)
    result.store(x.num_elements() - 1, (
        x.get_scalar(x.num_elements() - 1).cast[out_dtype]()
        - x.get_scalar(x.num_elements() - 2).cast[out_dtype]()
    ) / (hu - hd))

    for i in range(1, x.num_elements() - 1):
        var hu: Scalar[out_dtype] = space.get_scalar(i + 1) - space.get_scalar(i)
        var hd: Scalar[out_dtype] = space.get_scalar(i) - space.get_scalar(i - 1)
        var fi: Scalar[out_dtype] = (
            hd**2 * x.get_scalar(i + 1).cast[out_dtype]()
            + (hu**2 - hd**2) * x.get_scalar(i).cast[out_dtype]()
            - hu**2 * x.get_scalar(i - 1).cast[out_dtype]()
        ) / (hu * hd * (hu + hd))
        result.store(i, fi)

    return result
