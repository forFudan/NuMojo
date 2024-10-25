import numojo as nm
from numojo.prelude import *
from time import now
from python import Python, PythonObject
from testing.testing import assert_raises, assert_true

# ===-----------------------------------------------------------------------===#
# Main functions
# ===-----------------------------------------------------------------------===#


fn check[
    dtype: DType
](matrix: nm.mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.equal(matrix.to_numpy(), np_sol)), st)


fn check_is_close[
    dtype: DType
](matrix: nm.mat.Matrix[dtype], np_sol: PythonObject, st: String) raises:
    var np = Python.import_module("numpy")
    assert_true(np.all(np.isclose(matrix.to_numpy(), np_sol, atol=0.1)), st)


# ===-----------------------------------------------------------------------===#
# Creation
# ===-----------------------------------------------------------------------===#


def test_full():
    var np = Python.import_module("numpy")
    check(
        nm.mat.full[f64]((10, 10), 10),
        np.full((10, 10), 10, dtype=np.float64),
        "Full is broken",
    )


# ===-----------------------------------------------------------------------===#
# Arithmetic
# ===-----------------------------------------------------------------------===#


def test_arithmetic():
    var np = Python.import_module("numpy")
    var A = nm.mat.rand[f64]((100, 100))
    var B = nm.mat.rand[f64]((100, 100))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    check_is_close(A + B, Ap + Bp, "Add is broken")
    check_is_close(A - B, Ap - Bp, "Sub is broken")
    check_is_close(A * B, Ap * Bp, "Mul is broken")
    check_is_close(A @ B, np.matmul(Ap, Bp), "Matmul is broken")


def test_logic():
    var np = Python.import_module("numpy")
    var A = nm.mat.ones((5, 1))
    var B = nm.mat.ones((5, 1))
    var Ap = A.to_numpy()
    var Bp = B.to_numpy()
    check(A > B, Ap > Bp, "gt is broken")
    check(A < B, Ap < Bp, "lt is broken")


# ===-----------------------------------------------------------------------===#
# Linear algebra
# ===-----------------------------------------------------------------------===#


def test_transpose():
    var np = Python.import_module("numpy")
    var a = nm.mat.rand[f64]((100, 100))
    var ap = a.to_numpy()
    check_is_close(
        a.transpose(),
        ap.transpose(),
        "Transpose is broken",
    )


def test_solve():
    var np = Python.import_module("numpy")
    var arr1 = nm.mat.rand[f64]((100, 100))
    var arr2 = nm.mat.rand[f64]((100, 100))
    var np_arr1 = arr1.to_numpy()
    var np_arr2 = arr2.to_numpy()
    check_is_close(
        nm.mat.solve(arr1, arr2),
        np.linalg.solve(np_arr1, np_arr2),
        "Solve is broken",
    )
    check_is_close(
        nm.mat.inv(arr1),
        np.linalg.inv(np_arr1),
        "Inverse is broken",
    )
