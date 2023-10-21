from algorithm.functional import unroll
from utils.list import VariadicList
from utils.index import StaticIntTuple as Ind
from tensor import Tensor, TensorShape, TensorSpec




fn array[dt: DType, *Ts: AnyType](a: ListLiteral[Ts]) -> Tensor[dt]:
    alias Types: VariadicList[AnyType] = VariadicList[AnyType](Ts)
    alias count: Int = len(Types)
    let result: Tensor[dt] = Tensor[dt](count)

    @parameter
    fn _cast[i: Int]():
        alias Type: AnyType = Types[i]
        @parameter
        if Type == Int:
            result[i] = SIMD[dt,1](a.get[i, Int]())
        elif Type == FloatLiteral:
            result[i] = SIMD[dt,1](a.get[i, FloatLiteral]())

    unroll[count, _cast]()
    return result



fn array[dt: DType, *a: SIMD[dt,1]]() -> Tensor[dt]:
    alias elements = VariadicList[SIMD[dt,1]](a)
    alias count = len(elements)
    var result: Tensor[dt] = Tensor[dt](count)

    @unroll
    for i in range(count):
        result[i] = elements[i]
    return result



"""
fn _find_shape[*Ts: AnyType](a: ListLiteral) -> TensorShape:
    let result: TensorShape = TensorShape()
    return result

fn _find_spec[*Ts: AnyType]() -> TensorSpec: pass

fn _shape_from_tuple[*Ts: AnyType](a: Tuple[Ts]): pass

fn array[dt: DType, *Ts: AnyType](a: ListLiteral[Ts]): pass
def array[*Ts: AnyType](a: ListLiteral[Ts]): pass
fn zeros(shape: TensorShape): pass
fn zeros(spec: TensorSpec): pass
fn ones(shape: TensorShape): pass
fn ones(spec: TensorSpec): pass
"""



"""
from utils.index import StaticIntTuple as Ind

fn array(*a: Int) -> array[(1),DType.int64]: pass
fn array(a: VariadicList[Int]): pass
fn array[size: Int](a: StaticTuple[size, Int]): pass

fn array(*a: FloatLiteral): pass
fn array(a: VariadicList[FloatLiteral]): pass
fn array[size: Int](a: StaticTuple[size, FloatLiteral]): pass

fn array()
"""