from algorithm.functional import unroll
from algorithm.functional import _variadic_get
from utils.list import VariadicList
from tensor import Tensor, TensorShape, TensorSpec

fn array[dt: DType, *Ts: AnyType](a: ListLiteral[Ts]) -> Tensor[dt]:
    alias Types: VariadicList[AnyType] = VariadicList[AnyType](Ts)
    alias FirstType: AnyType = Types[0]
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


'''
from utils.index import StaticIntTuple as Ind

fn array(*a: Int) -> array[(1),DType.int64]: pass
fn array(a: VariadicList[Int]): pass
fn array[size: Int](a: StaticTuple[size, Int]): pass

fn array(*a: FloatLiteral): pass
fn array(a: VariadicList[FloatLiteral]): pass
fn array[size: Int](a: StaticTuple[size, FloatLiteral]): pass

fn array()
'''