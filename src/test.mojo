def main():
    import numojo as nj

    a = nj.array[DType.int32]([2,3,4,5])
    print(a[1])
    
    b = nj.array[DType.int32, 2.0,3.0,4.0,5.0,6.0]()
    print(b[1])
