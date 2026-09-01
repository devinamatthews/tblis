import gdb
import gdb.printing
 
class ArrayPrinter:
    def __init__(self, val):
        self.__val = val

    def children(self):
        N = int(self.__val.type.template_argument(1))
        data = self.__val['_M_elems']
        for i in range(N):
            yield (f'[{i}]', data[i])

    def display_hint(self):
        return 'array'
 
class ShortVectorPrinter:
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        size = int(self.__val['_size'])
        N = int(self.__val.type.template_argument(1))
        cap = N if size <= N else int(self.__val['_capacity'])
        return f'MArray::short_vector<{N}> of length {size}, capacity {cap}'

    def children(self):
        size = int(self.__val['_size'])
        data = self.__val['_alloc']['_data']
        for i in range(size):
            yield (f'[{i}]', data[i])

    def display_hint(self):
        return 'array'
 
class MArrayPrinter:
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        shape = self.__val['len_']
        stride = self.__val['stride_']
        base = self.__val['base_']
        T = self.__val.type.template_argument(0)
        N = self.__val.type.template_argument(1)
        return f'MArray::marray<{T}{",{N}" if N > 0 else ""}> of shape {shape}, stride {stride}, base {base}'
 
class MArrayViewPrinter:
    def __init__(self, val):
        self.__val = val

    def to_string(self):
        shape = self.__val['len_']
        stride = self.__val['stride_']
        base = self.__val['base_']
        T = self.__val.type.template_argument(0)
        N = self.__val.type.template_argument(1)
        return f'MArray::marray_view<{T}{",{N}" if N > 0 else ""}> of shape {shape}, stride {stride}, base {base}'

def build_marray_pretty_printers():
    pp = gdb.printing.RegexpCollectionPrettyPrinter(
        "MArray")
    pp.add_printer('array', "^std::array<.*>$", ArrayPrinter)
    pp.add_printer('short_vector', "^MArray::(debug::|release::)?short_vector<.*>$", ShortVectorPrinter)
    pp.add_printer('marray', "^MArray::(debug::|release::)?marray<.*>$", MArrayPrinter)
    pp.add_printer('marray_view', "^MArray::(debug::|release::)?marray_view<.*>$", MArrayViewPrinter)
    return pp
    