import gdb
import re
 
class ShortVectorPrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        size = int(self.val['_size'])
        N = int(self.val.type.template_argument(1))
        cap = N if size <= N else int(self.val['_capacity'])
        return 'MArray::short_vector<%d> of length %d, capacity %d' % (N, size, cap)

    def children(self):
        size = int(self.val['_size'])
        data = self.val['_alloc']['_data']
        for i in range(size):
            yield ('[%d]' % i, data.dereference())
            data = data + 1

    def display_hint(self):
        return 'array'
    
def str_lookup_function(val):
    lookup_tag = val.type.strip_typedefs().tag
    if lookup_tag == None:
        return None
    regex = re.compile("^MArray::short_vector<.*>$")
    if regex.match(lookup_tag):
        return ShortVectorPrinter(val)
    return None

gdb.pretty_printers.append(str_lookup_function)
