import gdb.printing
from marray_printers import build_marray_pretty_printers

gdb.printing.register_pretty_printer(
    gdb.current_objfile(),
    build_marray_pretty_printers())
