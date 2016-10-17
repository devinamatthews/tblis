# TCI: Thread Control Interface

TCI is an interface for creating and organizing teams of threads. It uses a communicator model, similar to MPI, where threads synchronize and share data through a handle (the `communicator`) to a shared state object (the `context`). Communicators may be split into smaller sub-communicators for hierarchical parallelism, with independent synchronization and communication.

Threads may be initially created using the `tci_parallelize` construct, which blends pthreads-style thread creation with semantics similar to OpenMP parallel blocks. Alternatively, thread contexts may be created ad-hoc and thread communicators joined as needed.

A C++11 interface is also available that provides convient template and object-oriented wrappers.
