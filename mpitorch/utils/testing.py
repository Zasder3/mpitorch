import pytest
from mpi4py import MPI


def exact_mpi_size(size):
    """Decorator that marks a test as MPI test and checks for exact process count"""

    def wrapper(func):
        def skip_if_wrong_size(*args, **kwargs):
            if MPI.COMM_WORLD.Get_size() != size:
                pytest.skip(f"This test requires exactly {size} MPI processes")
            return func(*args, **kwargs)

        # Apply pytest.mark.mpi() to the wrapped function
        return pytest.mark.mpi()(skip_if_wrong_size)

    return wrapper
