import os
import time

import numpy as np
import h5py
import bitshuffle.h5


class h5_data_file:
    def __init__(self, filename, dsetname, frames, offset):
        self.filename = filename
        self.dsetname = dsetname
        self.file = h5py.File(filename, "r")
        self.dset = self.file["/data"]
        self.frames = frames
        self.offset = offset
        for j in range(frames):
            assert self.dset.id.get_chunk_info_by_coord((j, 0, 0)).size > 0


class vds_facade:
    """Class to wrap around a HDF5 (virtual) data set, returning the chunks on
    demand rather than working through the HDF5 libraries to access the data."""

    def __init__(self, master):
        """Initialise the system from a pointer to an opened master / NeXus
        file."""

        dataset = master["/entry/data/data"]
        root = os.path.split(master.filename)[0]

        self._shape = dataset.shape

        plist = dataset.id.get_create_plist()

        assert plist.get_layout() == h5py.h5d.VIRTUAL

        self._data_files = []

        virtual_count = plist.get_virtual_count()

        for j in range(virtual_count):
            filename = plist.get_virtual_filename(j)
            dsetname = plist.get_virtual_dsetname(j)

            if filename == ".":
                link = master.get(dsetname, getlink=True)
                filename = os.path.join(root, link.filename)
                dsetname = link.path

            vspace = plist.get_virtual_vspace(j)
            frames = vspace.get_regular_hyperslab()[3][0]
            offset = vspace.get_regular_hyperslab()[0][0]

            self._data_files.append(h5_data_file(filename, dsetname, frames, offset))

        self._dtype = self._data_files[0].dset.dtype

    def get_shape(self):
        return self._shape

    def get_dtype(self):
        return self._dtype

    def chunk(self, frame):
        for f in self._data_files:
            if (frame - f.offset) < f.frames:
                local = (frame - f.offset, 0, 0)
                filter_mask, chunk = f.dset.id.read_direct_chunk(local)
                return chunk

    def data(self, frame):
        c = self.chunk(frame)
        c = np.frombuffer(c[12:], dtype=np.uint8)
        return bitshuffle.decompress_lz4(c, self._shape[1:], self._dtype, 0)


if __name__ == "__main__":
    import sys

    with h5py.File(sys.argv[1], "r", swmr=True) as f:
        vf = vds_facade(f)
        print(f"Shape = {vf.get_shape()} dtype = {vf.get_dtype()}")
        t0 = time.time()
        for j in range(vf.get_shape()[0]):
            d = vf.data(j)
            print(j, np.count_nonzero(d == 0xFFFF))
        t1 = time.time()

        print(f"Reading {vf.get_shape()[0]} frames took {(t1 - t0):.1f}s")
