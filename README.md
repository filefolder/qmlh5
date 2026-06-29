# qmlh5
A QuakeML to HDF5 read/write utility for ObsPy earthquake catalog objects

Large XML catalogs are increasingly difficult to load and share. This utility hopes to alleviate this.

For an example 80,000 event catalog object (with arrivals),
 * writing to h5 now takes < 5 minutes (a 3.3G file)
 * reading the h5 back in takes < 15 minutes


Version 1.1 (29 June 2026)
 - add chunking to significantly reduce RAM consumption for writes
 - refactor reading process (fix O(n^2) bug) for >95% speedup
 - add TQDM to show write progress

Version 1.0 (April 26 2026)
 - initial release

```
import qmlh5

cat = qmlh5.read_catalog('huge_ml_catalog.h5')
# cat is an ObsPy catalog object you can manipulate as needed

# you can write this as QML or SCML or whatever obspy supports
cat.write('out.qml',format='QUAKEML')

# OR you can write out again as an hdf5 object
qmlh5.write_catalog(cat,'out.h5')
```

Yes written in AI (sorry! far too complex otherwise)
Tested pretty thoroughly but as always be careful!
