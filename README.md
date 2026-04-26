# qmlh5
A QuakeML to HDF5 read/write utility for ObsPy earthquake catalog objects

Large XML catalogs are increasingly difficult to load and share. This utility hopes to alleviate this.

```
import qmlh5

cat = qmlh5.read_catalog('huge_ml_catalog.h5')
# cat is an ObsPy catalog object you can manipulate as needed

# you can write this as QML or SCML or whatever obspy supports
cat.write('out.qml',format='QUAKEML')

# OR you can write out again as an hdf5 object
qmlh5.write_catalog(cat,'out.h5')
```

One day may attempt to add this into ObsPy directly...
