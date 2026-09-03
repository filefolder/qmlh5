# qmlh5
A QuakeML to HDF5 read/write utility for ObsPy earthquake catalog objects

Large XML catalogs are increasingly difficult to load and share. This utility hopes to alleviate this.

For an example 80,000 event catalog object (with arrivals),
 * writing to h5 now takes < 5 minutes (a 3.3G file)
 * reading the h5 back in takes < 15 minutes


Version 1.1 (29 June 2026)
 - add chunking to significantly reduce RAM consumption for writes
 - refactor reading process (fix O(n^2) bug) for >95% speedup
 - add TQDM to show read & write progress
 - add time filter kwargs (e.g. starttime and endtime) to read_events

Version 1.0 (April 26 2026)
 - initial release

```python
import qmlh5

cat = qmlh5.read_catalog('huge_ml_catalog.h5')
# cat is an ObsPy catalog object

# can also quickly read a specific timerange
cat_2006 = qmlh5.read_catalog('huge_ml_catalog.h5',
                              starttime=UTCDateTime(2006,1,1),
                              endtime=UTCDateTime(2007,1,1))

# you can write this as QML or SCML or whatever obspy supports
cat.write('out.qml',format='QUAKEML')

# OR you can write out again as an hdf5 object
qmlh5.write_catalog(cat,'out.h5')

# Some other convenience queries..
with qmlh5("catalog.h5") as q:
    idx = q.query_bbox(min_lat=35.0, max_lat=40.0,
                       min_lon=-125.0, max_lon=-115.0)

    idx = q.query_time("2021-03-01", "2021-04-01")   # UTCDateTime or float

    idx = q.query_magnitude(min_mag=5.0, max_mag=8.0, mag_type="Mw")

    idx = q.query_radius(center_lat=37.5, center_lon=-121.0,
                         max_radius_deg=1.0, min_radius_deg=0.2)

    # Ray-casting point-in-polygon; vertices are (lat, lon) pairs.
    idx = q.query_polygon([(40, -10), (40, 30), (55, 30), (55, -10)])

    # Depth in *meters*
    idx = q.query_depth(min_depth_m=0.0, max_depth_m=5000)

    # "Defining arrivals" = origin_quality.used_phase_count.
    idx = q.query_arrivals(min_count=8)   # optionally max_count=...

```

# Schema coverage

Every class below is serialized field-for-field against the QuakeML 1.2 BED schema.

| Object | Fields stored |
|---|---|
| `Event` | public ID, preferred origin/magnitude/focal-mechanism IDs, `event_type`, `event_type_certainty`, descriptions, comments, creation info |
| `EventDescription` | text, type |
| `Origin` | time/latitude/longitude/depth (each with 4 error sub-fields), `depth_type`, `time_fixed`, `epicenter_fixed`, reference system / method / earth model IDs, `origin_type`, region, evaluation mode/status |
| `OriginQuality` | associated/used phase & station counts, depth phase count, standard error, azimuthal & secondary azimuthal gaps, ground truth level, min/max/median distance |
| `OriginUncertainty` | horizontal uncertainty (min/max), azimuth of max horizontal uncertainty, preferred description, confidence level |
| `ConfidenceEllipsoid` | semi-major/minor/intermediate axis lengths, major-axis plunge/azimuth/rotation |
| `CompositeTime` | year, month, day, hour, minute, second — each with all 4 error sub-fields |
| `Arrival` | pick ID, phase, time correction, azimuth, distance, takeoff angle (+ errors), time/horizontal-slowness/backazimuth residuals and weights, earth model ID |
| `Pick` | time (+ 4 errors), waveform ID, filter/method/slowness-method IDs, horizontal slowness & backazimuth (+ errors), onset, phase hint, polarity |
| `Magnitude` | mag (+ 4 errors), type, origin ID, method ID, station count, azimuthal gap, evaluation |
| `StationMagnitude` | mag (+ errors), type, amplitude ID, method ID, waveform ID, contributions |
| `StationMagnitudeContribution` | station magnitude ID, residual, weight |
| `Amplitude` | generic amplitude (+ 4 errors), type, category, unit, method ID, period (+ uncertainty), SNR, time window, pick ID, waveform ID, filter ID, scaling time (+ uncertainty), magnitude hint |
| `TimeWindow` | begin, end, reference |
| `FocalMechanism` | triggering origin ID, nodal planes (strike/dip/rake + uncertainties), principal axes (azimuth/plunge/length + uncertainties), preferred plane, azimuthal gap, station polarity count, misfit, station distribution ratio, method ID, waveform IDs |
| `MomentTensor` | derived origin ID, moment magnitude ID, scalar moment (+ uncertainty), all six tensor components (+ uncertainties), variance, variance reduction, double couple, CLVD, ISO, Green's function ID, filter ID, source time function (type/duration/rise/decay), method ID, category, inversion type |
| `DataUsed` | wave type, station count, component count, shortest/longest period |
| `Comment` | text, resource ID, creation info |
| `CreationInfo` | agency ID/URI, author/URI, creation time, version |
| `WaveformStreamID` | network/station/location/channel codes, resource URI |

`Comment`, `CreationInfo`, and `WaveformStreamID` instances are **deduplicated**
into shared tables and referenced by integer index from their parent objects.


AI-assisted (far too complex otherwise!), exercise appropriate caution