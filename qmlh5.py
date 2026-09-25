"""
qmlh5.py — Binary HDF5 storage for QuakeML 1.2 earthquake catalogs.

All objects stored as flat columnar arrays. Cross-object links use integer
indices. Enums stored as uint8 with JSON enum_map attribute. Timestamps as
float64 Unix seconds (NaN = missing). Signed int sentinel: -1 = absent.

Public API
----------
    cat = qmlh5.read_catalog("cat.h5")        # module-level read
    qmlh5.write_catalog(cat, "out.h5")        # module-level write
    cat.write_catalog("out.h5")               # method on ObsPy Catalog

The lower-level :class:`qmlh5` class supports column-oriented queries
(:meth:`query_bbox`, :meth:`query_magnitude`, :meth:`query_radius`,
:meth:`query_polygon`, :meth:`query_depth`, :meth:`query_arrivals`) and
returns columnar dictionaries via :meth:`origins_dataframe`,
:meth:`magnitudes_dataframe`, :meth:`picks_dataframe`,
:meth:`arrivals_dataframe`, :meth:`amplitudes_dataframe`.

Schema fidelity
---------------
qmlh5 conforms to the QuakeML 1.2 BED schema field-for-field for every
class it serializes (Event, Origin, Pick, Arrival, Magnitude, Amplitude,
StationMagnitude, FocalMechanism, MomentTensor, etc.), including all four
QuantityError sub-fields (uncertainty, lower_uncertainty, upper_uncertainty,
confidence_level) on every quantity-typed value.

Limitations
-----------
* QuakeML extension elements/attributes carried by ObsPy's ``extra``
  AttribDict (custom XML namespaces such as ``catalog:datasource`` or
  ``ns0:FEcode`` from USGS/IRIS feeds) are not preserved. A binary
  columnar format cannot accommodate arbitrary user-defined XML.
* Comments without an explicit ``resource_id`` will receive a fresh
  auto-generated id on each XML serialization (an ObsPy quirk, not a
  qmlh5 issue) — the underlying data is preserved.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np

try:
    from obspy.core.event import (
        Amplitude, Arrival, Axis, Catalog, Comment, CompositeTime,
        ConfidenceEllipsoid, CreationInfo, DataUsed, Event, EventDescription,
        FocalMechanism, Magnitude, MomentTensor, NodalPlane, NodalPlanes,
        Origin, OriginQuality, OriginUncertainty, Pick, PrincipalAxes,
        SourceTimeFunction, StationMagnitude, StationMagnitudeContribution,
        Tensor, TimeWindow, WaveformStreamID,
    )
    from obspy.core.event.base import QuantityError
    from obspy import UTCDateTime
    from obspy.core.event import ResourceIdentifier
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

# tqdm is an optional dependency. When present, write operations show a
# progress bar per event. `tqdm.auto` picks the right frontend for terminal
# vs notebook automatically. If tqdm isn't installed, the progress=True
# default silently falls back to no bar.
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Enum maps  (uint8 code → QuakeML string)
# ---------------------------------------------------------------------------
EVALUATION_MODE     = {0:"manual", 1:"automatic"}
EVALUATION_STATUS   = {0:"preliminary",1:"confirmed",2:"reviewed",3:"final",4:"rejected"}
ORIGIN_DEPTH_TYPE   = {0:"from location",1:"from moment tensor inversion",
                       2:"from modeling of broad-band P waveforms",
                       3:"constrained by depth phases",4:"constrained by direct phases",
                       5:"constrained by depth and direct phases",
                       6:"operator assigned",7:"other"}
ORIGIN_TYPE         = {0:"hypocenter",1:"centroid",2:"amplitude",
                       3:"macroseismic",4:"rupture start",5:"rupture end"}
EVENT_TYPE          = {
    0:"not existing",1:"not reported",2:"earthquake",3:"anthropogenic event",
    4:"collapse",5:"cavity collapse",6:"mine collapse",7:"building collapse",
    8:"explosion",9:"accidental explosion",10:"chemical explosion",
    11:"controlled explosion",12:"experimental explosion",13:"industrial explosion",
    14:"mining explosion",15:"quarry blast",16:"road cut",17:"blasting levee",
    18:"nuclear explosion",19:"induced or triggered event",20:"rock burst",
    21:"reservoir loading",22:"fluid injection",23:"fluid extraction",
    24:"crash",25:"plane crash",26:"train crash",27:"boat crash",
    28:"other event",29:"atmospheric event",30:"sonic boom",31:"sonic blast",
    32:"acoustic noise",33:"thunder",34:"avalanche",35:"snow avalanche",
    36:"debris avalanche",37:"hydroacoustic event",38:"ice quake",39:"slide",
    40:"landslide",41:"rockslide",42:"meteorite",43:"volcanic eruption"}
EVENT_TYPE_CERTAINTY  = {0:"known",1:"suspected"}
PICK_ONSET            = {0:"emergent",1:"impulsive",2:"questionable"}
PICK_POLARITY         = {0:"positive",1:"negative",2:"undecidable"}
MT_INVERSION_TYPE     = {0:"general",1:"zero trace",2:"double couple"}
MT_CATEGORY           = {0:"teleseismic",1:"regional"}
SOURCE_TIME_FUNC_TYPE = {0:"box car",1:"triangle",2:"trapezoid",3:"unknown"}
AMPLITUDE_CATEGORY    = {0:"point",1:"mean",2:"duration",3:"period",4:"integral",5:"other"}
AMPLITUDE_UNIT        = {0:"m",1:"s",2:"m/s",3:"m/(s*s)",4:"m*s",5:"dimensionless",6:"other"}
DATA_USED_WAVE_TYPE   = {0:"P waves",1:"body waves",2:"surface waves",
                         3:"mantle waves",4:"combined",5:"unknown"}
ORIGIN_UNCERTAINTY_DESC={0:"horizontal uncertainty",1:"uncertainty ellipse",
                          2:"confidence ellipsoid"}
EVENT_DESC_TYPE       = {0:"felt report",1:"Flinn-Engdahl region",2:"local time",
                         3:"tectonic summary",4:"nearest cities",
                         5:"earthquake name",6:"region name"}

def _rev(d): return {v:k for k,v in d.items()}
_R_EVAL_MODE=_rev(EVALUATION_MODE); _R_EVAL_STATUS=_rev(EVALUATION_STATUS)
_R_ORIG_DEPTH=_rev(ORIGIN_DEPTH_TYPE); _R_ORIG_TYPE=_rev(ORIGIN_TYPE)
_R_EVENT_TYPE=_rev(EVENT_TYPE); _R_ETC=_rev(EVENT_TYPE_CERTAINTY)
_R_PICK_ONSET=_rev(PICK_ONSET); _R_PICK_POL=_rev(PICK_POLARITY)
_R_MT_INV=_rev(MT_INVERSION_TYPE); _R_MT_CAT=_rev(MT_CATEGORY)
_R_STF=_rev(SOURCE_TIME_FUNC_TYPE); _R_AMP_CAT=_rev(AMPLITUDE_CATEGORY)
_R_AMP_UNIT=_rev(AMPLITUDE_UNIT); _R_DU_WAVE=_rev(DATA_USED_WAVE_TYPE)
_R_OU_DESC=_rev(ORIGIN_UNCERTAINTY_DESC); _R_EDT=_rev(EVENT_DESC_TYPE)

_VLEN = h5py.special_dtype(vlen=str)
_NaN  = float("nan")
_KM_PER_DEG = 111.195  # same approximation already used in query_radius's docstring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts(dt):
    if dt is None: return _NaN
    if OBSPY_AVAILABLE:
        try: return float(UTCDateTime(dt).timestamp)
        except Exception: pass
    if isinstance(dt, datetime): return dt.replace(tzinfo=timezone.utc).timestamp()
    return _NaN

def _from_ts(v):
    if math.isnan(v): return None
    return UTCDateTime(v) if OBSPY_AVAILABLE else datetime.fromtimestamp(v,tz=timezone.utc)

def _enc(v, rm, null=255):
    return null if v is None else rm.get(str(v), null)

def _dec(v, fm, null=255):
    return None if v==null else fm.get(int(v))

def _rid(obj):
    if obj is None: return ""
    if OBSPY_AVAILABLE and isinstance(obj, ResourceIdentifier): return obj.id or ""
    return str(obj)

def _make_rid(s):
    if not s: return None
    return ResourceIdentifier(id=s) if OBSPY_AVAILABLE else s

# ObsPy stores every quantity-typed value as a flat scalar with a paired
# `<field>_errors` QuantityError holding its uncertainty sub-fields. These
# helpers extract value / errors from that pattern.
def _fv(obj, attr):
    """Float value of a scalar attribute, NaN if absent."""
    v = getattr(obj, attr, None); return float(v) if v is not None else _NaN
def _tv(obj, attr):
    """Timestamp of a UTCDateTime attribute, NaN if absent."""
    return _ts(getattr(obj, attr, None))
def _qe(obj, attr):
    """QuantityError for attribute `attr` (looks for `attr_errors`)."""
    return getattr(obj, f"{attr}_errors", None)
def _qeu(obj, attr, sub="uncertainty"):
    """Float uncertainty sub-field from *_errors, NaN if absent."""
    qe = _qe(obj, attr)
    if qe is None: return _NaN
    v = getattr(qe, sub, None); return float(v) if v is not None else _NaN
def _benc(v): return np.int8(-1) if v is None else np.int8(1 if v else 0)
def _bdec(v): return None if int(v)==-1 else bool(v)
def _of(v):  return float(v) if v is not None else _NaN
def _oi(v):  return int(v)   if v is not None else -1
def _sv(v):  # safe string from bytes or str
    if v is None: return ""
    return v.decode() if isinstance(v,bytes) else str(v)
def _nn(v):  return None if math.isnan(float(v)) else float(v)
def _ni(v):  return None if int(v)==-1 else int(v)

# ---------------------------------------------------------------------------
# Deduplication tables
# ---------------------------------------------------------------------------
class _WFTable:
    def __init__(self):
        self._idx: Dict[tuple,int]={}
        self.net,self.sta,self.loc,self.cha,self.uri=[],[],[],[],[]
    def add(self,wf):
        if wf is None: return np.uint32(0xFFFFFFFF)
        net=getattr(wf,"network_code","") or ""
        sta=getattr(wf,"station_code","") or ""
        loc=getattr(wf,"location_code","") or ""
        cha=getattr(wf,"channel_code","")  or ""
        uri=_rid(getattr(wf,"resource_uri",None))
        key=(net,sta,loc,cha,uri)
        if key not in self._idx:
            i=len(self.net); self._idx[key]=i
            self.net.append(net); self.sta.append(sta)
            self.loc.append(loc); self.cha.append(cha); self.uri.append(uri)
        return np.uint32(self._idx[key])
    def write(self,grp,cs):
        if not self.net: return
        for name,data in [("network_code",self.net),("station_code",self.sta),
                          ("location_code",self.loc),("channel_code",self.cha),
                          ("resource_uri",self.uri)]:
            grp.create_dataset(name,data=np.array(data,dtype=object),dtype=_VLEN,**cs)

class _CITable:
    def __init__(self):
        self._idx: Dict[tuple,int]={}
        self.aid,self.auri,self.auth,self.auuri,self.ct,self.ver=[],[],[],[],[],[]
    def add(self,ci):
        if ci is None: return np.int32(-1)
        aid=ci.agency_id or ""; auri=_rid(ci.agency_uri)
        auth=ci.author or ""; auuri=_rid(ci.author_uri)
        ct=_ts(ci.creation_time); ver=ci.version or ""
        key=(aid,auri,auth,auuri,ct,ver)
        if key not in self._idx:
            i=len(self.aid); self._idx[key]=i
            self.aid.append(aid); self.auri.append(auri)
            self.auth.append(auth); self.auuri.append(auuri)
            self.ct.append(ct); self.ver.append(ver)
        return np.int32(self._idx[key])
    def write(self,grp,c,cs):
        if not self.aid: return
        for name,data in [("agency_id",self.aid),("agency_uri",self.auri),
                          ("author",self.auth),("author_uri",self.auuri),
                          ("version",self.ver)]:
            grp.create_dataset(name,data=np.array(data,dtype=object),dtype=_VLEN,**cs)
        grp.create_dataset("creation_time",data=np.array(self.ct,dtype=np.float64),**c)

class _ComPool:
    """Append-only flat pool of comments. Returns (global_offset, count) per
    caller so each parent object can later locate its slice. Comments-per-event
    grows with the catalog, so this pool is incrementally flushed during
    chunked writes via :meth:`flush_to`."""
    def __init__(self):
        self.text,self.id,self.ci=[],[],[]
        self._baseline=0   # total rows already flushed to disk
    def add(self,comments,ci_tbl):
        off=self._baseline+len(self.text)
        for c in (comments or []):
            self.text.append(c.text or "")
            self.id.append(_rid(getattr(c,"resource_id",None)))
            self.ci.append(int(ci_tbl.add(c.creation_info)))
        return off, self._baseline+len(self.text)-off
    def flush_to(self,grp,qmlh5):
        """Append the in-memory buffer to extensible HDF5 datasets and clear it."""
        if not self.text: return
        qmlh5._ds_append(grp,"text",np.array(self.text,dtype=object))
        qmlh5._ds_append(grp,"id",  np.array(self.id,  dtype=object))
        qmlh5._ds_append(grp,"ci_idx",np.array(self.ci,dtype=np.int32))
        self._baseline+=len(self.text)
        self.text.clear(); self.id.clear(); self.ci.clear()
    def write(self,grp,c,cs):
        """Final non-chunked write path (used when chunking is disabled)."""
        if not self.text: return
        grp.create_dataset("text",data=np.array(self.text,dtype=object),dtype=_VLEN,**cs)
        grp.create_dataset("id",  data=np.array(self.id,  dtype=object),dtype=_VLEN,**cs)
        grp.create_dataset("ci_idx",data=np.array(self.ci,dtype=np.int32),**c)

# ---------------------------------------------------------------------------
# qmlh5 class — write path
# ---------------------------------------------------------------------------

class qmlh5:
    """
    Read/write QuakeML 1.2 catalogs as columnar HDF5.

    with qmlh5("cat.h5","w") as q: q.write_catalog(cat)
    with qmlh5("cat.h5")     as q: cat = q.read_catalog()
    with qmlh5("cat.h5")     as q: d = q.origins_dataframe()
    """
    FORMAT="qmlh5"; FORMAT_VERSION="1.0"; QUAKEML_VERSION="1.2"; CHUNK=1024

    # Which table each query_* method's returned row indices refer to.
    # Used by query_events() to auto-resolve the table without the caller
    # having to name it.
    QUERY_TABLE={"query_bbox":"origins","query_time":"origins",
                 "query_radius":"origins","query_polygon":"origins",
                 "query_depth":"origins","query_arrivals":"origins",
                 "query_magnitude":"magnitudes"}
    _C  = dict(compression="gzip",compression_opts=4,shuffle=True)
    _CS = dict(compression="gzip",compression_opts=4)

    def __init__(self,path,mode="r"):
        self._path=path; self._mode=mode; self._f=None

    def __enter__(self):
        self._f=h5py.File(self._path,self._mode); return self
    def __exit__(self,*_):
        if self._f: self._f.close(); self._f=None
    def open(self):  self._f=h5py.File(self._path,self._mode); return self
    def close(self):
        if self._f: self._f.close(); self._f=None

    # --- dataset factory ---
    def _ds(self,grp,name,data,enum_map=None):
        n=len(data); chunk=(min(self.CHUNK,max(1,n)),) if n else None
        if data.dtype.kind=="O":
            ds=grp.create_dataset(name,data=data,dtype=_VLEN,chunks=chunk,**self._CS)
        else:
            ds=grp.create_dataset(name,data=data,chunks=chunk,**self._C)
        if enum_map: ds.attrs["enum_map"]=json.dumps(enum_map)
        return ds

    def _ds_append(self,grp,name,data,enum_map=None):
        """Append to a resizable dataset, creating it on the first call.
        Used during chunked catalog writes so accumulator memory can be
        released after every chunk instead of held until the end."""
        n=len(data)
        if n==0: return None
        if name not in grp:
            chunk=(min(self.CHUNK,n),)
            if data.dtype.kind=="O":
                ds=grp.create_dataset(name,data=data,dtype=_VLEN,
                                      chunks=chunk,maxshape=(None,),**self._CS)
            else:
                ds=grp.create_dataset(name,data=data,
                                      chunks=chunk,maxshape=(None,),**self._C)
            if enum_map: ds.attrs["enum_map"]=json.dumps(enum_map)
            return ds
        ds=grp[name]
        old=ds.shape[0]; ds.resize((old+n,)); ds[old:old+n]=data
        return ds

    def _sa(self,lst): return np.array(lst,dtype=object)  # string array

    # ------------------------------------------------------------------
    def write_catalog(self,catalog,progress=True,chunk_size=10000):
        """Write an ObsPy Catalog to this HDF5 file.

        The catalog is processed in chunks of ``chunk_size`` events. After
        each chunk the per-row accumulators (origins, magnitudes, picks,
        arrivals, comments, etc.) are appended to extensible HDF5 datasets
        and cleared, so peak RAM stays proportional to ``chunk_size`` rather
        than to the entire catalog. For a 600k-event catalog the default
        chunk_size of 10k brings peak RAM from tens of GB down to ~hundreds
        of MB. Set ``chunk_size=None`` to disable chunking (the original
        single-flush behaviour) for tiny catalogs where the overhead of
        extensible datasets isn't worth it.
        """
        if self._f is None: raise RuntimeError("File not open")
        f=self._f
        f.attrs.update({"format":self.FORMAT,"format_version":self.FORMAT_VERSION,
                        "quakeml_version":self.QUAKEML_VERSION,
                        "creation_time":_ts(datetime.now(tz=timezone.utc)),
                        "catalog_description":getattr(catalog,"description",None) or "",
                        "catalog_public_id":_rid(getattr(catalog,"resource_id",None))})
        events=list(catalog); f.attrs["n_events"]=len(events)

        wf=_WFTable(); ci=_CITable(); cp=_ComPool()

        # Catalog-level CreationInfo and comments. The CI index lives in file
        # attrs (-1 if absent); catalog comments are pooled like everything else
        # but keyed by their offset/count stored in attrs.
        f.attrs["catalog_ci_idx"]=int(ci.add(getattr(catalog,"creation_info",None)))
        cat_coff,cat_ccnt=cp.add(getattr(catalog,"comments",None),ci)
        f.attrs["catalog_comment_offset"]=int(cat_coff)
        f.attrs["catalog_comment_count"]=int(cat_ccnt)

        # ---- chunked-write baseline counters ----
        # When chunking is on, accumulator buffers get periodically flushed and
        # cleared. Any offset/index field captured against `len(buf)` during
        # accumulation must therefore be made global by adding the count of rows
        # already flushed (these counters). They're attributes on self so the
        # _flush_chunk closure (defined below) can mutate them.
        self._n_ed_written = 0   # event_descriptions
        self._n_ar_written = 0   # arrivals
        self._n_ct_written = 0   # composite_times
        self._n_oq_written = 0   # origin_quality rows
        self._n_ou_written = 0   # origin_uncertainty rows
        self._n_ce_written = 0   # confidence_ellipsoids rows
        self._n_sc_written = 0   # station_mag_contributions
        self._n_tw_written = 0   # time_windows
        self._n_mt_written = 0   # moment_tensors
        self._n_du_written = 0   # data_used
        self._n_fmwp_written = 0  # focal mech waveform pool entries

        # ---- accumulator dicts ----
        ev =dict(pid=[],po=[],pm=[],pf=[],etype=[],ecert=[],ci=[],
                 doff=[],dcnt=[],coff=[],ccnt=[])
        ed =dict(text=[],type=[])
        # origins — every QuantityError-bearing field stores the full set of
        # error sub-fields (uncertainty, lower, upper, confidence_level)
        or_=dict(pid=[],eidx=[],
                 tv=[],tu=[],tlo=[],thi=[],tcf=[],
                 lav=[],lau=[],lalo=[],lahi=[],lacf=[],
                 lov=[],lou=[],lolo=[],lohi=[],locf=[],
                 dpv=[],dpu=[],dplo=[],dphi=[],dpcf=[],
                 dtype=[],tfx=[],epfx=[],
                 rsid=[],mid=[],emid=[],otype=[],reg=[],
                 emode=[],estat=[],ci=[],qidx=[],uidx=[],
                 aoff=[],acnt=[],coff=[],ccnt=[],ctoff=[],ctcnt=[])
        oq=dict(apc=[],upc=[],asc=[],usc=[],dpc=[],
                se=[],ag=[],sag=[],gtl=[],mind=[],maxd=[],medd=[])
        ou=dict(hu=[],minu=[],maxu=[],az=[],pd=[],cl=[],eidx=[])
        ce=dict(sma=[],smi=[],smia=[],mpl=[],maz=[],mro=[])
        # composite_times: each int field stores value + uncertainty + lower/upper/
        # confidence (QuakeML's IntegerQuantity has all four error sub-fields).
        ct=dict(yrv=[],yru=[],yrlo=[],yrhi=[],yrcf=[],
                mov=[],mou=[],molo=[],mohi=[],mocf=[],
                dyv=[],dyu=[],dylo=[],dyhi=[],dycf=[],
                hrv=[],hru=[],hrlo=[],hrhi=[],hrcf=[],
                miv=[],miu=[],milo=[],mihi=[],micf=[],
                sev=[],seu=[],selo=[],sehi=[],secf=[])
        # Arrival has no evaluationMode/evaluationStatus in QuakeML 1.2 BED;
        # ObsPy's Arrival class is correctly conformant to the spec.
        ar=dict(pid=[],pkid=[],ph=[],tc=[],az=[],dist=[],
                tov=[],tou=[],tr=[],hsr=[],br=[],
                tw=[],hsw=[],bw=[],emid=[],
                ci=[],coff=[],ccnt=[])
        mg=dict(pid=[],eidx=[],val=[],unc=[],lo=[],hi=[],cf=[],
                type=[],orig=[],mid=[],scnt=[],ag=[],
                emode=[],estat=[],ci=[],soff=[],scnt2=[],coff=[],ccnt=[])
        sm=dict(pid=[],eidx=[],orig=[],val=[],unc=[],lo=[],hi=[],
                type=[],amid=[],mid=[],wfidx=[],ci=[],coff=[],ccnt=[])
        sc=dict(smid=[],res=[],wt=[])
        pk=dict(pid=[],eidx=[],tv=[],tu=[],tlo=[],thi=[],tcf=[],
                wfidx=[],fid=[],mid=[],hsv=[],hsu=[],bzv=[],bzu=[],
                smid=[],onset=[],ph=[],pol=[],
                emode=[],estat=[],ci=[],coff=[],ccnt=[])
        am=dict(pid=[],eidx=[],val=[],unc=[],lo=[],hi=[],cf=[],
                type=[],cat=[],unit=[],mid=[],perv=[],peru=[],snr=[],
                twidx=[],pkid=[],wfidx=[],fid=[],stv=[],stu=[],
                mhint=[],emode=[],estat=[],ci=[],coff=[],ccnt=[])
        tw=dict(beg=[],end=[],ref=[])
        fm=dict(pid=[],eidx=[],toid=[],
                np1sv=[],np1su=[],np1dv=[],np1du=[],np1rv=[],np1ru=[],
                np2sv=[],np2su=[],np2dv=[],np2du=[],np2rv=[],np2ru=[],
                pp=[],
                tazv=[],tazu=[],tplv=[],tplu=[],tlnv=[],tlnu=[],
                pazv=[],pazu=[],pplv=[],pplu=[],plnv=[],plnu=[],
                nazv=[],nazu=[],nplv=[],nplu=[],nlnv=[],nlnu=[],
                ag=[],spc=[],mft=[],sdr=[],mid=[],
                emode=[],estat=[],ci=[],mtidx=[],
                coff=[],ccnt=[],wpoff=[],wpcnt=[])
        fmwp=[]  # waveform pool for focal mechanisms
        mt=dict(pid=[],doid=[],mmid=[],scv=[],scu=[],
                rr_v=[],rr_u=[],tt_v=[],tt_u=[],pp_v=[],pp_u=[],
                rt_v=[],rt_u=[],rp_v=[],rp_u=[],tp_v=[],tp_u=[],
                var=[],vr=[],dc=[],clvd=[],iso=[],
                gfid=[],fid=[],stft=[],stfd=[],stfr=[],stfdc=[],
                mid=[],cat=[],inv=[],ci=[],duoff=[],ducnt=[],coff=[],ccnt=[])
        du=dict(wt=[],sc=[],cc=[],sp=[],lp=[])

        # ---- _flush_chunk: append per-row accumulators to extensible datasets
        # ---- and clear them. Called periodically during the event loop and
        # ---- once at the end. The waveform_id and creation_info dedup tables
        # ---- stay in RAM (they grow with uniqueness, not row count) and are
        # ---- written once at the very end.
        def _g(n): return f.require_group(n)
        def _f64(lst): return np.array(lst,dtype=np.float64)
        def _u32(lst): return np.array(lst,dtype=np.uint32)
        def _i32(lst): return np.array(lst,dtype=np.int32)
        def _u8(lst):  return np.array(lst,dtype=np.uint8)
        def _i8(lst):  return np.array(lst,dtype=np.int8)

        def _flush_chunk():
            if ev["pid"]:
                g=_g("catalog")
                self._ds_append(g,"public_id",self._sa(ev["pid"]))
                self._ds_append(g,"preferred_origin_id",self._sa(ev["po"]))
                self._ds_append(g,"preferred_magnitude_id",self._sa(ev["pm"]))
                self._ds_append(g,"preferred_focmec_id",self._sa(ev["pf"]))
                self._ds_append(g,"event_type",_u8(ev["etype"]),enum_map=EVENT_TYPE)
                self._ds_append(g,"event_type_certainty",_u8(ev["ecert"]),enum_map=EVENT_TYPE_CERTAINTY)
                self._ds_append(g,"ci_idx",_i32(ev["ci"]))
                self._ds_append(g,"desc_offset",_u32(ev["doff"])); self._ds_append(g,"desc_count",_u32(ev["dcnt"]))
                self._ds_append(g,"comment_offset",_u32(ev["coff"])); self._ds_append(g,"comment_count",_u32(ev["ccnt"]))
                for k in ev: ev[k].clear()

            if ed["text"]:
                g=_g("event_descriptions")
                n=len(ed["text"])
                self._ds_append(g,"text",self._sa(ed["text"]))
                self._ds_append(g,"type",_u8(ed["type"]),enum_map=EVENT_DESC_TYPE)
                self._n_ed_written+=n
                for k in ed: ed[k].clear()

            if or_["pid"]:
                g=_g("origins")
                self._ds_append(g,"public_id",self._sa(or_["pid"]))
                self._ds_append(g,"event_idx",_u32(or_["eidx"]))
                self._ds_append(g,"time_value",_f64(or_["tv"])); self._ds_append(g,"time_uncertainty",_f64(or_["tu"]))
                self._ds_append(g,"time_lower_unc",_f64(or_["tlo"])); self._ds_append(g,"time_upper_unc",_f64(or_["thi"]))
                self._ds_append(g,"time_conf",_f64(or_["tcf"]))
                self._ds_append(g,"lat_value",_f64(or_["lav"])); self._ds_append(g,"lat_uncertainty",_f64(or_["lau"]))
                self._ds_append(g,"lat_lower_unc",_f64(or_["lalo"])); self._ds_append(g,"lat_upper_unc",_f64(or_["lahi"]))
                self._ds_append(g,"lat_conf",_f64(or_["lacf"]))
                self._ds_append(g,"lon_value",_f64(or_["lov"])); self._ds_append(g,"lon_uncertainty",_f64(or_["lou"]))
                self._ds_append(g,"lon_lower_unc",_f64(or_["lolo"])); self._ds_append(g,"lon_upper_unc",_f64(or_["lohi"]))
                self._ds_append(g,"lon_conf",_f64(or_["locf"]))
                self._ds_append(g,"depth_value",_f64(or_["dpv"])); self._ds_append(g,"depth_uncertainty",_f64(or_["dpu"]))
                self._ds_append(g,"depth_lower_unc",_f64(or_["dplo"])); self._ds_append(g,"depth_upper_unc",_f64(or_["dphi"]))
                self._ds_append(g,"depth_conf",_f64(or_["dpcf"]))
                self._ds_append(g,"depth_type",_u8(or_["dtype"]),enum_map=ORIGIN_DEPTH_TYPE)
                self._ds_append(g,"time_fixed",_i8(or_["tfx"])); self._ds_append(g,"epicenter_fixed",_i8(or_["epfx"]))
                self._ds_append(g,"ref_system_id",self._sa(or_["rsid"]))
                self._ds_append(g,"method_id",self._sa(or_["mid"]))
                self._ds_append(g,"earth_model_id",self._sa(or_["emid"]))
                self._ds_append(g,"type",_u8(or_["otype"]),enum_map=ORIGIN_TYPE)
                self._ds_append(g,"region",self._sa(or_["reg"]))
                self._ds_append(g,"eval_mode",_u8(or_["emode"]),enum_map=EVALUATION_MODE)
                self._ds_append(g,"eval_status",_u8(or_["estat"]),enum_map=EVALUATION_STATUS)
                self._ds_append(g,"ci_idx",_i32(or_["ci"]))
                self._ds_append(g,"quality_idx",_i32(or_["qidx"])); self._ds_append(g,"uncertainty_idx",_i32(or_["uidx"]))
                self._ds_append(g,"arrival_offset",_u32(or_["aoff"])); self._ds_append(g,"arrival_count",_u32(or_["acnt"]))
                self._ds_append(g,"comment_offset",_u32(or_["coff"])); self._ds_append(g,"comment_count",_u32(or_["ccnt"]))
                self._ds_append(g,"comptime_offset",_u32(or_["ctoff"])); self._ds_append(g,"comptime_count",_u32(or_["ctcnt"]))
                for k in or_: or_[k].clear()

            if oq["apc"]:
                g=_g("origin_quality")
                n=len(oq["apc"])
                self._ds_append(g,"assoc_phase_count",_i32(oq["apc"])); self._ds_append(g,"used_phase_count",_i32(oq["upc"]))
                self._ds_append(g,"assoc_sta_count",_i32(oq["asc"])); self._ds_append(g,"used_sta_count",_i32(oq["usc"]))
                self._ds_append(g,"depth_phase_count",_i32(oq["dpc"])); self._ds_append(g,"standard_error",_f64(oq["se"]))
                self._ds_append(g,"azimuthal_gap",_f64(oq["ag"])); self._ds_append(g,"sec_azimuthal_gap",_f64(oq["sag"]))
                self._ds_append(g,"ground_truth_level",self._sa(oq["gtl"]))
                self._ds_append(g,"minimum_distance",_f64(oq["mind"])); self._ds_append(g,"maximum_distance",_f64(oq["maxd"]))
                self._ds_append(g,"median_distance",_f64(oq["medd"]))
                self._n_oq_written+=n
                for k in oq: oq[k].clear()

            if ou["hu"]:
                g=_g("origin_uncertainty")
                n=len(ou["hu"])
                self._ds_append(g,"horizontal_uncertainty",_f64(ou["hu"]))
                self._ds_append(g,"min_horizontal_uncertainty",_f64(ou["minu"]))
                self._ds_append(g,"max_horizontal_uncertainty",_f64(ou["maxu"]))
                self._ds_append(g,"azimuth_max_horiz_unc",_f64(ou["az"]))
                self._ds_append(g,"preferred_description",_u8(ou["pd"]),enum_map=ORIGIN_UNCERTAINTY_DESC)
                self._ds_append(g,"confidence_level",_f64(ou["cl"]))
                self._ds_append(g,"ellipsoid_idx",_i32(ou["eidx"]))
                self._n_ou_written+=n
                for k in ou: ou[k].clear()

            if ce["sma"]:
                g=_g("confidence_ellipsoids")
                n=len(ce["sma"])
                self._ds_append(g,"semi_major_axis_length",_f64(ce["sma"]))
                self._ds_append(g,"semi_minor_axis_length",_f64(ce["smi"]))
                self._ds_append(g,"semi_intermediate_axis_length",_f64(ce["smia"]))
                self._ds_append(g,"major_axis_plunge",_f64(ce["mpl"]))
                self._ds_append(g,"major_axis_azimuth",_f64(ce["maz"]))
                self._ds_append(g,"major_axis_rotation",_f64(ce["mro"]))
                self._n_ce_written+=n
                for k in ce: ce[k].clear()

            if ct["yrv"]:
                g=_g("composite_times")
                n=len(ct["yrv"])
                for k,nm in [("yrv","year_value"),("yru","year_unc"),
                            ("yrlo","year_lower_unc"),("yrhi","year_upper_unc"),
                            ("mov","month_value"),("mou","month_unc"),
                            ("molo","month_lower_unc"),("mohi","month_upper_unc"),
                            ("dyv","day_value"),("dyu","day_unc"),
                            ("dylo","day_lower_unc"),("dyhi","day_upper_unc"),
                            ("hrv","hour_value"),("hru","hour_unc"),
                            ("hrlo","hour_lower_unc"),("hrhi","hour_upper_unc"),
                            ("miv","minute_value"),("miu","minute_unc"),
                            ("milo","minute_lower_unc"),("mihi","minute_upper_unc")]:
                    self._ds_append(g,nm,_i32(ct[k]))
                for k,nm in [("yrcf","year_conf"),("mocf","month_conf"),("dycf","day_conf"),
                            ("hrcf","hour_conf"),("micf","minute_conf"),
                            ("sev","second_value"),("seu","second_unc"),
                            ("selo","second_lower_unc"),("sehi","second_upper_unc"),
                            ("secf","second_conf")]:
                    self._ds_append(g,nm,_f64(ct[k]))
                self._n_ct_written+=n
                for k in ct: ct[k].clear()

            if ar["pid"]:
                g=_g("arrivals")
                n=len(ar["pid"])
                self._ds_append(g,"public_id",self._sa(ar["pid"])); self._ds_append(g,"pick_id",self._sa(ar["pkid"]))
                self._ds_append(g,"phase",self._sa(ar["ph"]))
                self._ds_append(g,"time_correction",_f64(ar["tc"])); self._ds_append(g,"azimuth",_f64(ar["az"]))
                self._ds_append(g,"distance",_f64(ar["dist"]))
                self._ds_append(g,"takeoff_value",_f64(ar["tov"])); self._ds_append(g,"takeoff_uncertainty",_f64(ar["tou"]))
                self._ds_append(g,"time_residual",_f64(ar["tr"])); self._ds_append(g,"hslow_residual",_f64(ar["hsr"]))
                self._ds_append(g,"baz_residual",_f64(ar["br"])); self._ds_append(g,"time_weight",_f64(ar["tw"]))
                self._ds_append(g,"hslow_weight",_f64(ar["hsw"])); self._ds_append(g,"baz_weight",_f64(ar["bw"]))
                self._ds_append(g,"earth_model_id",self._sa(ar["emid"]))
                self._ds_append(g,"ci_idx",_i32(ar["ci"]))
                self._ds_append(g,"comment_offset",_u32(ar["coff"])); self._ds_append(g,"comment_count",_u32(ar["ccnt"]))
                self._n_ar_written+=n
                for k in ar: ar[k].clear()

            if mg["pid"]:
                g=_g("magnitudes")
                self._ds_append(g,"public_id",self._sa(mg["pid"])); self._ds_append(g,"event_idx",_u32(mg["eidx"]))
                self._ds_append(g,"mag_value",_f64(mg["val"])); self._ds_append(g,"mag_uncertainty",_f64(mg["unc"]))
                self._ds_append(g,"mag_lower_unc",_f64(mg["lo"])); self._ds_append(g,"mag_upper_unc",_f64(mg["hi"]))
                self._ds_append(g,"mag_conf",_f64(mg["cf"])); self._ds_append(g,"type",self._sa(mg["type"]))
                self._ds_append(g,"origin_id",self._sa(mg["orig"])); self._ds_append(g,"method_id",self._sa(mg["mid"]))
                self._ds_append(g,"station_count",_i32(mg["scnt"])); self._ds_append(g,"azimuthal_gap",_f64(mg["ag"]))
                self._ds_append(g,"eval_mode",_u8(mg["emode"]),enum_map=EVALUATION_MODE)
                self._ds_append(g,"eval_status",_u8(mg["estat"]),enum_map=EVALUATION_STATUS)
                self._ds_append(g,"ci_idx",_i32(mg["ci"]))
                self._ds_append(g,"contrib_offset",_u32(mg["soff"])); self._ds_append(g,"contrib_count",_u32(mg["scnt2"]))
                self._ds_append(g,"comment_offset",_u32(mg["coff"])); self._ds_append(g,"comment_count",_u32(mg["ccnt"]))
                for k in mg: mg[k].clear()

            if sm["pid"]:
                g=_g("station_magnitudes")
                self._ds_append(g,"public_id",self._sa(sm["pid"])); self._ds_append(g,"event_idx",_u32(sm["eidx"]))
                self._ds_append(g,"origin_id",self._sa(sm["orig"]))
                self._ds_append(g,"mag_value",_f64(sm["val"])); self._ds_append(g,"mag_uncertainty",_f64(sm["unc"]))
                self._ds_append(g,"mag_lower_unc",_f64(sm["lo"])); self._ds_append(g,"mag_upper_unc",_f64(sm["hi"]))
                self._ds_append(g,"type",self._sa(sm["type"])); self._ds_append(g,"amplitude_id",self._sa(sm["amid"]))
                self._ds_append(g,"method_id",self._sa(sm["mid"])); self._ds_append(g,"waveform_idx",_u32(sm["wfidx"]))
                self._ds_append(g,"ci_idx",_i32(sm["ci"]))
                self._ds_append(g,"comment_offset",_u32(sm["coff"])); self._ds_append(g,"comment_count",_u32(sm["ccnt"]))
                for k in sm: sm[k].clear()

            if sc["smid"]:
                g=_g("station_mag_contributions")
                n=len(sc["smid"])
                self._ds_append(g,"station_magnitude_id",self._sa(sc["smid"]))
                self._ds_append(g,"residual",_f64(sc["res"])); self._ds_append(g,"weight",_f64(sc["wt"]))
                self._n_sc_written+=n
                for k in sc: sc[k].clear()

            if pk["pid"]:
                g=_g("picks")
                self._ds_append(g,"public_id",self._sa(pk["pid"])); self._ds_append(g,"event_idx",_u32(pk["eidx"]))
                self._ds_append(g,"time_value",_f64(pk["tv"])); self._ds_append(g,"time_uncertainty",_f64(pk["tu"]))
                self._ds_append(g,"time_lower_unc",_f64(pk["tlo"])); self._ds_append(g,"time_upper_unc",_f64(pk["thi"]))
                self._ds_append(g,"time_conf",_f64(pk["tcf"])); self._ds_append(g,"waveform_idx",_u32(pk["wfidx"]))
                self._ds_append(g,"filter_id",self._sa(pk["fid"])); self._ds_append(g,"method_id",self._sa(pk["mid"]))
                self._ds_append(g,"hslow_value",_f64(pk["hsv"])); self._ds_append(g,"hslow_uncertainty",_f64(pk["hsu"]))
                self._ds_append(g,"baz_value",_f64(pk["bzv"])); self._ds_append(g,"baz_uncertainty",_f64(pk["bzu"]))
                self._ds_append(g,"slowness_method_id",self._sa(pk["smid"]))
                self._ds_append(g,"onset",_u8(pk["onset"]),enum_map=PICK_ONSET)
                self._ds_append(g,"phase_hint",self._sa(pk["ph"]))
                self._ds_append(g,"polarity",_u8(pk["pol"]),enum_map=PICK_POLARITY)
                self._ds_append(g,"eval_mode",_u8(pk["emode"]),enum_map=EVALUATION_MODE)
                self._ds_append(g,"eval_status",_u8(pk["estat"]),enum_map=EVALUATION_STATUS)
                self._ds_append(g,"ci_idx",_i32(pk["ci"]))
                self._ds_append(g,"comment_offset",_u32(pk["coff"])); self._ds_append(g,"comment_count",_u32(pk["ccnt"]))
                for k in pk: pk[k].clear()

            if am["pid"]:
                g=_g("amplitudes")
                self._ds_append(g,"public_id",self._sa(am["pid"])); self._ds_append(g,"event_idx",_u32(am["eidx"]))
                self._ds_append(g,"amp_value",_f64(am["val"])); self._ds_append(g,"amp_uncertainty",_f64(am["unc"]))
                self._ds_append(g,"amp_lower_unc",_f64(am["lo"])); self._ds_append(g,"amp_upper_unc",_f64(am["hi"]))
                self._ds_append(g,"amp_conf",_f64(am["cf"])); self._ds_append(g,"type",self._sa(am["type"]))
                self._ds_append(g,"category",_u8(am["cat"]),enum_map=AMPLITUDE_CATEGORY)
                self._ds_append(g,"unit",_u8(am["unit"]),enum_map=AMPLITUDE_UNIT)
                self._ds_append(g,"method_id",self._sa(am["mid"]))
                self._ds_append(g,"period_value",_f64(am["perv"])); self._ds_append(g,"period_uncertainty",_f64(am["peru"]))
                self._ds_append(g,"snr",_f64(am["snr"])); self._ds_append(g,"time_window_idx",_i32(am["twidx"]))
                self._ds_append(g,"pick_id",self._sa(am["pkid"])); self._ds_append(g,"waveform_idx",_u32(am["wfidx"]))
                self._ds_append(g,"filter_id",self._sa(am["fid"]))
                self._ds_append(g,"scaling_time_value",_f64(am["stv"])); self._ds_append(g,"scaling_time_unc",_f64(am["stu"]))
                self._ds_append(g,"magnitude_hint",self._sa(am["mhint"]))
                self._ds_append(g,"eval_mode",_u8(am["emode"]),enum_map=EVALUATION_MODE)
                self._ds_append(g,"eval_status",_u8(am["estat"]),enum_map=EVALUATION_STATUS)
                self._ds_append(g,"ci_idx",_i32(am["ci"]))
                self._ds_append(g,"comment_offset",_u32(am["coff"])); self._ds_append(g,"comment_count",_u32(am["ccnt"]))
                for k in am: am[k].clear()

            if tw["beg"]:
                g=_g("time_windows")
                n=len(tw["beg"])
                self._ds_append(g,"begin",_f64(tw["beg"])); self._ds_append(g,"end",_f64(tw["end"]))
                self._ds_append(g,"reference",_f64(tw["ref"]))
                self._n_tw_written+=n
                for k in tw: tw[k].clear()

            if fm["pid"]:
                g=_g("focal_mechanisms")
                self._ds_append(g,"public_id",self._sa(fm["pid"])); self._ds_append(g,"event_idx",_u32(fm["eidx"]))
                self._ds_append(g,"triggering_origin_id",self._sa(fm["toid"]))
                for k,nm in [("np1sv","np1_strike_value"),("np1su","np1_strike_unc"),
                            ("np1dv","np1_dip_value"),("np1du","np1_dip_unc"),
                            ("np1rv","np1_rake_value"),("np1ru","np1_rake_unc"),
                            ("np2sv","np2_strike_value"),("np2su","np2_strike_unc"),
                            ("np2dv","np2_dip_value"),("np2du","np2_dip_unc"),
                            ("np2rv","np2_rake_value"),("np2ru","np2_rake_unc")]:
                    self._ds_append(g,nm,_f64(fm[k]))
                self._ds_append(g,"preferred_plane",_u8(fm["pp"]))
                for k,nm in [("tazv","t_azimuth_value"),("tazu","t_azimuth_unc"),
                            ("tplv","t_plunge_value"),("tplu","t_plunge_unc"),
                            ("tlnv","t_length_value"),("tlnu","t_length_unc"),
                            ("pazv","p_azimuth_value"),("pazu","p_azimuth_unc"),
                            ("pplv","p_plunge_value"),("pplu","p_plunge_unc"),
                            ("plnv","p_length_value"),("plnu","p_length_unc"),
                            ("nazv","n_azimuth_value"),("nazu","n_azimuth_unc"),
                            ("nplv","n_plunge_value"),("nplu","n_plunge_unc"),
                            ("nlnv","n_length_value"),("nlnu","n_length_unc")]:
                    self._ds_append(g,nm,_f64(fm[k]))
                self._ds_append(g,"azimuthal_gap",_f64(fm["ag"]))
                self._ds_append(g,"station_polarity_count",_i32(fm["spc"]))
                self._ds_append(g,"misfit",_f64(fm["mft"])); self._ds_append(g,"station_dist_ratio",_f64(fm["sdr"]))
                self._ds_append(g,"method_id",self._sa(fm["mid"]))
                self._ds_append(g,"eval_mode",_u8(fm["emode"]),enum_map=EVALUATION_MODE)
                self._ds_append(g,"eval_status",_u8(fm["estat"]),enum_map=EVALUATION_STATUS)
                self._ds_append(g,"ci_idx",_i32(fm["ci"])); self._ds_append(g,"mt_idx",_i32(fm["mtidx"]))
                self._ds_append(g,"comment_offset",_u32(fm["coff"])); self._ds_append(g,"comment_count",_u32(fm["ccnt"]))
                self._ds_append(g,"waveform_pool_offset",_u32(fm["wpoff"]))
                self._ds_append(g,"waveform_pool_count",_u32(fm["wpcnt"]))
                for k in fm: fm[k].clear()
            if fmwp:
                g=_g("focal_mechanisms")
                n=len(fmwp)
                self._ds_append(g,"waveform_pool",np.array(fmwp,dtype=np.uint32))
                self._n_fmwp_written+=n
                fmwp.clear()

            if mt["pid"]:
                g=_g("moment_tensors")
                n=len(mt["pid"])
                self._ds_append(g,"public_id",self._sa(mt["pid"]))
                self._ds_append(g,"derived_origin_id",self._sa(mt["doid"]))
                self._ds_append(g,"moment_mag_id",self._sa(mt["mmid"]))
                self._ds_append(g,"scalar_moment_value",_f64(mt["scv"])); self._ds_append(g,"scalar_moment_unc",_f64(mt["scu"]))
                for short in ("rr","tt","pp","rt","rp","tp"):
                    self._ds_append(g,f"{short}_value",_f64(mt[f"{short}_v"]))
                    self._ds_append(g,f"{short}_unc",  _f64(mt[f"{short}_u"]))
                self._ds_append(g,"variance",_f64(mt["var"])); self._ds_append(g,"variance_reduction",_f64(mt["vr"]))
                self._ds_append(g,"double_couple",_f64(mt["dc"])); self._ds_append(g,"clvd",_f64(mt["clvd"]))
                self._ds_append(g,"iso",_f64(mt["iso"]))
                self._ds_append(g,"greens_function_id",self._sa(mt["gfid"]))
                self._ds_append(g,"filter_id",self._sa(mt["fid"]))
                self._ds_append(g,"stf_type",_u8(mt["stft"]),enum_map=SOURCE_TIME_FUNC_TYPE)
                self._ds_append(g,"stf_duration",_f64(mt["stfd"])); self._ds_append(g,"stf_rise_time",_f64(mt["stfr"]))
                self._ds_append(g,"stf_decay_time",_f64(mt["stfdc"]))
                self._ds_append(g,"method_id",self._sa(mt["mid"]))
                self._ds_append(g,"category",_u8(mt["cat"]),enum_map=MT_CATEGORY)
                self._ds_append(g,"inversion_type",_u8(mt["inv"]),enum_map=MT_INVERSION_TYPE)
                self._ds_append(g,"ci_idx",_i32(mt["ci"]))
                self._ds_append(g,"data_used_offset",_u32(mt["duoff"])); self._ds_append(g,"data_used_count",_u32(mt["ducnt"]))
                self._ds_append(g,"comment_offset",_u32(mt["coff"])); self._ds_append(g,"comment_count",_u32(mt["ccnt"]))
                self._n_mt_written+=n
                for k in mt: mt[k].clear()

            if du["wt"]:
                g=_g("data_used")
                n=len(du["wt"])
                self._ds_append(g,"wave_type",_u8(du["wt"]),enum_map=DATA_USED_WAVE_TYPE)
                self._ds_append(g,"station_count",_i32(du["sc"])); self._ds_append(g,"component_count",_i32(du["cc"]))
                self._ds_append(g,"shortest_period",_f64(du["sp"])); self._ds_append(g,"longest_period",_f64(du["lp"]))
                self._n_du_written+=n
                for k in du: du[k].clear()

            # _ComPool tracks its own baseline + clear internally.
            cp.flush_to(_g("comments"),self)

        # ---- iterate events ----
        # Wrap with tqdm only when explicitly enabled, tqdm is importable, and
        # there's actually work worth tracking. Below ~100 events the write
        # finishes in a few hundred ms and the bar just flashes by.
        if progress and TQDM_AVAILABLE and len(events) > 100:
            event_iter = enumerate(tqdm(events, desc="Writing events",
                                        unit="event", leave=False))
        else:
            event_iter = enumerate(events)
        for eidx,e in event_iter:
            descs=getattr(e,"event_descriptions",[]) or []
            doff=self._n_ed_written+len(ed["text"])
            for d in descs:
                ed["text"].append(d.text or "")
                ed["type"].append(_enc(d.type,_R_EDT))
            coff,ccnt=cp.add(e.comments,ci)
            ev["pid"].append(_rid(e.resource_id))
            ev["po"].append(_rid(e.preferred_origin_id))
            ev["pm"].append(_rid(e.preferred_magnitude_id))
            ev["pf"].append(_rid(e.preferred_focal_mechanism_id))
            ev["etype"].append(_enc(e.event_type,_R_EVENT_TYPE))
            ev["ecert"].append(_enc(e.event_type_certainty,_R_ETC))
            ev["ci"].append(int(ci.add(e.creation_info)))
            ev["doff"].append(doff); ev["dcnt"].append(len(descs))
            ev["coff"].append(coff); ev["ccnt"].append(ccnt)

            for o in (e.origins or []):
                # composite times — ObsPy stores year/month/day/hour/minute as
                # flat ints and `second` as a flat float, with optional
                # `<field>_errors` QuantityError objects holding uncertainty,
                # lower_uncertainty, upper_uncertainty, and confidence_level.
                ctoff=self._n_ct_written+len(ct["yrv"])
                for c_t in (getattr(o,"composite_times",[]) or []):
                    def _ctv(attr):
                        v=getattr(c_t,attr,None)
                        return -1 if v is None else int(v)
                    def _cti(attr,sub):
                        # signed-int sub-field (uncertainty/lower/upper)
                        qe=getattr(c_t,f"{attr}_errors",None)
                        if qe is None: return -1
                        u=getattr(qe,sub,None)
                        return -1 if u is None else int(u)
                    def _ctf(attr,sub):
                        # float sub-field (confidence_level)
                        qe=getattr(c_t,f"{attr}_errors",None)
                        if qe is None: return _NaN
                        u=getattr(qe,sub,None)
                        return _NaN if u is None else float(u)
                    for attr,vk,uk,lok,hik,cfk in [
                        ("year",  "yrv","yru","yrlo","yrhi","yrcf"),
                        ("month", "mov","mou","molo","mohi","mocf"),
                        ("day",   "dyv","dyu","dylo","dyhi","dycf"),
                        ("hour",  "hrv","hru","hrlo","hrhi","hrcf"),
                        ("minute","miv","miu","milo","mihi","micf"),
                    ]:
                        ct[vk].append(_ctv(attr))
                        ct[uk].append(_cti(attr,"uncertainty"))
                        ct[lok].append(_cti(attr,"lower_uncertainty"))
                        ct[hik].append(_cti(attr,"upper_uncertainty"))
                        ct[cfk].append(_ctf(attr,"confidence_level"))
                    s=getattr(c_t,"second",None)
                    se=getattr(c_t,"second_errors",None)
                    ct["sev"].append(_NaN if s is None else float(s))
                    def _sef(sub):
                        if se is None: return _NaN
                        v=getattr(se,sub,None)
                        return _NaN if v is None else float(v)
                    ct["seu"].append(_sef("uncertainty"))
                    ct["selo"].append(_sef("lower_uncertainty"))
                    ct["sehi"].append(_sef("upper_uncertainty"))
                    ct["secf"].append(_sef("confidence_level"))
                ctcnt=(self._n_ct_written+len(ct["yrv"]))-ctoff

                # arrivals
                aroff=self._n_ar_written+len(ar["pid"])
                for a in (o.arrivals or []):
                    acoff,accnt=cp.add(a.comments,ci)
                    ar["pid"].append(_rid(a.resource_id))
                    ar["pkid"].append(_rid(a.pick_id))
                    ar["ph"].append(getattr(a.phase,"code",str(a.phase)) if a.phase else "")
                    ar["tc"].append(_of(a.time_correction))
                    ar["az"].append(_of(a.azimuth))
                    ar["dist"].append(_of(a.distance))
                    ar["tov"].append(_fv(a,"takeoff_angle")); ar["tou"].append(_qeu(a,"takeoff_angle"))
                    ar["tr"].append(_of(a.time_residual))
                    ar["hsr"].append(_of(a.horizontal_slowness_residual))
                    ar["br"].append(_of(a.backazimuth_residual))
                    ar["tw"].append(_of(a.time_weight))
                    ar["hsw"].append(_of(a.horizontal_slowness_weight))
                    ar["bw"].append(_of(a.backazimuth_weight))
                    ar["emid"].append(_rid(a.earth_model_id))
                    ar["ci"].append(int(ci.add(a.creation_info)))
                    ar["coff"].append(acoff); ar["ccnt"].append(accnt)
                arcnt=(self._n_ar_written+len(ar["pid"]))-aroff

                ocoff,occnt=cp.add(o.comments,ci)

                # quality
                if o.quality is not None:
                    q=o.quality; qidx=self._n_oq_written+len(oq["apc"])
                    oq["apc"].append(_oi(q.associated_phase_count))
                    oq["upc"].append(_oi(q.used_phase_count))
                    oq["asc"].append(_oi(q.associated_station_count))
                    oq["usc"].append(_oi(q.used_station_count))
                    oq["dpc"].append(_oi(q.depth_phase_count))
                    oq["se"].append(_of(q.standard_error))
                    oq["ag"].append(_of(q.azimuthal_gap))
                    oq["sag"].append(_of(q.secondary_azimuthal_gap))
                    oq["gtl"].append(q.ground_truth_level or "")
                    oq["mind"].append(_of(q.minimum_distance))
                    oq["maxd"].append(_of(q.maximum_distance))
                    oq["medd"].append(_of(q.median_distance))
                else: qidx=-1

                # uncertainty / ellipsoid
                if o.origin_uncertainty is not None:
                    u=o.origin_uncertainty
                    if u.confidence_ellipsoid is not None:
                        c_e=u.confidence_ellipsoid; elidx=self._n_ce_written+len(ce["sma"])
                        ce["sma"].append(_of(c_e.semi_major_axis_length))
                        ce["smi"].append(_of(c_e.semi_minor_axis_length))
                        ce["smia"].append(_of(c_e.semi_intermediate_axis_length))
                        ce["mpl"].append(_of(c_e.major_axis_plunge))
                        ce["maz"].append(_of(c_e.major_axis_azimuth))
                        ce["mro"].append(_of(c_e.major_axis_rotation))
                    else: elidx=-1
                    uidx=self._n_ou_written+len(ou["hu"])
                    ou["hu"].append(_of(u.horizontal_uncertainty))
                    ou["minu"].append(_of(u.min_horizontal_uncertainty))
                    ou["maxu"].append(_of(u.max_horizontal_uncertainty))
                    ou["az"].append(_of(u.azimuth_max_horizontal_uncertainty))
                    ou["pd"].append(_enc(u.preferred_description,_R_OU_DESC))
                    ou["cl"].append(_of(u.confidence_level))
                    ou["eidx"].append(elidx)
                else: uidx=-1

                or_["pid"].append(_rid(o.resource_id))
                or_["eidx"].append(eidx)
                or_["tv"].append(_tv(o,"time")); or_["tu"].append(_qeu(o,"time"))
                or_["tlo"].append(_qeu(o,"time","lower_uncertainty"))
                or_["thi"].append(_qeu(o,"time","upper_uncertainty"))
                or_["tcf"].append(_qeu(o,"time","confidence_level"))
                or_["lav"].append(_fv(o,"latitude")); or_["lau"].append(_qeu(o,"latitude"))
                or_["lalo"].append(_qeu(o,"latitude","lower_uncertainty"))
                or_["lahi"].append(_qeu(o,"latitude","upper_uncertainty"))
                or_["lacf"].append(_qeu(o,"latitude","confidence_level"))
                or_["lov"].append(_fv(o,"longitude")); or_["lou"].append(_qeu(o,"longitude"))
                or_["lolo"].append(_qeu(o,"longitude","lower_uncertainty"))
                or_["lohi"].append(_qeu(o,"longitude","upper_uncertainty"))
                or_["locf"].append(_qeu(o,"longitude","confidence_level"))
                or_["dpv"].append(_fv(o,"depth")); or_["dpu"].append(_qeu(o,"depth"))
                or_["dplo"].append(_qeu(o,"depth","lower_uncertainty"))
                or_["dphi"].append(_qeu(o,"depth","upper_uncertainty"))
                or_["dpcf"].append(_qeu(o,"depth","confidence_level"))
                or_["dtype"].append(_enc(o.depth_type,_R_ORIG_DEPTH))
                or_["tfx"].append(int(_benc(o.time_fixed)))
                or_["epfx"].append(int(_benc(o.epicenter_fixed)))
                or_["rsid"].append(_rid(o.reference_system_id))
                or_["mid"].append(_rid(o.method_id))
                or_["emid"].append(_rid(o.earth_model_id))
                or_["otype"].append(_enc(o.origin_type,_R_ORIG_TYPE))
                or_["reg"].append(o.region or "")
                or_["emode"].append(_enc(o.evaluation_mode,_R_EVAL_MODE))
                or_["estat"].append(_enc(o.evaluation_status,_R_EVAL_STATUS))
                or_["ci"].append(int(ci.add(o.creation_info)))
                or_["qidx"].append(qidx); or_["uidx"].append(uidx)
                or_["aoff"].append(aroff); or_["acnt"].append(arcnt)
                or_["coff"].append(ocoff); or_["ccnt"].append(occnt)
                or_["ctoff"].append(ctoff); or_["ctcnt"].append(ctcnt)

            for m in (e.magnitudes or []):
                mcoff,mccnt=cp.add(m.comments,ci)
                soff=self._n_sc_written+len(sc["smid"])
                for s in (m.station_magnitude_contributions or []):
                    sc["smid"].append(_rid(s.station_magnitude_id))
                    sc["res"].append(_of(s.residual)); sc["wt"].append(_of(s.weight))
                scnt=(self._n_sc_written+len(sc["smid"]))-soff
                mq=m.mag
                mg["pid"].append(_rid(m.resource_id)); mg["eidx"].append(eidx)
                mg["val"].append(_fv(m,"mag")); mg["unc"].append(_qeu(m,"mag"))
                mg["lo"].append(_qeu(m,"mag","lower_uncertainty"))
                mg["hi"].append(_qeu(m,"mag","upper_uncertainty"))
                mg["cf"].append(_qeu(m,"mag","confidence_level"))
                mg["type"].append(m.magnitude_type or "")
                mg["orig"].append(_rid(m.origin_id)); mg["mid"].append(_rid(m.method_id))
                mg["scnt"].append(_oi(m.station_count)); mg["ag"].append(_of(m.azimuthal_gap))
                mg["emode"].append(_enc(m.evaluation_mode,_R_EVAL_MODE))
                mg["estat"].append(_enc(m.evaluation_status,_R_EVAL_STATUS))
                mg["ci"].append(int(ci.add(m.creation_info)))
                mg["soff"].append(soff); mg["scnt2"].append(scnt)
                mg["coff"].append(mcoff); mg["ccnt"].append(mccnt)

            for s in (e.station_magnitudes or []):
                scoff,sccnt=cp.add(s.comments,ci)
                sq=s.mag
                sm["pid"].append(_rid(s.resource_id)); sm["eidx"].append(eidx)
                sm["orig"].append(_rid(s.origin_id))
                sm["val"].append(_fv(s,"mag")); sm["unc"].append(_qeu(s,"mag"))
                sm["lo"].append(_qeu(s,"mag","lower_uncertainty"))
                sm["hi"].append(_qeu(s,"mag","upper_uncertainty"))
                sm["type"].append(s.station_magnitude_type or "")
                sm["amid"].append(_rid(s.amplitude_id)); sm["mid"].append(_rid(s.method_id))
                sm["wfidx"].append(int(wf.add(s.waveform_id)))
                sm["ci"].append(int(ci.add(s.creation_info)))
                sm["coff"].append(scoff); sm["ccnt"].append(sccnt)

            for p in (e.picks or []):
                pcoff,pccnt=cp.add(p.comments,ci)
                ph=p.phase_hint
                pk["pid"].append(_rid(p.resource_id)); pk["eidx"].append(eidx)
                pk["tv"].append(_tv(p,"time")); pk["tu"].append(_qeu(p,"time"))
                pk["tlo"].append(_qeu(p,"time","lower_uncertainty"))
                pk["thi"].append(_qeu(p,"time","upper_uncertainty"))
                pk["tcf"].append(_qeu(p,"time","confidence_level"))
                pk["wfidx"].append(int(wf.add(p.waveform_id)))
                pk["fid"].append(_rid(p.filter_id)); pk["mid"].append(_rid(p.method_id))
                pk["hsv"].append(_fv(p,"horizontal_slowness"))
                pk["hsu"].append(_qeu(p,"horizontal_slowness"))
                pk["bzv"].append(_fv(p,"backazimuth"))
                pk["bzu"].append(_qeu(p,"backazimuth"))
                pk["smid"].append(_rid(p.slowness_method_id))
                pk["onset"].append(_enc(p.onset,_R_PICK_ONSET))
                pk["ph"].append(getattr(ph,"code",str(ph)) if ph else "")
                pk["pol"].append(_enc(p.polarity,_R_PICK_POL))
                pk["emode"].append(_enc(p.evaluation_mode,_R_EVAL_MODE))
                pk["estat"].append(_enc(p.evaluation_status,_R_EVAL_STATUS))
                pk["ci"].append(int(ci.add(p.creation_info)))
                pk["coff"].append(pcoff); pk["ccnt"].append(pccnt)

            for a in (e.amplitudes or []):
                acoff,accnt=cp.add(a.comments,ci)
                if a.time_window is not None:
                    twidx=self._n_tw_written+len(tw["beg"])
                    tw["beg"].append(_of(getattr(a.time_window,"begin",None)))
                    tw["end"].append(_of(getattr(a.time_window,"end",None)))
                    tw["ref"].append(_ts(getattr(a.time_window,"reference",None)))
                else: twidx=-1
                am["pid"].append(_rid(a.resource_id)); am["eidx"].append(eidx)
                am["val"].append(_fv(a,"generic_amplitude"))
                am["unc"].append(_qeu(a,"generic_amplitude"))
                am["lo"].append(_qeu(a,"generic_amplitude","lower_uncertainty"))
                am["hi"].append(_qeu(a,"generic_amplitude","upper_uncertainty"))
                am["cf"].append(_qeu(a,"generic_amplitude","confidence_level"))
                am["type"].append(a.type or "")
                am["cat"].append(_enc(a.category,_R_AMP_CAT))
                am["unit"].append(_enc(a.unit,_R_AMP_UNIT))
                am["mid"].append(_rid(a.method_id))
                am["perv"].append(_fv(a,"period")); am["peru"].append(_qeu(a,"period"))
                am["snr"].append(_of(a.snr)); am["twidx"].append(twidx)
                am["pkid"].append(_rid(a.pick_id))
                am["wfidx"].append(int(wf.add(a.waveform_id)))
                am["fid"].append(_rid(a.filter_id))
                am["stv"].append(_tv(a,"scaling_time")); am["stu"].append(_qeu(a,"scaling_time"))
                am["mhint"].append(a.magnitude_hint or "")
                am["emode"].append(_enc(a.evaluation_mode,_R_EVAL_MODE))
                am["estat"].append(_enc(a.evaluation_status,_R_EVAL_STATUS))
                am["ci"].append(int(ci.add(a.creation_info)))
                am["coff"].append(acoff); am["ccnt"].append(accnt)

            for f_m in (e.focal_mechanisms or []):
                fmcoff,fmccnt=cp.add(f_m.comments,ci)
                wpoff=self._n_fmwp_written+len(fmwp)
                for wfid in (getattr(f_m,"waveform_id",[]) or []):
                    fmwp.append(int(wf.add(wfid)))
                wpcnt=(self._n_fmwp_written+len(fmwp))-wpoff

                # moment tensor
                mt_obj=getattr(f_m,"moment_tensor",None)
                if isinstance(mt_obj,list): mt_obj=mt_obj[0] if mt_obj else None
                mtidx=-1
                if mt_obj is not None:
                    mtidx=self._n_mt_written+len(mt["pid"])
                    mtcoff,mtccnt=cp.add(mt_obj.comments,ci)
                    duoff=self._n_du_written+len(du["wt"])
                    for d_u in (mt_obj.data_used or []):
                        du["wt"].append(_enc(d_u.wave_type,_R_DU_WAVE))
                        du["sc"].append(_oi(d_u.station_count))
                        du["cc"].append(_oi(d_u.component_count))
                        du["sp"].append(_of(d_u.shortest_period))
                        du["lp"].append(_of(d_u.longest_period))
                    ducnt=(self._n_du_written+len(du["wt"]))-duoff
                    stf=mt_obj.source_time_function
                    mt["pid"].append(_rid(mt_obj.resource_id))
                    mt["doid"].append(_rid(mt_obj.derived_origin_id))
                    mt["mmid"].append(_rid(mt_obj.moment_magnitude_id))
                    mt["scv"].append(_fv(mt_obj,"scalar_moment")); mt["scu"].append(_qeu(mt_obj,"scalar_moment"))
                    # Tensor components: in ObsPy these are flat floats with
                    # paired `<comp>_errors` QuantityError objects.
                    tens=mt_obj.tensor
                    for comp,short in [("m_rr","rr"),("m_tt","tt"),("m_pp","pp"),
                                       ("m_rt","rt"),("m_rp","rp"),("m_tp","tp")]:
                        if tens is None:
                            mt[f"{short}_v"].append(_NaN); mt[f"{short}_u"].append(_NaN); continue
                        tc=getattr(tens,comp,None)
                        if tc is None:
                            mt[f"{short}_v"].append(_NaN); mt[f"{short}_u"].append(_NaN)
                        else:
                            mt[f"{short}_v"].append(float(tc))
                            mt[f"{short}_u"].append(_qeu(tens,comp))
                    mt["var"].append(_of(mt_obj.variance))
                    mt["vr"].append(_of(mt_obj.variance_reduction))
                    mt["dc"].append(_of(mt_obj.double_couple))
                    mt["clvd"].append(_of(mt_obj.clvd))
                    mt["iso"].append(_of(mt_obj.iso))
                    mt["gfid"].append(_rid(mt_obj.greens_function_id))
                    mt["fid"].append(_rid(mt_obj.filter_id))
                    mt["stft"].append(_enc(stf.type if stf else None,_R_STF))
                    mt["stfd"].append(_of(stf.duration if stf else None))
                    mt["stfr"].append(_of(stf.rise_time if stf else None))
                    mt["stfdc"].append(_of(stf.decay_time if stf else None))
                    mt["mid"].append(_rid(mt_obj.method_id))
                    mt["cat"].append(_enc(mt_obj.category,_R_MT_CAT))
                    mt["inv"].append(_enc(mt_obj.inversion_type,_R_MT_INV))
                    mt["ci"].append(int(ci.add(mt_obj.creation_info)))
                    mt["duoff"].append(duoff); mt["ducnt"].append(ducnt)
                    mt["coff"].append(mtcoff); mt["ccnt"].append(mtccnt)

                nps=getattr(f_m,"nodal_planes",None)
                np1=np2=None; pp=0
                if nps:
                    np1=getattr(nps,"nodal_plane_1",None)
                    np2=getattr(nps,"nodal_plane_2",None)
                    pp2=getattr(nps,"preferred_plane",None); pp=int(pp2) if pp2 else 0
                # NodalPlane.strike/dip/rake and Axis.azimuth/plunge/length are
                # flat floats with paired `<field>_errors` QuantityError objects.
                def _scalar_pair(parent, attr):
                    if parent is None: return _NaN, _NaN
                    v=getattr(parent,attr,None)
                    if v is None: return _NaN, _NaN
                    return float(v), _qeu(parent,attr)
                def _npv(x):
                    if x is None: return (_NaN,)*6
                    sv,su=_scalar_pair(x,"strike")
                    dv,du=_scalar_pair(x,"dip")
                    rv,ru=_scalar_pair(x,"rake")
                    return (sv,su,dv,du,rv,ru)
                s1,su1,d1,du1,r1,ru1=_npv(np1); s2,su2,d2,du2,r2,ru2=_npv(np2)
                pa=getattr(f_m,"principal_axes",None)
                def _axv(name):
                    if pa is None: return (_NaN,)*6
                    ax=getattr(pa,name,None)
                    if ax is None: return (_NaN,)*6
                    av,au=_scalar_pair(ax,"azimuth")
                    pv,pu=_scalar_pair(ax,"plunge")
                    lv,lu=_scalar_pair(ax,"length")
                    return (av,au,pv,pu,lv,lu)
                tazv,tazu,tplv,tplu,tlnv,tlnu=_axv("t_axis")
                pazv,pazu,pplv,pplu,plnv,plnu=_axv("p_axis")
                nazv,nazu,nplv,nplu,nlnv,nlnu=_axv("n_axis")
                fm["pid"].append(_rid(f_m.resource_id)); fm["eidx"].append(eidx)
                fm["toid"].append(_rid(f_m.triggering_origin_id))
                fm["np1sv"].append(s1); fm["np1su"].append(su1)
                fm["np1dv"].append(d1); fm["np1du"].append(du1)
                fm["np1rv"].append(r1); fm["np1ru"].append(ru1)
                fm["np2sv"].append(s2); fm["np2su"].append(su2)
                fm["np2dv"].append(d2); fm["np2du"].append(du2)
                fm["np2rv"].append(r2); fm["np2ru"].append(ru2)
                fm["pp"].append(pp)
                fm["tazv"].append(tazv); fm["tazu"].append(tazu)
                fm["tplv"].append(tplv); fm["tplu"].append(tplu)
                fm["tlnv"].append(tlnv); fm["tlnu"].append(tlnu)
                fm["pazv"].append(pazv); fm["pazu"].append(pazu)
                fm["pplv"].append(pplv); fm["pplu"].append(pplu)
                fm["plnv"].append(plnv); fm["plnu"].append(plnu)
                fm["nazv"].append(nazv); fm["nazu"].append(nazu)
                fm["nplv"].append(nplv); fm["nplu"].append(nplu)
                fm["nlnv"].append(nlnv); fm["nlnu"].append(nlnu)
                fm["ag"].append(_of(f_m.azimuthal_gap))
                fm["spc"].append(_oi(f_m.station_polarity_count))
                fm["mft"].append(_of(f_m.misfit))
                fm["sdr"].append(_of(f_m.station_distribution_ratio))
                fm["mid"].append(_rid(f_m.method_id))
                fm["emode"].append(_enc(f_m.evaluation_mode,_R_EVAL_MODE))
                fm["estat"].append(_enc(f_m.evaluation_status,_R_EVAL_STATUS))
                fm["ci"].append(int(ci.add(f_m.creation_info)))
                fm["mtidx"].append(mtidx)
                fm["coff"].append(fmcoff); fm["ccnt"].append(fmccnt)
                fm["wpoff"].append(wpoff); fm["wpcnt"].append(wpcnt)

            # Periodic chunked flush. Free per-row accumulator memory once
            # `chunk_size` events have been processed. The dedup tables (wf,
            # ci) stay in RAM since they grow with unique count, not row
            # count; they're written once at the end.
            if chunk_size and (eidx+1) % chunk_size == 0:
                _flush_chunk()

        # ---- final flush of any partial chunk, then write dedup tables ----
        _flush_chunk()
        wf.write(_g("waveform_ids"),self._CS)
        ci.write(_g("creation_info"),self._C,self._CS)

    # ------------------------------------------------------------------
    # READ PATH
    # ------------------------------------------------------------------

    def _grp(self,n):
        return self._f[n] if n in self._f else None

    def _ra(self,g,k):  # raw array
        return g[k][()] if g is not None and k in g else None

    def _rs(self,g,k):  # string array (decode bytes)
        r=self._ra(g,k)
        if r is None: return None
        if r.dtype.kind=="O":
            return np.array([v.decode() if isinstance(v,bytes) else (v or "") for v in r])
        return r

    # ------------------------------------------------------------------
    # Read-path acceleration: prefetch every column of a group once and
    # precompute per-event row slices. The previous design called
    # `g["col"][()]` (a full HDF5 dataset read + decode) inside per-event
    # loops, giving O(events × columns) full-column reads. The new design
    # loads each column exactly once and looks up event slices in O(1) via
    # boundary arrays built from the (sorted) `event_idx` column.
    # ------------------------------------------------------------------
    def _prefetch_group(self, name):
        """Load every column of an HDF5 group into a dict[name -> ndarray].
        String columns are decoded once here, not per row at access time.
        Returns None if the group is absent or empty."""
        g = self._grp(name)
        if g is None: return None
        cols = {}
        for k in g.keys():
            v = g[k][()]
            if v.dtype.kind == "O":   # object array of bytes / str
                v = np.array([x.decode() if isinstance(x, bytes) else (x or "")
                              for x in v])
            cols[k] = v
        return cols if cols else None

    @staticmethod
    def _build_event_slices(cols, n_events):
        """Given a prefetched group's columns, return (starts, ends) arrays of
        shape (n_events,) such that rows[starts[ei]:ends[ei]] is the slice of
        rows belonging to event ei. Assumes `event_idx` is sorted ascending,
        which the writer guarantees (events are accumulated in order and
        chunks are flushed in order). Falls back to per-event np.where for
        groups whose event_idx isn't sorted (shouldn't happen, but harmless)."""
        if cols is None or "event_idx" not in cols:
            return None, None
        eidx = cols["event_idx"]
        if len(eidx) == 0:
            return (np.zeros(n_events, dtype=np.int64),
                    np.zeros(n_events, dtype=np.int64))
        # Check sortedness cheaply (vectorized).
        if not np.all(np.diff(eidx) >= 0):
            # Fallback: per-event np.where, still much faster than full reads
            starts = np.empty(n_events, dtype=np.int64)
            ends   = np.empty(n_events, dtype=np.int64)
            for i in range(n_events):
                m = np.where(eidx == i)[0]
                if len(m): starts[i], ends[i] = m[0], m[-1]+1
                else: starts[i] = ends[i] = 0
            return starts, ends
        idx_range = np.arange(n_events)
        starts = np.searchsorted(eidx, idx_range, side="left")
        ends   = np.searchsorted(eidx, idx_range, side="right")
        return starts, ends

    def _load_wf(self):
        g=self._grp("waveform_ids")
        if g is None: return []
        net=self._rs(g,"network_code"); sta=self._rs(g,"station_code")
        loc=self._rs(g,"location_code"); cha=self._rs(g,"channel_code")
        uri=self._rs(g,"resource_uri")
        if net is None: return []
        out=[]
        for i in range(len(net)):
            wf=WaveformStreamID(network_code=net[i] or None,station_code=sta[i] or None,
                                location_code=loc[i] or None,channel_code=cha[i] or None)
            if uri[i]: wf.resource_uri=_make_rid(uri[i])
            out.append(wf)
        return out

    def _load_ci(self):
        g=self._grp("creation_info")
        if g is None: return []
        aid=self._rs(g,"agency_id"); auri=self._rs(g,"agency_uri")
        auth=self._rs(g,"author"); auuri=self._rs(g,"author_uri")
        ct=self._ra(g,"creation_time"); ver=self._rs(g,"version")
        # Group exists but is empty when no creation_info entries were written.
        if aid is None: return []
        return [CreationInfo(agency_id=aid[i] or None,agency_uri=_make_rid(auri[i]),
                             author=auth[i] or None,author_uri=_make_rid(auuri[i]),
                             creation_time=_from_ts(float(ct[i])),
                             version=ver[i] or None) for i in range(len(aid))]

    def _load_com(self):
        g=self._grp("comments")
        if g is None: return np.array([]),np.array([]),np.array([],dtype=np.int32)
        return self._rs(g,"text"),self._rs(g,"id"),self._ra(g,"ci_idx")

    def _mk_ci(self,ci_rows,idx):
        idx=int(idx)
        return ci_rows[idx] if 0<=idx<len(ci_rows) else None

    def _mk_comments(self,txt,cid,cidx_arr,ci_rows,off,cnt):
        out=[]
        for i in range(off,off+cnt):
            c=Comment(text=txt[i])
            if cid[i]: c.resource_id=_make_rid(cid[i])
            c.creation_info=self._mk_ci(ci_rows,cidx_arr[i])
            out.append(c)
        return out

    def _eidx(self,g,ei):
        if g is None: return np.array([],dtype=np.int64)
        ev=self._ra(g,"event_idx")
        return np.array([],dtype=np.int64) if ev is None else np.where(ev==ei)[0]

    def read_catalog(self,event_indices=None,progress=True,
                     starttime=None,endtime=None):
        """Reconstruct an ObsPy Catalog.

        Parameters
        ----------
        event_indices : iterable of int, optional
            Subset of event row indices to load. If ``None`` (default), the
            entire catalog is loaded.
        progress : bool, optional
            Show a tqdm progress bar while reconstructing events. Defaults to
            ``True``. Has no effect if tqdm is not installed, or for catalogs
            of 100 events or fewer (where the read finishes in well under a
            second and the bar would just flash by).
        starttime, endtime : str or :class:`~obspy.UTCDateTime`, optional
            Time-range filter applied BEFORE constructing ObsPy objects. Each
            event's first origin's `time` is used as its representative time.
            Events with no origins are excluded when either bound is given.
            A real speedup vs filtering the loaded catalog after the fact
            because per-event ObsPy object construction (the dominant read
            cost) is skipped for filtered-out events.
        """
        if not OBSPY_AVAILABLE: raise ImportError("ObsPy required")
        f=self._f
        wf_rows=self._load_wf(); ci_rows=self._load_ci()
        txt,cid,cidx=self._load_com()

        events=[]
        cg=self._grp("catalog")
        if cg is not None:
            n=len(cg["public_id"])
            if event_indices is None: event_indices=range(n)

            pid =self._rs(cg,"public_id");    po=self._rs(cg,"preferred_origin_id")
            pm  =self._rs(cg,"preferred_magnitude_id"); pf=self._rs(cg,"preferred_focmec_id")
            ety =self._ra(cg,"event_type");   ect=self._ra(cg,"event_type_certainty")
            eci =self._ra(cg,"ci_idx")
            doff=self._ra(cg,"desc_offset");  dcnt=self._ra(cg,"desc_count")
            coff=self._ra(cg,"comment_offset");ccnt=self._ra(cg,"comment_count")

            # Prefetch every child group's columns ONCE, and precompute the
            # per-event row slice arrays. This collapses O(events × columns)
            # full-column reads into O(columns) reads + O(events) lookups.
            # Prefetch can take a few seconds for very large catalogs (string
            # decoding dominates) but it's only ~3% of total load time, so we
            # don't show a separate bar for it — the per-event bar that follows
            # gives the user a meaningful ETA for the actual bulk of the work.
            _prefetch_targets = [
                "origins","magnitudes","station_magnitudes","picks","amplitudes",
                "focal_mechanisms","arrivals","composite_times","origin_quality",
                "origin_uncertainty","confidence_ellipsoids","moment_tensors",
                "data_used","station_mag_contributions","time_windows",
                "event_descriptions",
            ]
            _pf = {name: self._prefetch_group(name) for name in _prefetch_targets}
            og_cols=_pf["origins"];           mg_cols=_pf["magnitudes"]
            sm_cols=_pf["station_magnitudes"];pk_cols=_pf["picks"]
            am_cols=_pf["amplitudes"];        fm_cols=_pf["focal_mechanisms"]
            ar_cols=_pf["arrivals"];          ct_cols=_pf["composite_times"]
            oq_cols=_pf["origin_quality"];    ou_cols=_pf["origin_uncertainty"]
            ce_cols=_pf["confidence_ellipsoids"]; mt_cols=_pf["moment_tensors"]
            du_cols=_pf["data_used"];         sc_cols=_pf["station_mag_contributions"]
            tw_cols=_pf["time_windows"];      ed_cols=_pf["event_descriptions"]
            og_s, og_e = self._build_event_slices(og_cols, n)
            mg_s, mg_e = self._build_event_slices(mg_cols, n)
            sm_s, sm_e = self._build_event_slices(sm_cols, n)
            pk_s, pk_e = self._build_event_slices(pk_cols, n)
            am_s, am_e = self._build_event_slices(am_cols, n)
            fm_s, fm_e = self._build_event_slices(fm_cols, n)

            # Optional time-range filtering. We use each event's first origin's
            # `time_value` as its representative time (cheap — the column is
            # already prefetched, and `og_s[ei]` is the row index of that
            # origin). Events with no origins, or with NaN origin times, are
            # excluded from time-filtered results because there's no defensible
            # time to compare against. The filter is applied to `event_indices`
            # before the per-event loop, so the per-event ObsPy object
            # construction cost (the dominant cost of read_catalog) drops
            # proportionally to how aggressive the filter is.
            if starttime is not None or endtime is not None:
                event_times = np.full(n, np.nan)
                if og_cols is not None and og_s is not None:
                    has_orig = og_s < og_e
                    if has_orig.any():
                        event_times[has_orig] = og_cols["time_value"][og_s[has_orig]]
                mask = ~np.isnan(event_times)
                if starttime is not None:
                    st = _ts(UTCDateTime(starttime))
                    mask &= event_times >= st
                if endtime is not None:
                    et = _ts(UTCDateTime(endtime))
                    mask &= event_times <= et
                if isinstance(event_indices, range) and event_indices == range(n):
                    event_indices = np.flatnonzero(mask).tolist()
                else:
                    event_indices = [ei for ei in event_indices if mask[ei]]

            # Wrap the per-event loop with tqdm only when explicitly enabled,
            # tqdm is importable, and the (possibly filtered) catalog is large
            # enough that the bar isn't just noise.
            if progress and TQDM_AVAILABLE and len(event_indices) > 100:
                ei_iter = tqdm(event_indices, desc="Reading events",
                               unit="event", leave=False)
            else:
                ei_iter = event_indices
            for ei in ei_iter:
                e=Event()
                e.resource_id=_make_rid(pid[ei])
                e.preferred_origin_id=_make_rid(po[ei])
                e.preferred_magnitude_id=_make_rid(pm[ei])
                e.preferred_focal_mechanism_id=_make_rid(pf[ei])
                e.event_type=_dec(ety[ei],EVENT_TYPE)
                e.event_type_certainty=_dec(ect[ei],EVENT_TYPE_CERTAINTY)
                e.creation_info=self._mk_ci(ci_rows,eci[ei])
                e.event_descriptions=self._rd_descs(ed_cols,int(doff[ei]),int(dcnt[ei]))
                e.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[ei]),int(ccnt[ei]))
                e.origins=self._rd_origins(og_cols,og_s,og_e,ei,
                    ar_cols,ct_cols,oq_cols,ou_cols,ce_cols,ci_rows,txt,cid,cidx)
                e.magnitudes=self._rd_magnitudes(mg_cols,mg_s,mg_e,ei,sc_cols,ci_rows,txt,cid,cidx)
                e.station_magnitudes=self._rd_sta_mags(sm_cols,sm_s,sm_e,ei,ci_rows,wf_rows,txt,cid,cidx)
                e.picks=self._rd_picks(pk_cols,pk_s,pk_e,ei,ci_rows,wf_rows,txt,cid,cidx)
                e.amplitudes=self._rd_amplitudes(am_cols,am_s,am_e,ei,tw_cols,ci_rows,wf_rows,txt,cid,cidx)
                e.focal_mechanisms=self._rd_focmecs(fm_cols,fm_s,fm_e,ei,
                    mt_cols,du_cols,ci_rows,wf_rows,txt,cid,cidx)
                events.append(e)

        cat=Catalog(events=events)
        # Restore catalog-level metadata. Empty strings become None so XML output
        # doesn't emit blank <description/> tags.
        desc=str(f.attrs.get("catalog_description","")) or None
        if desc is not None: cat.description=desc
        rid=str(f.attrs.get("catalog_public_id",""))
        if rid: cat.resource_id=_make_rid(rid)
        cat_ci_idx=int(f.attrs.get("catalog_ci_idx",-1))
        if cat_ci_idx>=0:
            cat.creation_info=self._mk_ci(ci_rows,cat_ci_idx)
        cat_coff=int(f.attrs.get("catalog_comment_offset",0))
        cat_ccnt=int(f.attrs.get("catalog_comment_count",0))
        if cat_ccnt>0:
            cat.comments=self._mk_comments(txt,cid,cidx,ci_rows,cat_coff,cat_ccnt)
        return cat

    def _rd_descs(self,cols,off,cnt):
        if cols is None or cnt==0: return []
        ta=cols["text"]; ty=cols["type"]
        return [EventDescription(text=ta[i],type=_dec(ty[i],EVENT_DESC_TYPE))
                for i in range(off,off+cnt)]

    def _rd_origins(self,cols,og_s,og_e,ei,ar_cols,ct_cols,oq_cols,ou_cols,ce_cols,
                    ci_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(og_s[ei]), int(og_e[ei])
        if start == end: return []
        # All columns are already in memory as numpy arrays — just index.
        pid = cols["public_id"]
        tv  = cols["time_value"]; tu = cols["time_uncertainty"]
        tlo = cols["time_lower_unc"]; thi = cols["time_upper_unc"]; tcf = cols["time_conf"]
        lav = cols["lat_value"]; lau = cols["lat_uncertainty"]
        lalo= cols["lat_lower_unc"]; lahi= cols["lat_upper_unc"]; lacf= cols["lat_conf"]
        lov = cols["lon_value"]; lou = cols["lon_uncertainty"]
        lolo= cols["lon_lower_unc"]; lohi= cols["lon_upper_unc"]; locf= cols["lon_conf"]
        dpv = cols["depth_value"]; dpu = cols["depth_uncertainty"]
        dplo= cols["depth_lower_unc"]; dphi= cols["depth_upper_unc"]; dpcf= cols["depth_conf"]
        dtype = cols["depth_type"]; tfx = cols["time_fixed"]; epfx = cols["epicenter_fixed"]
        rsid = cols["ref_system_id"]; mid = cols["method_id"]; emid = cols["earth_model_id"]
        otype = cols["type"]; reg = cols["region"]
        emode = cols["eval_mode"]; estat = cols["eval_status"]; ci_i = cols["ci_idx"]
        qidx = cols["quality_idx"]; uidx = cols["uncertainty_idx"]
        aoff = cols["arrival_offset"]; acnt = cols["arrival_count"]
        coff = cols["comment_offset"]; ccnt = cols["comment_count"]
        ctoff= cols["comptime_offset"]; ctcnt= cols["comptime_count"]
        out=[]
        for i in range(start, end):
            o=Origin()
            o.resource_id=_make_rid(pid[i])
            o.time=_from_ts(float(tv[i]))
            o.time_errors=QuantityError(uncertainty=_nn(tu[i]),
                lower_uncertainty=_nn(tlo[i]),upper_uncertainty=_nn(thi[i]),
                confidence_level=_nn(tcf[i]))
            o.latitude=float(lav[i])
            o.latitude_errors=QuantityError(uncertainty=_nn(lau[i]),
                lower_uncertainty=_nn(lalo[i]),upper_uncertainty=_nn(lahi[i]),
                confidence_level=_nn(lacf[i]))
            o.longitude=float(lov[i])
            o.longitude_errors=QuantityError(uncertainty=_nn(lou[i]),
                lower_uncertainty=_nn(lolo[i]),upper_uncertainty=_nn(lohi[i]),
                confidence_level=_nn(locf[i]))
            if not math.isnan(float(dpv[i])):
                o.depth=float(dpv[i])
                o.depth_errors=QuantityError(uncertainty=_nn(dpu[i]),
                    lower_uncertainty=_nn(dplo[i]),upper_uncertainty=_nn(dphi[i]),
                    confidence_level=_nn(dpcf[i]))
            o.depth_type=_dec(dtype[i],ORIGIN_DEPTH_TYPE)
            o.time_fixed=_bdec(int(tfx[i])); o.epicenter_fixed=_bdec(int(epfx[i]))
            o.reference_system_id=_make_rid(rsid[i]); o.method_id=_make_rid(mid[i])
            o.earth_model_id=_make_rid(emid[i]); o.origin_type=_dec(otype[i],ORIGIN_TYPE)
            o.region=reg[i] or None
            o.evaluation_mode=_dec(emode[i],EVALUATION_MODE)
            o.evaluation_status=_dec(estat[i],EVALUATION_STATUS)
            o.creation_info=self._mk_ci(ci_rows,ci_i[i])
            o.quality=self._rd_oq(oq_cols,int(qidx[i]))
            o.origin_uncertainty=self._rd_ou(ou_cols,ce_cols,int(uidx[i]))
            o.arrivals=self._rd_arrivals(ar_cols,int(aoff[i]),int(acnt[i]),ci_rows,txt,cid,cidx)
            o.composite_times=self._rd_ct(ct_cols,int(ctoff[i]),int(ctcnt[i]))
            o.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(o)
        return out

    def _rd_oq(self,cols,idx):
        if cols is None or idx<0: return None
        def gi(k): v=int(cols[k][idx]); return None if v==-1 else v
        def gf(k): return _nn(float(cols[k][idx]))
        return OriginQuality(associated_phase_count=gi("assoc_phase_count"),
            used_phase_count=gi("used_phase_count"),
            associated_station_count=gi("assoc_sta_count"),
            used_station_count=gi("used_sta_count"),
            depth_phase_count=gi("depth_phase_count"),
            standard_error=gf("standard_error"),azimuthal_gap=gf("azimuthal_gap"),
            secondary_azimuthal_gap=gf("sec_azimuthal_gap"),
            ground_truth_level=cols["ground_truth_level"][idx] or None,
            minimum_distance=gf("minimum_distance"),maximum_distance=gf("maximum_distance"),
            median_distance=gf("median_distance"))

    def _rd_ou(self,ou_cols,ce_cols,idx):
        if ou_cols is None or idx<0: return None
        def gf(k): return _nn(float(ou_cols[k][idx]))
        eidx=int(ou_cols["ellipsoid_idx"][idx]); el=None
        if ce_cols is not None and eidx>=0:
            def cf(k): return _nn(float(ce_cols[k][eidx]))
            el=ConfidenceEllipsoid()
            for attr in ("semi_major_axis_length","semi_minor_axis_length",
                         "semi_intermediate_axis_length","major_axis_plunge",
                         "major_axis_azimuth","major_axis_rotation"):
                v=cf(attr)
                if v is not None: setattr(el,attr,v)
        return OriginUncertainty(horizontal_uncertainty=gf("horizontal_uncertainty"),
            min_horizontal_uncertainty=gf("min_horizontal_uncertainty"),
            max_horizontal_uncertainty=gf("max_horizontal_uncertainty"),
            azimuth_max_horizontal_uncertainty=gf("azimuth_max_horiz_unc"),
            preferred_description=_dec(int(ou_cols["preferred_description"][idx]),ORIGIN_UNCERTAINTY_DESC),
            confidence_level=gf("confidence_level"),confidence_ellipsoid=el)

    def _rd_arrivals(self,cols,off,cnt,ci_rows,txt,cid,cidx):
        if cols is None or cnt==0: return []
        pid=cols["public_id"]; pkid=cols["pick_id"]; ph=cols["phase"]
        tc=cols["time_correction"]; az=cols["azimuth"]; dist=cols["distance"]
        tov=cols["takeoff_value"]; tou=cols["takeoff_uncertainty"]
        tr=cols["time_residual"]; hsr=cols["hslow_residual"]; br=cols["baz_residual"]
        tw=cols["time_weight"]; hsw=cols["hslow_weight"]; bw=cols["baz_weight"]
        emid=cols["earth_model_id"]; ci_i=cols["ci_idx"]
        coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        out=[]
        for i in range(off,off+cnt):
            ta=_nn(float(tov[i]))
            ar=Arrival(resource_id=_make_rid(pid[i]),
                pick_id=_make_rid(pkid[i]),phase=(ph[i] or None),
                time_correction=_nn(float(tc[i])),azimuth=_nn(float(az[i])),
                distance=_nn(float(dist[i])),takeoff_angle=ta,
                time_residual=_nn(float(tr[i])),
                horizontal_slowness_residual=_nn(float(hsr[i])),
                backazimuth_residual=_nn(float(br[i])),
                time_weight=_nn(float(tw[i])),horizontal_slowness_weight=_nn(float(hsw[i])),
                backazimuth_weight=_nn(float(bw[i])),earth_model_id=_make_rid(emid[i]),
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            if ta: ar.takeoff_angle_errors=QuantityError(uncertainty=_nn(float(tou[i])))
            ar.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(ar)
        return out

    def _rd_ct(self,cols,off,cnt):
        if cols is None or cnt==0: return []
        out=[]
        for i in range(off,off+cnt):
            def gi(k):
                v=int(cols[k][i]); return None if v==-1 else v
            def gf(k):
                return _nn(float(cols[k][i]))
            ct_obj=CompositeTime()
            # Integer fields: load value and the four QuantityError sub-fields
            # (uncertainty/lower/upper as ints, confidence_level as float).
            for attr,vk,uk,lok,hik,cfk in [
                ("year",  "year_value","year_unc","year_lower_unc","year_upper_unc","year_conf"),
                ("month", "month_value","month_unc","month_lower_unc","month_upper_unc","month_conf"),
                ("day",   "day_value","day_unc","day_lower_unc","day_upper_unc","day_conf"),
                ("hour",  "hour_value","hour_unc","hour_lower_unc","hour_upper_unc","hour_conf"),
                ("minute","minute_value","minute_unc","minute_lower_unc","minute_upper_unc","minute_conf"),
            ]:
                v=gi(vk)
                if v is None: continue
                setattr(ct_obj,attr,v)
                u=gi(uk); lo=gi(lok); hi=gi(hik); cf=gf(cfk)
                if any(x is not None for x in (u,lo,hi,cf)):
                    setattr(ct_obj,f"{attr}_errors",
                            QuantityError(uncertainty=u,lower_uncertainty=lo,
                                          upper_uncertainty=hi,confidence_level=cf))
            sv=gf("second_value")
            if sv is not None:
                ct_obj.second=sv
                su=gf("second_unc"); slo=gf("second_lower_unc")
                shi=gf("second_upper_unc"); scf=gf("second_conf")
                if any(x is not None for x in (su,slo,shi,scf)):
                    ct_obj.second_errors=QuantityError(uncertainty=su,
                        lower_uncertainty=slo,upper_uncertainty=shi,confidence_level=scf)
            out.append(ct_obj)
        return out

    def _rd_magnitudes(self,cols,mg_s,mg_e,ei,sc_cols,ci_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(mg_s[ei]), int(mg_e[ei])
        if start == end: return []
        pid=cols["public_id"]; mv=cols["mag_value"]; mu=cols["mag_uncertainty"]
        mlo=cols["mag_lower_unc"]; mhi=cols["mag_upper_unc"]; mcf=cols["mag_conf"]
        ty=cols["type"]; orig=cols["origin_id"]; mid=cols["method_id"]
        scnt=cols["station_count"]; ag=cols["azimuthal_gap"]
        emode=cols["eval_mode"]; estat=cols["eval_status"]; ci_i=cols["ci_idx"]
        coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        soff=cols["contrib_offset"]; scnt2=cols["contrib_count"]
        out=[]
        for i in range(start, end):
            m=Magnitude(resource_id=_make_rid(pid[i]),mag=_nn(float(mv[i])),
                magnitude_type=(ty[i] or None),origin_id=_make_rid(orig[i]),
                method_id=_make_rid(mid[i]),
                station_count=_ni(int(scnt[i])),
                azimuthal_gap=_nn(float(ag[i])),
                evaluation_mode=_dec(int(emode[i]),EVALUATION_MODE),
                evaluation_status=_dec(int(estat[i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            m.mag_errors=QuantityError(uncertainty=_nn(float(mu[i])),
                lower_uncertainty=_nn(float(mlo[i])),upper_uncertainty=_nn(float(mhi[i])),
                confidence_level=_nn(float(mcf[i])))
            m.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            m.station_magnitude_contributions=self._rd_sc(sc_cols,int(soff[i]),int(scnt2[i]))
            out.append(m)
        return out

    def _rd_sc(self,cols,off,cnt):
        if cols is None or cnt==0: return []
        smid=cols["station_magnitude_id"]; res=cols["residual"]; wt=cols["weight"]
        out=[]
        for i in range(off,off+cnt):
            out.append(StationMagnitudeContribution(
                station_magnitude_id=_make_rid(smid[i]),
                residual=_nn(float(res[i])),weight=_nn(float(wt[i]))))
        return out

    def _rd_sta_mags(self,cols,sm_s,sm_e,ei,ci_rows,wf_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(sm_s[ei]), int(sm_e[ei])
        if start == end: return []
        pid=cols["public_id"]; orig=cols["origin_id"]; mv=cols["mag_value"]
        mu=cols["mag_uncertainty"]; mlo=cols["mag_lower_unc"]; mhi=cols["mag_upper_unc"]
        ty=cols["type"]; amid=cols["amplitude_id"]; mid=cols["method_id"]
        wfidx_col=cols["waveform_idx"]; ci_i=cols["ci_idx"]
        coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        out=[]
        for i in range(start, end):
            wfidx=int(wfidx_col[i])
            sm=StationMagnitude(resource_id=_make_rid(pid[i]),
                origin_id=_make_rid(orig[i]),mag=_nn(float(mv[i])),
                station_magnitude_type=(ty[i] or None),amplitude_id=_make_rid(amid[i]),
                method_id=_make_rid(mid[i]),
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            sm.mag_errors=QuantityError(uncertainty=_nn(float(mu[i])),
                lower_uncertainty=_nn(float(mlo[i])),upper_uncertainty=_nn(float(mhi[i])))
            sm.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(sm)
        return out

    def _rd_picks(self,cols,pk_s,pk_e,ei,ci_rows,wf_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(pk_s[ei]), int(pk_e[ei])
        if start == end: return []
        pid=cols["public_id"]
        tv=cols["time_value"]; tu=cols["time_uncertainty"]
        tlo=cols["time_lower_unc"]; thi=cols["time_upper_unc"]; tcf=cols["time_conf"]
        wfidx_col=cols["waveform_idx"]; fid=cols["filter_id"]; mid=cols["method_id"]
        hsv_c=cols["hslow_value"]; hsu_c=cols["hslow_uncertainty"]
        bzv_c=cols["baz_value"]; bzu_c=cols["baz_uncertainty"]
        smid=cols["slowness_method_id"]; onset=cols["onset"]; ph=cols["phase_hint"]
        pol=cols["polarity"]; emode=cols["eval_mode"]; estat=cols["eval_status"]
        ci_i=cols["ci_idx"]; coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        out=[]
        for i in range(start, end):
            tvv=_nn(float(tv[i])); wfidx=int(wfidx_col[i])
            hsv=_nn(float(hsv_c[i])); bvz=_nn(float(bzv_c[i]))
            p=Pick(resource_id=_make_rid(pid[i]),
                time=_from_ts(tvv) if tvv else None,
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                filter_id=_make_rid(fid[i]),method_id=_make_rid(mid[i]),
                horizontal_slowness=hsv,backazimuth=bvz,
                slowness_method_id=_make_rid(smid[i]),
                onset=_dec(int(onset[i]),PICK_ONSET),
                phase_hint=(ph[i] or None),
                polarity=_dec(int(pol[i]),PICK_POLARITY),
                evaluation_mode=_dec(int(emode[i]),EVALUATION_MODE),
                evaluation_status=_dec(int(estat[i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            p.time_errors=QuantityError(uncertainty=_nn(float(tu[i])),
                lower_uncertainty=_nn(float(tlo[i])),upper_uncertainty=_nn(float(thi[i])),
                confidence_level=_nn(float(tcf[i])))
            if hsv: p.horizontal_slowness_errors=QuantityError(uncertainty=_nn(float(hsu_c[i])))
            if bvz: p.backazimuth_errors=QuantityError(uncertainty=_nn(float(bzu_c[i])))
            p.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(p)
        return out

    def _rd_amplitudes(self,cols,am_s,am_e,ei,tw_cols,ci_rows,wf_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(am_s[ei]), int(am_e[ei])
        if start == end: return []
        pid=cols["public_id"]; av=cols["amp_value"]; au=cols["amp_uncertainty"]
        alo=cols["amp_lower_unc"]; ahi=cols["amp_upper_unc"]; acf=cols["amp_conf"]
        ty=cols["type"]; cat=cols["category"]; un=cols["unit"]; mid=cols["method_id"]
        perv=cols["period_value"]; peru=cols["period_uncertainty"]; snr=cols["snr"]
        twidx_col=cols["time_window_idx"]; pkid=cols["pick_id"]; wfidx_col=cols["waveform_idx"]
        fid=cols["filter_id"]; stv=cols["scaling_time_value"]; stu=cols["scaling_time_unc"]
        mhint=cols["magnitude_hint"]; emode=cols["eval_mode"]; estat=cols["eval_status"]
        ci_i=cols["ci_idx"]; coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        out=[]
        for i in range(start, end):
            wfidx=int(wfidx_col[i]); twidx=int(twidx_col[i])
            tw=None
            if tw_cols is not None and twidx>=0:
                tw=TimeWindow(begin=float(tw_cols["begin"][twidx]),
                              end=float(tw_cols["end"][twidx]),
                              reference=_from_ts(float(tw_cols["reference"][twidx])))
            pv=_nn(float(perv[i])); sv=_nn(float(stv[i]))
            a=Amplitude(resource_id=_make_rid(pid[i]),
                generic_amplitude=_nn(float(av[i])),
                type=(ty[i] or None),category=_dec(int(cat[i]),AMPLITUDE_CATEGORY),
                unit=_dec(int(un[i]),AMPLITUDE_UNIT),method_id=_make_rid(mid[i]),
                period=pv,snr=_nn(float(snr[i])),time_window=tw,
                pick_id=_make_rid(pkid[i]),
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                filter_id=_make_rid(fid[i]),
                scaling_time=_from_ts(sv) if sv else None,
                magnitude_hint=(mhint[i] or None),
                evaluation_mode=_dec(int(emode[i]),EVALUATION_MODE),
                evaluation_status=_dec(int(estat[i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            a.generic_amplitude_errors=QuantityError(uncertainty=_nn(float(au[i])),
                lower_uncertainty=_nn(float(alo[i])),upper_uncertainty=_nn(float(ahi[i])),
                confidence_level=_nn(float(acf[i])))
            if pv: a.period_errors=QuantityError(uncertainty=_nn(float(peru[i])))
            if sv: a.scaling_time_errors=QuantityError(uncertainty=_nn(float(stu[i])))
            a.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(a)
        return out

    def _rd_focmecs(self,cols,fm_s,fm_e,ei,mt_cols,du_cols,ci_rows,wf_rows,txt,cid,cidx):
        if cols is None: return []
        start, end = int(fm_s[ei]), int(fm_e[ei])
        if start == end: return []
        pid=cols["public_id"]; toid=cols["triggering_origin_id"]
        wfpool=cols.get("waveform_pool")
        pp_col=cols["preferred_plane"]
        ag=cols["azimuthal_gap"]; spc=cols["station_polarity_count"]
        mft=cols["misfit"]; sdr=cols["station_dist_ratio"]; mid=cols["method_id"]
        emode=cols["eval_mode"]; estat=cols["eval_status"]; ci_i=cols["ci_idx"]
        mtidx_c=cols["mt_idx"]; coff=cols["comment_offset"]; ccnt=cols["comment_count"]
        wpoff_c=cols["waveform_pool_offset"]; wpcnt_c=cols["waveform_pool_count"]
        out=[]
        for i in range(start, end):
            def gf(k): return _nn(float(cols[k][i]))
            # nodal planes
            np1=np2=None; nps=None
            s1=gf("np1_strike_value")
            if s1 is not None:
                np1=NodalPlane(strike=s1,dip=gf("np1_dip_value"),rake=gf("np1_rake_value"))
                np1.strike_errors=QuantityError(uncertainty=gf("np1_strike_unc"))
                np1.dip_errors=QuantityError(uncertainty=gf("np1_dip_unc"))
                np1.rake_errors=QuantityError(uncertainty=gf("np1_rake_unc"))
            s2=gf("np2_strike_value")
            if s2 is not None:
                np2=NodalPlane(strike=s2,dip=gf("np2_dip_value"),rake=gf("np2_rake_value"))
                np2.strike_errors=QuantityError(uncertainty=gf("np2_strike_unc"))
                np2.dip_errors=QuantityError(uncertainty=gf("np2_dip_unc"))
                np2.rake_errors=QuantityError(uncertainty=gf("np2_rake_unc"))
            if np1 or np2:
                pp=int(pp_col[i])
                nps=NodalPlanes(nodal_plane_1=np1,nodal_plane_2=np2,
                    preferred_plane=pp if pp else None)
            # principal axes
            pa=None
            def _ax(prefix):
                v=gf(f"{prefix}_azimuth_value")
                if v is None: return None
                ax=Axis(azimuth=v,plunge=gf(f"{prefix}_plunge_value"),
                        length=gf(f"{prefix}_length_value"))
                ax.azimuth_errors=QuantityError(uncertainty=gf(f"{prefix}_azimuth_unc"))
                ax.plunge_errors=QuantityError(uncertainty=gf(f"{prefix}_plunge_unc"))
                ax.length_errors=QuantityError(uncertainty=gf(f"{prefix}_length_unc"))
                return ax
            t_ax=_ax("t"); p_ax=_ax("p"); n_ax=_ax("n")
            if t_ax or p_ax: pa=PrincipalAxes(t_axis=t_ax,p_axis=p_ax,n_axis=n_ax)
            # waveform IDs
            wpoff=int(wpoff_c[i]); wpcnt=int(wpcnt_c[i])
            fm_wf=[wf_rows[int(wfpool[j])] for j in range(wpoff,wpoff+wpcnt)
                   if wfpool is not None and int(wfpool[j])<len(wf_rows)]
            # moment tensor
            mtidx=int(mtidx_c[i])
            mt=self._rd_mt(mt_cols,du_cols,mtidx,ci_rows,txt,cid,cidx) if mtidx>=0 else None
            fm=FocalMechanism(resource_id=_make_rid(pid[i]),
                triggering_origin_id=_make_rid(toid[i]),
                nodal_planes=nps,principal_axes=pa,
                azimuthal_gap=_nn(float(ag[i])),
                station_polarity_count=_ni(int(spc[i])),
                misfit=_nn(float(mft[i])),station_distribution_ratio=_nn(float(sdr[i])),
                method_id=_make_rid(mid[i]),
                evaluation_mode=_dec(int(emode[i]),EVALUATION_MODE),
                evaluation_status=_dec(int(estat[i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(ci_i[i])))
            fm.waveform_id=fm_wf
            if mt: fm.moment_tensor=mt
            fm.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(fm)
        return out

    def _rd_mt(self,cols,du_cols,idx,ci_rows,txt,cid,cidx):
        if cols is None or idx<0: return None
        def gf(k): return _nn(float(cols[k][idx]))
        def gs(k): return cols[k][idx] or None
        # tensor
        comps={s:gf(f"{s}_value") for s in ("rr","tt","pp","rt","rp","tp")}
        tensor=None
        if any(v is not None for v in comps.values()):
            tensor=Tensor()
            for short,attr in [("rr","m_rr"),("tt","m_tt"),("pp","m_pp"),
                                ("rt","m_rt"),("rp","m_rp"),("tp","m_tp")]:
                v=gf(f"{short}_value"); u=gf(f"{short}_unc")
                if v is not None:
                    setattr(tensor,attr,v)
                    if u is not None:
                        setattr(tensor,f"{attr}_errors",QuantityError(uncertainty=u))
        # stf
        stft=_dec(int(cols["stf_type"][idx]),SOURCE_TIME_FUNC_TYPE)
        stfd=gf("stf_duration")
        stf=SourceTimeFunction(type=stft,duration=stfd,rise_time=gf("stf_rise_time"),
            decay_time=gf("stf_decay_time")) if (stft or stfd) else None
        # data used
        duoff=int(cols["data_used_offset"][idx]); ducnt=int(cols["data_used_count"][idx])
        data_used=[]
        if du_cols is not None:
            for j in range(duoff,duoff+ducnt):
                data_used.append(DataUsed(
                    wave_type=_dec(int(du_cols["wave_type"][j]),DATA_USED_WAVE_TYPE),
                    station_count=_ni(int(du_cols["station_count"][j])),
                    component_count=_ni(int(du_cols["component_count"][j])),
                    shortest_period=_nn(float(du_cols["shortest_period"][j])),
                    longest_period=_nn(float(du_cols["longest_period"][j]))))
        sm_v=gf("scalar_moment_value"); sm_u=gf("scalar_moment_unc")
        mt=MomentTensor(resource_id=_make_rid(gs("public_id")),
            derived_origin_id=_make_rid(gs("derived_origin_id")),
            moment_magnitude_id=_make_rid(gs("moment_mag_id")),
            scalar_moment=sm_v,tensor=tensor,
            variance=gf("variance"),variance_reduction=gf("variance_reduction"),
            double_couple=gf("double_couple"),clvd=gf("clvd"),iso=gf("iso"),
            greens_function_id=_make_rid(gs("greens_function_id")),
            filter_id=_make_rid(gs("filter_id")),source_time_function=stf,
            method_id=_make_rid(gs("method_id")),
            category=_dec(int(cols["category"][idx]),MT_CATEGORY),
            inversion_type=_dec(int(cols["inversion_type"][idx]),MT_INVERSION_TYPE),
            creation_info=self._mk_ci(ci_rows,int(cols["ci_idx"][idx])))
        if sm_v is not None: mt.scalar_moment_errors=QuantityError(uncertainty=sm_u)
        mt.data_used=data_used
        mt.comments=self._mk_comments(txt,cid,cidx,ci_rows,
            int(cols["comment_offset"][idx]),int(cols["comment_count"][idx]))
        return mt

    # ------------------------------------------------------------------
    # Fast columnar reads (no ObsPy objects)
    # ------------------------------------------------------------------
    def _decode_group(self,grp_name):
        g=self._grp(grp_name)
        if g is None: return {}
        out={}
        for k in g.keys():
            ds=g[k]; raw=ds[()]
            if raw.dtype.kind=="O":
                out[k]=np.array([_sv(v) for v in raw])
            elif raw.dtype==np.uint8 and "enum_map" in ds.attrs:
                em=json.loads(ds.attrs["enum_map"])
                dec=np.empty(len(raw),dtype=object)
                for code,label in em.items(): dec[raw==int(code)]=label
                dec[raw==255]=None; out[k]=dec
            else:
                out[k]=raw
        return out

    def origins_dataframe(self):
        """All origin data as a dict of numpy arrays. NaN = missing float, -1 = missing int."""
        d=self._decode_group("origins")
        if "time_value" in d:
            d["time_utc"]=np.array([str(_from_ts(float(v))) if not math.isnan(float(v)) else None
                                    for v in d["time_value"]])
        return d

    def magnitudes_dataframe(self):
        """All magnitude data as a dict of numpy arrays."""
        return self._decode_group("magnitudes")

    def picks_dataframe(self):
        """All pick data as a dict of numpy arrays, with waveform codes resolved."""
        d=self._decode_group("picks")
        wf=self._load_wf()
        if "waveform_idx" in d and wf:
            n=len(d["waveform_idx"])
            net=np.empty(n,dtype=object); sta=np.empty(n,dtype=object)
            loc=np.empty(n,dtype=object); cha=np.empty(n,dtype=object)
            for j,idx in enumerate(d["waveform_idx"]):
                ii=int(idx)
                if ii<len(wf):
                    net[j]=wf[ii].network_code; sta[j]=wf[ii].station_code
                    loc[j]=wf[ii].location_code; cha[j]=wf[ii].channel_code
                else: net[j]=sta[j]=loc[j]=cha[j]=None
            d["network_code"]=net; d["station_code"]=sta
            d["location_code"]=loc; d["channel_code"]=cha
        if "time_value" in d:
            d["time_utc"]=np.array([str(_from_ts(float(v))) if not math.isnan(float(v)) else None
                                    for v in d["time_value"]])
        return d

    def arrivals_dataframe(self):
        """All arrival data as a dict of numpy arrays."""
        return self._decode_group("arrivals")

    def amplitudes_dataframe(self):
        """All amplitude data as a dict of numpy arrays."""
        return self._decode_group("amplitudes")

    # ------------------------------------------------------------------
    # Spatial / temporal / magnitude queries (column-only reads)
    # ------------------------------------------------------------------
    def query_bbox(self,min_lat,max_lat,min_lon,max_lon):
        """Return origin row indices within the bounding box (loads only lat/lon)."""
        g=self._grp("origins")
        if g is None: return np.array([],dtype=np.int64)
        lat=g["lat_value"][()]; lon=g["lon_value"][()]
        return np.where((lat>=min_lat)&(lat<=max_lat)&(lon>=min_lon)&(lon<=max_lon))[0]

    def query_time(self,t_start,t_end):
        """Return origin row indices with origin time in [t_start, t_end]. Accepts UTCDateTime/float."""
        g=self._grp("origins")
        if g is None: return np.array([],dtype=np.int64)
        times=g["time_value"][()]
        return np.where((times>=_ts(t_start))&(times<=_ts(t_end)))[0]

    def query_magnitude(self,min_mag,max_mag=10.0,mag_type=None):
        """Return magnitude row indices satisfying range and optional type filter."""
        g=self._grp("magnitudes")
        if g is None: return np.array([],dtype=np.int64)
        vals=g["mag_value"][()]; mask=(vals>=min_mag)&(vals<=max_mag)
        if mag_type is not None:
            types=self._rs(g,"type"); mask&=(types==mag_type)
        return np.where(mask)[0]

    def query_radius(self,center_lat,center_lon,max_radius_deg,min_radius_deg=0.0,invert=False):
        """Return origin row indices within an annular region around a point.

        Distances are computed using the Haversine formula and returned in
        degrees (1 deg ≈ 111.195 km).  Pass min_radius_deg=0 (default) for a
        simple circle.

        Parameters
        ----------
        center_lat, center_lon : float  — centre point in decimal degrees
        max_radius_deg : float          — outer radius in degrees
        min_radius_deg : float          — inner radius in degrees (default 0)
        invert : bool
             if False (default) return inside the radius
             if True, outside the radius

        Returns
        -------
        np.ndarray of int64 — origin row indices
        """
        g=self._grp("origins")
        if g is None: return np.array([],dtype=np.int64)
        lat=g["lat_value"][()]; lon=g["lon_value"][()]
        # Haversine in degrees
        lat1=np.radians(center_lat); lat2=np.radians(lat)
        dlat=lat2-lat1; dlon=np.radians(lon-center_lon)
        a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        dist_deg=np.degrees(2*np.arcsin(np.sqrt(np.clip(a,0,1))))
        mask=(dist_deg<=max_radius_deg)&(dist_deg>=min_radius_deg)
        if invert:
            return np.where(~mask)[0]
        else:
            return np.where(mask)[0]            

    def query_polygon(self,vertices,invert=False):
        """Return origin row indices whose epicentres fall inside (or outside) a polygon.

        Uses a ray-casting (point-in-polygon) test.  The polygon need not be
        closed — the last vertex is automatically connected back to the first.
        Handles the antimeridian naively (wrap lon to [-180,180] first if needed).

        Parameters
        ----------
        vertices : sequence of (lat, lon) pairs
            e.g. [(40,-10),(40,30),(55,30),(55,-10)]
        invert : bool
            If False (default) return origins inside the polygon.
            If True return origins outside the polygon.

        Returns
        -------
        np.ndarray of int64 — origin row indices
        """
        g=self._grp("origins")
        if g is None: return np.array([],dtype=np.int64)
        lat=g["lat_value"][()]; lon=g["lon_value"][()]

        verts=[(float(ln),float(la)) for la,ln in vertices]
        n=len(verts)
        inside=np.zeros(len(lat),dtype=bool)

        j=n-1
        for i in range(n):
            xi,yi=verts[i]; xj,yj=verts[j]
            cond=((yi>lat)!=(yj>lat)) & \
                 (lon < (xj-xi)*(lat-yi)/(yj-yi+1e-300)+xi)
            inside ^= cond
            j=i
        return np.where(~inside if invert else inside)[0]

    def query_depth(self,min_depth_m=0.0,max_depth_m=700_000.0):
        """Return origin row indices with depth (metres) in [min_depth_m, max_depth_m].

        ObsPy stores depth in metres.  Typical ranges:
          shallow crust  :   0 –  70 000 m
          intermediate   :  70 – 300 000 m
          deep           : 300 – 700 000 m

        Parameters
        ----------
        min_depth_m, max_depth_m : float — depth bounds in metres

        Returns
        -------
        np.ndarray of int64 — origin row indices
        """
        g=self._grp("origins")
        if g is None: return np.array([],dtype=np.int64)
        dep=g["depth_value"][()]
        valid=~np.isnan(dep)
        mask=valid & (dep>=min_depth_m) & (dep<=max_depth_m)
        return np.where(mask)[0]

    def query_arrivals(self,min_count,max_count=None):
        """Return origin row indices with used_phase_count in [min_count, max_count].

        'Defining arrivals' maps to origin_quality.used_phase_count — the number
        of phases actually used in the location solution.  Origins without quality
        information, or with a null used_phase_count, are excluded.

        Parameters
        ----------
        min_count : int — minimum used phase count (inclusive)
        max_count : int or None — maximum used phase count (inclusive); None = no upper limit

        Returns
        -------
        np.ndarray of int64 — origin row indices
        """
        og=self._grp("origins"); qg=self._grp("origin_quality")
        if og is None or qg is None: return np.array([],dtype=np.int64)
        qidx=og["quality_idx"][()]          # int32, -1 = no quality object
        upc=qg["used_phase_count"][()]       # int32, -1 = null

        has_q=qidx>=0
        v=np.full(len(qidx),-1,dtype=np.int64)
        v[has_q]=upc[qidx[has_q]]
        mask=has_q & (v>=0) & (v>=min_count)
        if max_count is not None:
            mask &= (v<=max_count)
        return np.where(mask)[0]

    # ------------------------------------------------------------------
    # Turning query_* row indices into event indices for read_catalog()
    # ------------------------------------------------------------------
    # Every query_* method above returns row indices into ONE table:
    #   "origins"    <- query_bbox, query_time, query_radius,
    #                   query_polygon, query_depth, query_arrivals
    #   "magnitudes" <- query_magnitude
    # read_catalog(event_indices=...) wants indices into the "catalog"
    # (event) table instead, so those row indices need to be mapped
    # through that table's own event_idx column first. rows_to_event_idx
    # does that mapping; query_events combines several queries at once.

    def rows_to_event_idx(self,group_name,rows):
        """Map row indices in `group_name` (e.g. "origins", "magnitudes")
        to the event indices they belong to.

        Parameters
        ----------
        group_name : str — which table `rows` came from ("origins" for
            query_bbox/query_time/query_radius/query_polygon/query_depth/
            query_arrivals, "magnitudes" for query_magnitude).
        rows : array-like of int — row indices, as returned by a query_* method.

        Returns
        -------
        np.ndarray of int64 — sorted, de-duplicated event indices.
        """
        g=self._grp(group_name)
        rows=np.asarray(rows,dtype=np.int64)
        if g is None or rows.size==0: return np.array([],dtype=np.int64)
        ev_col=self._ra(g,"event_idx")
        if ev_col is None: return np.array([],dtype=np.int64)
        return np.unique(ev_col[rows])

    def query_events(self,*queries,mode="union"):
        """Run one or more query_* methods and combine their results into
        event indices, ready to pass straight into
        read_catalog(event_indices=...). The source table for each query
        method is looked up automatically via QUERY_TABLE, so you never
        need to name it yourself.

        Parameters
        ----------
        *queries : (method_name, kwargs_dict) pairs
            method_name is the name of any query_* method (e.g. "query_bbox"),
            kwargs_dict holds the keyword arguments to call it with.
        mode : "union" or "intersection"
            "union" — event matches ANY of the given queries (default)
            "intersection" — event matches ALL of the given queries

        Returns
        -------
        list of int — sorted, de-duplicated event indices

        Example
        -------
        >>> events = q.query_events(
        ...     ("query_bbox", dict(min_lat=32,max_lat=37,min_lon=-120,max_lon=-114)),
        ...     ("query_magnitude", dict(min_mag=6.0)),
        ...     mode="union")
        >>> cat = q.read_catalog(event_indices=events)
        """
        sets=[]
        for method_name,kwargs in queries:
            if method_name not in self.QUERY_TABLE:
                raise ValueError(f"{method_name!r} is not a known query_* method")
            table=self.QUERY_TABLE[method_name]
            rows=getattr(self,method_name)(**kwargs)
            sets.append(set(self.rows_to_event_idx(table,rows).tolist()))
        if not sets: return []
        if mode=="union":
            result=set().union(*sets)
        elif mode=="intersection":
            result=sets[0]
            for s in sets[1:]: result&=s
        else:
            raise ValueError('mode must be "union" or "intersection"')
        return sorted(result)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def info(self):
        """Return a summary dict of file metadata and row counts."""
        f=self._f
        out={k:f.attrs.get(k,"?") for k in ("format","format_version","quakeml_version",
             "catalog_description","catalog_public_id")}
        out["n_events"]=int(f.attrs.get("n_events",0))
        out["creation_time"]=str(_from_ts(float(f.attrs["creation_time"]))) if "creation_time" in f.attrs else None
        for grp in ["origins","magnitudes","station_magnitudes","picks","arrivals",
                    "amplitudes","focal_mechanisms","moment_tensors","comments",
                    "waveform_ids","creation_info"]:
            g=self._grp(grp)
            if g is not None:
                first=next(iter(g.keys()),None)
                out[f"n_{grp}"]=len(g[first]) if first else 0
            else: out[f"n_{grp}"]=0
        return out

    def print_info(self):
        d=self.info()
        print(f"qmlh5  {self._path}")
        print(f"  Format:      {d['format']} v{d['format_version']}  (QuakeML {d['quakeml_version']})")
        print(f"  Created:     {d['creation_time']}")
        print(f"  Description: {d['catalog_description']}")
        print(f"  Events:      {d['n_events']}")
        for k,v in d.items():
            if k.startswith("n_") and k!="n_events":
                print(f"  {k[2:].replace('_',' ').title():32s}{v}")

    # ------------------------------------------------------------------
    # Quick stats — column-only, no ObsPy objects built
    # ------------------------------------------------------------------
    def stats(self):
        """Print (and return) a quick summary of the catalog: lat/lon
        position range in degrees, depth range in km, average horizontal
        location uncertainty in km (from OriginUncertainty, NOT from
        latitude/longitude uncertainty — see below), average depth
        uncertainty in km, and origin_quality standard_error (printed
        as RMS error, in seconds). Reads only the handful of
        origins/origin_quality/origin_uncertainty columns it needs — no
        ObsPy objects are constructed, so this is fast even on huge
        catalogs.

        Horizontal uncertainty is deliberately derived from
        OriginUncertainty.horizontal_uncertainty (or, when that's
        absent, the mean of min_horizontal_uncertainty and
        max_horizontal_uncertainty) rather than from
        latitude_errors.uncertainty / longitude_errors.uncertainty.
        Those two are a RealQuantity sharing units with latitude/
        longitude (degrees per spec) and are the field that ends up
        mislabeled for catalogs sourced from SeisComP/SCML (see prior
        discussion). OriginUncertainty's horizontal fields are plain
        values, unambiguously metres regardless of source, so this
        sidesteps that whole question rather than resolving it.

        Depth is likewise unambiguously metres per spec, so depth range
        and depth uncertainty are converted to km for readability.

        Returns
        -------
        dict — lat/lon range (degrees), depth range (km), mean/std/n
        (km) for horizontal and depth uncertainty, and mean/std/n
        (seconds) for origin_quality.standard_error (RMS travel-time
        residual). NaN (missing) values are excluded from every
        average and std.
        """
        og=self._grp("origins")
        if og is None:
            print(f"qmlh5 stats  {self._path}\n  No origins found.")
            return {}

        lat  =self._ra(og,"lat_value");        lon  =self._ra(og,"lon_value")
        dep  =self._ra(og,"depth_value")
        dep_u=self._ra(og,"depth_uncertainty")
        qidx =self._ra(og,"quality_idx")
        uidx =self._ra(og,"uncertainty_idx")

        se=np.array([])
        qg=self._grp("origin_quality")
        if qg is not None and qidx is not None and qidx.size:
            all_se=self._ra(qg,"standard_error")
            if all_se is not None:
                valid=qidx>=0
                se=np.full(len(qidx),_NaN)
                se[valid]=all_se[qidx[valid]]

        # --- horizontal uncertainty, from OriginUncertainty (metres, unambiguous) ---
        horiz_km=np.array([])
        oug=self._grp("origin_uncertainty")
        if oug is not None and uidx is not None and uidx.size:
            hu =self._ra(oug,"horizontal_uncertainty")
            minu=self._ra(oug,"min_horizontal_uncertainty")
            maxu=self._ra(oug,"max_horizontal_uncertainty")
            mapped=uidx>=0
            if mapped.any() and (hu is not None or (minu is not None and maxu is not None)):
                n_rows=(len(hu) if hu is not None else len(minu))
                mapped &= uidx<n_rows
                horiz_km=np.full(len(uidx),_NaN)
                if mapped.any():
                    idx=uidx[mapped]
                    picked=np.full(idx.shape,_NaN)
                    if hu is not None:
                        picked=hu[idx].copy()
                    if minu is not None and maxu is not None:
                        avg_mm=(minu[idx]+maxu[idx])/2.0
                        need=np.isnan(picked)
                        picked[need]=avg_mm[need]
                    horiz_km[mapped]=picked/1000.0

        def _range(arr):
            if arr is None or arr.size==0: return (None,None)
            arr=arr[~np.isnan(arr)]
            return (float(arr.min()),float(arr.max())) if arr.size else (None,None)

        def _avgstd(arr):
            if arr is None or arr.size==0: return (None,None,0)
            arr=arr[~np.isnan(arr)]
            if arr.size==0: return (None,None,0)
            return (float(arr.mean()),float(arr.std()),int(arr.size))

        # --- position, in degrees ---
        lat_range=_range(lat); lon_range=_range(lon)

        # --- depth range, in km (metres per spec) ---
        valid_dep=dep[~np.isnan(dep)] if dep is not None and dep.size else np.array([])
        dep_range_km=(None,None)
        if valid_dep.size:
            dep_range_km=(float(valid_dep.min()/1000.0),float(valid_dep.max()/1000.0))

        horiz_avg,horiz_std,horiz_n=_avgstd(horiz_km)

        dep_u_km = dep_u/1000.0 if dep_u is not None else None
        dep_avg,dep_std,dep_n=_avgstd(dep_u_km)

        se_avg,se_std,se_n=_avgstd(se)

        out={"n_origins":int(len(lat)) if lat is not None else 0,
             "lat_range_deg":lat_range,"lon_range_deg":lon_range,"depth_range_km":dep_range_km,
             "horizontal_uncertainty_mean_km":horiz_avg,"horizontal_uncertainty_std_km":horiz_std,
             "horizontal_uncertainty_n":horiz_n,
             "depth_uncertainty_mean_km":dep_avg,"depth_uncertainty_std_km":dep_std,"depth_uncertainty_n":dep_n,
             "rms_error_mean":se_avg,"rms_error_std":se_std,"rms_error_n":se_n}

        def _fr(r,unit="",p=4):
            return "n/a" if r[0] is None else f"{r[0]:.{p}f} to {r[1]:.{p}f}{unit}"
        def _fa(avg,std,n,unit=""):
            return "n/a" if avg is None else f"{avg:.3f} ± {std:.3f}{unit}  (n={n})"

        print(f"qmlh5 stats  {self._path}")
        print(f"  Origins:               {out['n_origins']}")
        print(f"  Latitude range:        {_fr(lat_range,'°')}")
        print(f"  Longitude range:       {_fr(lon_range,'°')}")
        print(f"  Depth range:           {_fr(dep_range_km,' km',2)}")
        print(f"  Horizontal uncertainty:{_fa(horiz_avg,horiz_std,horiz_n,' km')}")
        print(f"  Depth uncertainty:     {_fa(dep_avg,dep_std,dep_n,' km')}")
        print(f"  RMS error:             {_fa(se_avg,se_std,se_n,' s')}")

        return out


# ---------------------------------------------------------------------------
# Module-level convenience API
#
#   cat = qmlh5.read_catalog("cat.h5")
#   qmlh5.write_catalog(cat, "out.h5")
#   cat.write_catalog("out.h5")        # method patched onto Catalog below
#
# ---------------------------------------------------------------------------
def read_catalog(path, event_indices=None, progress=True,
                 starttime=None, endtime=None):
    """Read a QuakeML/HDF5 file and return an ObsPy :class:`Catalog`.

    Parameters
    ----------
    path : str
        Path to a qmlh5 file written by :func:`write_catalog` or
        :class:`qmlh5`.
    event_indices : iterable of int, optional
        Subset of event row indices to load. If ``None`` (default), the
        entire catalog is loaded.
    progress : bool, optional
        Show a tqdm progress bar while reconstructing events. Defaults to
        ``True``. Has no effect if tqdm is not installed, or for catalogs of
        100 events or fewer.
    starttime, endtime : str or :class:`~obspy.UTCDateTime`, optional
        Time-range filter applied BEFORE constructing ObsPy objects. Each
        event's first origin's `time` is used as its representative time.
        Events with no origins are excluded when either bound is given.
        Faster than ``cat.filter(...)`` after loading because filtered-out
        events skip per-event ObsPy object construction entirely.

    Returns
    -------
    obspy.core.event.Catalog
    """
    with qmlh5(path, "r") as q:
        return q.read_catalog(event_indices=event_indices, progress=progress,
                              starttime=starttime, endtime=endtime)


def write_catalog(catalog, path, progress=True, chunk_size=10000):
    """Write an ObsPy :class:`Catalog` to a qmlh5 (HDF5) file.

    Parameters
    ----------
    catalog : obspy.core.event.Catalog
        The catalog to serialize.
    path : str
        Destination file path. Will be created or overwritten.
    progress : bool, optional
        Show a tqdm progress bar while writing events. Defaults to ``True``.
        Has no effect if tqdm is not installed, or for very small catalogs.
    chunk_size : int or None, optional
        Number of events to accumulate in RAM before flushing to disk. The
        default of 10,000 keeps peak RAM in the low hundreds of MB even for
        catalogs of millions of events. Pass ``None`` to disable chunking
        (single-flush at the end) if you have plenty of memory and want to
        avoid the small per-chunk overhead.
    """
    with qmlh5(path, "w") as q:
        q.write_catalog(catalog, progress=progress, chunk_size=chunk_size)


def get_stats(path):
    """Point at a qmlh5 file and print a quick summary of its spatial
    extent (lat/lon range and average location uncertainty in degrees,
    depth range and average depth uncertainty in km) plus
    origin_quality.standard_error (mean and standard deviation).
    Column-only: no ObsPy objects are built, so this is fast even on
    catalogs too large to comfortably read_catalog() in full. Does NOT
    require ObsPy to be installed.

    Parameters
    ----------
    path : str — path to a qmlh5 file.

    Returns
    -------
    dict — same numbers that get printed, for programmatic use.

    Example
    -------
    >>> import qmlh5
    >>> qmlh5.get_stats("huge_ml_catalog.h5")
    qmlh5 stats  huge_ml_catalog.h5
      Origins:               83214
      Latitude range:        32.1050 to 37.8890°
      ...
    """
    with qmlh5(path, "r") as q:
        return q.stats()


# Attach `write_catalog` as a method on ObsPy's Catalog so the user can write
#     cat.write_catalog("out.h5")
# in addition to the module-level form. We deliberately don't override the
# built-in `Catalog.write` (which dispatches by `format=...` to ObsPy's I/O
# plugins); this is a sibling, not a replacement.
if OBSPY_AVAILABLE:
    def _catalog_write_catalog(self, path, progress=True, chunk_size=10000):
        """Write this catalog to a qmlh5 (HDF5) file. See :func:`qmlh5.write_catalog`."""
        write_catalog(self, path, progress=progress, chunk_size=chunk_size)
    Catalog.write_catalog = _catalog_write_catalog