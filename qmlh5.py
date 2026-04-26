"""
VERSION 1.0


qmlh5.py — Binary HDF5 storage for QuakeML (1.2 hardwired) earthquake catalogs.

All objects stored as flat columnar arrays. Cross-object links use integer
indices. Enums stored as uint8 with JSON enum_map attribute. Timestamps as
float64 Unix seconds (NaN = missing). Signed int sentinel: -1 = absent.

Public API
----------
    cat = qmlh5.read_catalog("incat.h5")        # module-level read
    qmlh5.write_catalog(cat, "outcat.h5")       # module-level write
    cat.write_catalog("outcat.h5")              # method on ObsPy Catalog

The lower-level :class:`QMLH5` class supports column-oriented queries
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
        key=(_rid(getattr(wf,"network_code","")) if not hasattr(wf,"network_code")
             else (wf.network_code or ""),
             wf.station_code or "" if hasattr(wf,"station_code") else "",
             wf.location_code or "" if hasattr(wf,"location_code") else "",
             wf.channel_code  or "" if hasattr(wf,"channel_code")  else "",
             _rid(getattr(wf,"resource_uri",None)))
        # simpler:
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
    def __init__(self): self.text,self.id,self.ci=[],[],[]
    def add(self,comments,ci_tbl):
        off=len(self.text)
        for c in (comments or []):
            self.text.append(c.text or "")
            self.id.append(_rid(getattr(c,"resource_id",None)))
            self.ci.append(int(ci_tbl.add(c.creation_info)))
        return off, len(self.text)-off
    def write(self,grp,c,cs):
        if not self.text: return
        grp.create_dataset("text",data=np.array(self.text,dtype=object),dtype=_VLEN,**cs)
        grp.create_dataset("id",  data=np.array(self.id,  dtype=object),dtype=_VLEN,**cs)
        grp.create_dataset("ci_idx",data=np.array(self.ci,dtype=np.int32),**c)

# ---------------------------------------------------------------------------
# QMLH5 class — write path
# ---------------------------------------------------------------------------

class QMLH5:
    """
    Read/write QuakeML 1.2 catalogs as columnar HDF5.

    with QMLH5("cat.h5","w") as q: q.write_catalog(cat)
    with QMLH5("cat.h5")     as q: cat = q.read_catalog()
    with QMLH5("cat.h5")     as q: d = q.origins_dataframe()
    """
    FORMAT="qmlh5"; FORMAT_VERSION="1.0"; QUAKEML_VERSION="1.2"; CHUNK=1024
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
    def _sa(self,lst): return np.array(lst,dtype=object)  # string array

    # ------------------------------------------------------------------
    def write_catalog(self,catalog):
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

        # ---- iterate events ----
        for eidx,e in enumerate(events):
            descs=getattr(e,"event_descriptions",[]) or []
            doff=len(ed["text"])
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
                ctoff=len(ct["yrv"])
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
                ctcnt=len(ct["yrv"])-ctoff

                # arrivals
                aroff=len(ar["pid"])
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
                arcnt=len(ar["pid"])-aroff

                ocoff,occnt=cp.add(o.comments,ci)

                # quality
                if o.quality is not None:
                    q=o.quality; qidx=len(oq["apc"])
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
                        c_e=u.confidence_ellipsoid; elidx=len(ce["sma"])
                        ce["sma"].append(_of(c_e.semi_major_axis_length))
                        ce["smi"].append(_of(c_e.semi_minor_axis_length))
                        ce["smia"].append(_of(c_e.semi_intermediate_axis_length))
                        ce["mpl"].append(_of(c_e.major_axis_plunge))
                        ce["maz"].append(_of(c_e.major_axis_azimuth))
                        ce["mro"].append(_of(c_e.major_axis_rotation))
                    else: elidx=-1
                    uidx=len(ou["hu"])
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
                soff=len(sc["smid"])
                for s in (m.station_magnitude_contributions or []):
                    sc["smid"].append(_rid(s.station_magnitude_id))
                    sc["res"].append(_of(s.residual)); sc["wt"].append(_of(s.weight))
                scnt=len(sc["smid"])-soff
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
                    twidx=len(tw["beg"])
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
                wpoff=len(fmwp)
                for wfid in (getattr(f_m,"waveform_id",[]) or []):
                    fmwp.append(int(wf.add(wfid)))
                wpcnt=len(fmwp)-wpoff

                # moment tensor
                mt_obj=getattr(f_m,"moment_tensor",None)
                if isinstance(mt_obj,list): mt_obj=mt_obj[0] if mt_obj else None
                mtidx=-1
                if mt_obj is not None:
                    mtidx=len(mt["pid"])
                    mtcoff,mtccnt=cp.add(mt_obj.comments,ci)
                    duoff=len(du["wt"])
                    for d_u in (mt_obj.data_used or []):
                        du["wt"].append(_enc(d_u.wave_type,_R_DU_WAVE))
                        du["sc"].append(_oi(d_u.station_count))
                        du["cc"].append(_oi(d_u.component_count))
                        du["sp"].append(_of(d_u.shortest_period))
                        du["lp"].append(_of(d_u.longest_period))
                    ducnt=len(du["wt"])-duoff
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

        # ---- flush to HDF5 ----
        def _g(n): return f.require_group(n)
        def _f64(lst): return np.array(lst,dtype=np.float64)
        def _u32(lst): return np.array(lst,dtype=np.uint32)
        def _i32(lst): return np.array(lst,dtype=np.int32)
        def _u8(lst):  return np.array(lst,dtype=np.uint8)
        def _i8(lst):  return np.array(lst,dtype=np.int8)

        if ev["pid"]:
            g=_g("catalog")
            self._ds(g,"public_id",self._sa(ev["pid"]))
            self._ds(g,"preferred_origin_id",self._sa(ev["po"]))
            self._ds(g,"preferred_magnitude_id",self._sa(ev["pm"]))
            self._ds(g,"preferred_focmec_id",self._sa(ev["pf"]))
            self._ds(g,"event_type",_u8(ev["etype"]),enum_map=EVENT_TYPE)
            self._ds(g,"event_type_certainty",_u8(ev["ecert"]),enum_map=EVENT_TYPE_CERTAINTY)
            self._ds(g,"ci_idx",_i32(ev["ci"]))
            self._ds(g,"desc_offset",_u32(ev["doff"])); self._ds(g,"desc_count",_u32(ev["dcnt"]))
            self._ds(g,"comment_offset",_u32(ev["coff"])); self._ds(g,"comment_count",_u32(ev["ccnt"]))

        if ed["text"]:
            g=_g("event_descriptions")
            self._ds(g,"text",self._sa(ed["text"]))
            self._ds(g,"type",_u8(ed["type"]),enum_map=EVENT_DESC_TYPE)

        if or_["pid"]:
            g=_g("origins")
            self._ds(g,"public_id",self._sa(or_["pid"]))
            self._ds(g,"event_idx",_u32(or_["eidx"]))
            self._ds(g,"time_value",_f64(or_["tv"])); self._ds(g,"time_uncertainty",_f64(or_["tu"]))
            self._ds(g,"time_lower_unc",_f64(or_["tlo"])); self._ds(g,"time_upper_unc",_f64(or_["thi"]))
            self._ds(g,"time_conf",_f64(or_["tcf"]))
            self._ds(g,"lat_value",_f64(or_["lav"])); self._ds(g,"lat_uncertainty",_f64(or_["lau"]))
            self._ds(g,"lat_lower_unc",_f64(or_["lalo"])); self._ds(g,"lat_upper_unc",_f64(or_["lahi"]))
            self._ds(g,"lat_conf",_f64(or_["lacf"]))
            self._ds(g,"lon_value",_f64(or_["lov"])); self._ds(g,"lon_uncertainty",_f64(or_["lou"]))
            self._ds(g,"lon_lower_unc",_f64(or_["lolo"])); self._ds(g,"lon_upper_unc",_f64(or_["lohi"]))
            self._ds(g,"lon_conf",_f64(or_["locf"]))
            self._ds(g,"depth_value",_f64(or_["dpv"])); self._ds(g,"depth_uncertainty",_f64(or_["dpu"]))
            self._ds(g,"depth_lower_unc",_f64(or_["dplo"])); self._ds(g,"depth_upper_unc",_f64(or_["dphi"]))
            self._ds(g,"depth_conf",_f64(or_["dpcf"]))
            self._ds(g,"depth_type",_u8(or_["dtype"]),enum_map=ORIGIN_DEPTH_TYPE)
            self._ds(g,"time_fixed",_i8(or_["tfx"])); self._ds(g,"epicenter_fixed",_i8(or_["epfx"]))
            self._ds(g,"ref_system_id",self._sa(or_["rsid"]))
            self._ds(g,"method_id",self._sa(or_["mid"]))
            self._ds(g,"earth_model_id",self._sa(or_["emid"]))
            self._ds(g,"type",_u8(or_["otype"]),enum_map=ORIGIN_TYPE)
            self._ds(g,"region",self._sa(or_["reg"]))
            self._ds(g,"eval_mode",_u8(or_["emode"]),enum_map=EVALUATION_MODE)
            self._ds(g,"eval_status",_u8(or_["estat"]),enum_map=EVALUATION_STATUS)
            self._ds(g,"ci_idx",_i32(or_["ci"]))
            self._ds(g,"quality_idx",_i32(or_["qidx"])); self._ds(g,"uncertainty_idx",_i32(or_["uidx"]))
            self._ds(g,"arrival_offset",_u32(or_["aoff"])); self._ds(g,"arrival_count",_u32(or_["acnt"]))
            self._ds(g,"comment_offset",_u32(or_["coff"])); self._ds(g,"comment_count",_u32(or_["ccnt"]))
            self._ds(g,"comptime_offset",_u32(or_["ctoff"])); self._ds(g,"comptime_count",_u32(or_["ctcnt"]))

        if oq["apc"]:
            g=_g("origin_quality")
            self._ds(g,"assoc_phase_count",_i32(oq["apc"])); self._ds(g,"used_phase_count",_i32(oq["upc"]))
            self._ds(g,"assoc_sta_count",_i32(oq["asc"])); self._ds(g,"used_sta_count",_i32(oq["usc"]))
            self._ds(g,"depth_phase_count",_i32(oq["dpc"])); self._ds(g,"standard_error",_f64(oq["se"]))
            self._ds(g,"azimuthal_gap",_f64(oq["ag"])); self._ds(g,"sec_azimuthal_gap",_f64(oq["sag"]))
            self._ds(g,"ground_truth_level",self._sa(oq["gtl"]))
            self._ds(g,"minimum_distance",_f64(oq["mind"])); self._ds(g,"maximum_distance",_f64(oq["maxd"]))
            self._ds(g,"median_distance",_f64(oq["medd"]))

        if ou["hu"]:
            g=_g("origin_uncertainty")
            self._ds(g,"horizontal_uncertainty",_f64(ou["hu"]))
            self._ds(g,"min_horizontal_uncertainty",_f64(ou["minu"]))
            self._ds(g,"max_horizontal_uncertainty",_f64(ou["maxu"]))
            self._ds(g,"azimuth_max_horiz_unc",_f64(ou["az"]))
            self._ds(g,"preferred_description",_u8(ou["pd"]),enum_map=ORIGIN_UNCERTAINTY_DESC)
            self._ds(g,"confidence_level",_f64(ou["cl"]))
            self._ds(g,"ellipsoid_idx",_i32(ou["eidx"]))

        if ce["sma"]:
            g=_g("confidence_ellipsoids")
            self._ds(g,"semi_major_axis_length",_f64(ce["sma"]))
            self._ds(g,"semi_minor_axis_length",_f64(ce["smi"]))
            self._ds(g,"semi_intermediate_axis_length",_f64(ce["smia"]))
            self._ds(g,"major_axis_plunge",_f64(ce["mpl"]))
            self._ds(g,"major_axis_azimuth",_f64(ce["maz"]))
            self._ds(g,"major_axis_rotation",_f64(ce["mro"]))

        if ct["yrv"]:
            g=_g("composite_times")
            # signed-int columns: value + uncertainty + lower + upper (-1 = absent)
            for k,n in [("yrv","year_value"),("yru","year_unc"),
                        ("yrlo","year_lower_unc"),("yrhi","year_upper_unc"),
                        ("mov","month_value"),("mou","month_unc"),
                        ("molo","month_lower_unc"),("mohi","month_upper_unc"),
                        ("dyv","day_value"),("dyu","day_unc"),
                        ("dylo","day_lower_unc"),("dyhi","day_upper_unc"),
                        ("hrv","hour_value"),("hru","hour_unc"),
                        ("hrlo","hour_lower_unc"),("hrhi","hour_upper_unc"),
                        ("miv","minute_value"),("miu","minute_unc"),
                        ("milo","minute_lower_unc"),("mihi","minute_upper_unc")]:
                self._ds(g,n,_i32(ct[k]))
            # float columns: confidence_level for each int field, plus all four
            # error sub-fields for `second` (which is itself a float)
            for k,n in [("yrcf","year_conf"),("mocf","month_conf"),("dycf","day_conf"),
                        ("hrcf","hour_conf"),("micf","minute_conf"),
                        ("sev","second_value"),("seu","second_unc"),
                        ("selo","second_lower_unc"),("sehi","second_upper_unc"),
                        ("secf","second_conf")]:
                self._ds(g,n,_f64(ct[k]))

        if ar["pid"]:
            g=_g("arrivals")
            self._ds(g,"public_id",self._sa(ar["pid"])); self._ds(g,"pick_id",self._sa(ar["pkid"]))
            self._ds(g,"phase",self._sa(ar["ph"]))
            self._ds(g,"time_correction",_f64(ar["tc"])); self._ds(g,"azimuth",_f64(ar["az"]))
            self._ds(g,"distance",_f64(ar["dist"]))
            self._ds(g,"takeoff_value",_f64(ar["tov"])); self._ds(g,"takeoff_uncertainty",_f64(ar["tou"]))
            self._ds(g,"time_residual",_f64(ar["tr"])); self._ds(g,"hslow_residual",_f64(ar["hsr"]))
            self._ds(g,"baz_residual",_f64(ar["br"])); self._ds(g,"time_weight",_f64(ar["tw"]))
            self._ds(g,"hslow_weight",_f64(ar["hsw"])); self._ds(g,"baz_weight",_f64(ar["bw"]))
            self._ds(g,"earth_model_id",self._sa(ar["emid"]))
            self._ds(g,"ci_idx",_i32(ar["ci"]))
            self._ds(g,"comment_offset",_u32(ar["coff"])); self._ds(g,"comment_count",_u32(ar["ccnt"]))

        if mg["pid"]:
            g=_g("magnitudes")
            self._ds(g,"public_id",self._sa(mg["pid"])); self._ds(g,"event_idx",_u32(mg["eidx"]))
            self._ds(g,"mag_value",_f64(mg["val"])); self._ds(g,"mag_uncertainty",_f64(mg["unc"]))
            self._ds(g,"mag_lower_unc",_f64(mg["lo"])); self._ds(g,"mag_upper_unc",_f64(mg["hi"]))
            self._ds(g,"mag_conf",_f64(mg["cf"])); self._ds(g,"type",self._sa(mg["type"]))
            self._ds(g,"origin_id",self._sa(mg["orig"])); self._ds(g,"method_id",self._sa(mg["mid"]))
            self._ds(g,"station_count",_i32(mg["scnt"])); self._ds(g,"azimuthal_gap",_f64(mg["ag"]))
            self._ds(g,"eval_mode",_u8(mg["emode"]),enum_map=EVALUATION_MODE)
            self._ds(g,"eval_status",_u8(mg["estat"]),enum_map=EVALUATION_STATUS)
            self._ds(g,"ci_idx",_i32(mg["ci"]))
            self._ds(g,"contrib_offset",_u32(mg["soff"])); self._ds(g,"contrib_count",_u32(mg["scnt2"]))
            self._ds(g,"comment_offset",_u32(mg["coff"])); self._ds(g,"comment_count",_u32(mg["ccnt"]))

        if sm["pid"]:
            g=_g("station_magnitudes")
            self._ds(g,"public_id",self._sa(sm["pid"])); self._ds(g,"event_idx",_u32(sm["eidx"]))
            self._ds(g,"origin_id",self._sa(sm["orig"]))
            self._ds(g,"mag_value",_f64(sm["val"])); self._ds(g,"mag_uncertainty",_f64(sm["unc"]))
            self._ds(g,"mag_lower_unc",_f64(sm["lo"])); self._ds(g,"mag_upper_unc",_f64(sm["hi"]))
            self._ds(g,"type",self._sa(sm["type"])); self._ds(g,"amplitude_id",self._sa(sm["amid"]))
            self._ds(g,"method_id",self._sa(sm["mid"])); self._ds(g,"waveform_idx",_u32(sm["wfidx"]))
            self._ds(g,"ci_idx",_i32(sm["ci"]))
            self._ds(g,"comment_offset",_u32(sm["coff"])); self._ds(g,"comment_count",_u32(sm["ccnt"]))

        if sc["smid"]:
            g=_g("station_mag_contributions")
            self._ds(g,"station_magnitude_id",self._sa(sc["smid"]))
            self._ds(g,"residual",_f64(sc["res"])); self._ds(g,"weight",_f64(sc["wt"]))

        if pk["pid"]:
            g=_g("picks")
            self._ds(g,"public_id",self._sa(pk["pid"])); self._ds(g,"event_idx",_u32(pk["eidx"]))
            self._ds(g,"time_value",_f64(pk["tv"])); self._ds(g,"time_uncertainty",_f64(pk["tu"]))
            self._ds(g,"time_lower_unc",_f64(pk["tlo"])); self._ds(g,"time_upper_unc",_f64(pk["thi"]))
            self._ds(g,"time_conf",_f64(pk["tcf"])); self._ds(g,"waveform_idx",_u32(pk["wfidx"]))
            self._ds(g,"filter_id",self._sa(pk["fid"])); self._ds(g,"method_id",self._sa(pk["mid"]))
            self._ds(g,"hslow_value",_f64(pk["hsv"])); self._ds(g,"hslow_uncertainty",_f64(pk["hsu"]))
            self._ds(g,"baz_value",_f64(pk["bzv"])); self._ds(g,"baz_uncertainty",_f64(pk["bzu"]))
            self._ds(g,"slowness_method_id",self._sa(pk["smid"]))
            self._ds(g,"onset",_u8(pk["onset"]),enum_map=PICK_ONSET)
            self._ds(g,"phase_hint",self._sa(pk["ph"]))
            self._ds(g,"polarity",_u8(pk["pol"]),enum_map=PICK_POLARITY)
            self._ds(g,"eval_mode",_u8(pk["emode"]),enum_map=EVALUATION_MODE)
            self._ds(g,"eval_status",_u8(pk["estat"]),enum_map=EVALUATION_STATUS)
            self._ds(g,"ci_idx",_i32(pk["ci"]))
            self._ds(g,"comment_offset",_u32(pk["coff"])); self._ds(g,"comment_count",_u32(pk["ccnt"]))

        if am["pid"]:
            g=_g("amplitudes")
            self._ds(g,"public_id",self._sa(am["pid"])); self._ds(g,"event_idx",_u32(am["eidx"]))
            self._ds(g,"amp_value",_f64(am["val"])); self._ds(g,"amp_uncertainty",_f64(am["unc"]))
            self._ds(g,"amp_lower_unc",_f64(am["lo"])); self._ds(g,"amp_upper_unc",_f64(am["hi"]))
            self._ds(g,"amp_conf",_f64(am["cf"])); self._ds(g,"type",self._sa(am["type"]))
            self._ds(g,"category",_u8(am["cat"]),enum_map=AMPLITUDE_CATEGORY)
            self._ds(g,"unit",_u8(am["unit"]),enum_map=AMPLITUDE_UNIT)
            self._ds(g,"method_id",self._sa(am["mid"]))
            self._ds(g,"period_value",_f64(am["perv"])); self._ds(g,"period_uncertainty",_f64(am["peru"]))
            self._ds(g,"snr",_f64(am["snr"])); self._ds(g,"time_window_idx",_i32(am["twidx"]))
            self._ds(g,"pick_id",self._sa(am["pkid"])); self._ds(g,"waveform_idx",_u32(am["wfidx"]))
            self._ds(g,"filter_id",self._sa(am["fid"]))
            self._ds(g,"scaling_time_value",_f64(am["stv"])); self._ds(g,"scaling_time_unc",_f64(am["stu"]))
            self._ds(g,"magnitude_hint",self._sa(am["mhint"]))
            self._ds(g,"eval_mode",_u8(am["emode"]),enum_map=EVALUATION_MODE)
            self._ds(g,"eval_status",_u8(am["estat"]),enum_map=EVALUATION_STATUS)
            self._ds(g,"ci_idx",_i32(am["ci"]))
            self._ds(g,"comment_offset",_u32(am["coff"])); self._ds(g,"comment_count",_u32(am["ccnt"]))

        if tw["beg"]:
            g=_g("time_windows")
            self._ds(g,"begin",_f64(tw["beg"])); self._ds(g,"end",_f64(tw["end"]))
            self._ds(g,"reference",_f64(tw["ref"]))

        if fm["pid"]:
            g=_g("focal_mechanisms")
            self._ds(g,"public_id",self._sa(fm["pid"])); self._ds(g,"event_idx",_u32(fm["eidx"]))
            self._ds(g,"triggering_origin_id",self._sa(fm["toid"]))
            for k,n in [("np1sv","np1_strike_value"),("np1su","np1_strike_unc"),
                        ("np1dv","np1_dip_value"),("np1du","np1_dip_unc"),
                        ("np1rv","np1_rake_value"),("np1ru","np1_rake_unc"),
                        ("np2sv","np2_strike_value"),("np2su","np2_strike_unc"),
                        ("np2dv","np2_dip_value"),("np2du","np2_dip_unc"),
                        ("np2rv","np2_rake_value"),("np2ru","np2_rake_unc")]:
                self._ds(g,n,_f64(fm[k]))
            self._ds(g,"preferred_plane",_u8(fm["pp"]))
            for k,n in [("tazv","t_azimuth_value"),("tazu","t_azimuth_unc"),
                        ("tplv","t_plunge_value"),("tplu","t_plunge_unc"),
                        ("tlnv","t_length_value"),("tlnu","t_length_unc"),
                        ("pazv","p_azimuth_value"),("pazu","p_azimuth_unc"),
                        ("pplv","p_plunge_value"),("pplu","p_plunge_unc"),
                        ("plnv","p_length_value"),("plnu","p_length_unc"),
                        ("nazv","n_azimuth_value"),("nazu","n_azimuth_unc"),
                        ("nplv","n_plunge_value"),("nplu","n_plunge_unc"),
                        ("nlnv","n_length_value"),("nlnu","n_length_unc")]:
                self._ds(g,n,_f64(fm[k]))
            self._ds(g,"azimuthal_gap",_f64(fm["ag"]))
            self._ds(g,"station_polarity_count",_i32(fm["spc"]))
            self._ds(g,"misfit",_f64(fm["mft"])); self._ds(g,"station_dist_ratio",_f64(fm["sdr"]))
            self._ds(g,"method_id",self._sa(fm["mid"]))
            self._ds(g,"eval_mode",_u8(fm["emode"]),enum_map=EVALUATION_MODE)
            self._ds(g,"eval_status",_u8(fm["estat"]),enum_map=EVALUATION_STATUS)
            self._ds(g,"ci_idx",_i32(fm["ci"])); self._ds(g,"mt_idx",_i32(fm["mtidx"]))
            self._ds(g,"comment_offset",_u32(fm["coff"])); self._ds(g,"comment_count",_u32(fm["ccnt"]))
            self._ds(g,"waveform_pool_offset",_u32(fm["wpoff"]))
            self._ds(g,"waveform_pool_count",_u32(fm["wpcnt"]))
            self._ds(g,"waveform_pool",np.array(fmwp,dtype=np.uint32) if fmwp else np.array([],dtype=np.uint32))

        if mt["pid"]:
            g=_g("moment_tensors")
            self._ds(g,"public_id",self._sa(mt["pid"]))
            self._ds(g,"derived_origin_id",self._sa(mt["doid"]))
            self._ds(g,"moment_mag_id",self._sa(mt["mmid"]))
            self._ds(g,"scalar_moment_value",_f64(mt["scv"])); self._ds(g,"scalar_moment_unc",_f64(mt["scu"]))
            for short in ("rr","tt","pp","rt","rp","tp"):
                self._ds(g,f"{short}_value",_f64(mt[f"{short}_v"]))
                self._ds(g,f"{short}_unc",  _f64(mt[f"{short}_u"]))
            self._ds(g,"variance",_f64(mt["var"])); self._ds(g,"variance_reduction",_f64(mt["vr"]))
            self._ds(g,"double_couple",_f64(mt["dc"])); self._ds(g,"clvd",_f64(mt["clvd"]))
            self._ds(g,"iso",_f64(mt["iso"]))
            self._ds(g,"greens_function_id",self._sa(mt["gfid"]))
            self._ds(g,"filter_id",self._sa(mt["fid"]))
            self._ds(g,"stf_type",_u8(mt["stft"]),enum_map=SOURCE_TIME_FUNC_TYPE)
            self._ds(g,"stf_duration",_f64(mt["stfd"])); self._ds(g,"stf_rise_time",_f64(mt["stfr"]))
            self._ds(g,"stf_decay_time",_f64(mt["stfdc"]))
            self._ds(g,"method_id",self._sa(mt["mid"]))
            self._ds(g,"category",_u8(mt["cat"]),enum_map=MT_CATEGORY)
            self._ds(g,"inversion_type",_u8(mt["inv"]),enum_map=MT_INVERSION_TYPE)
            self._ds(g,"ci_idx",_i32(mt["ci"]))
            self._ds(g,"data_used_offset",_u32(mt["duoff"])); self._ds(g,"data_used_count",_u32(mt["ducnt"]))
            self._ds(g,"comment_offset",_u32(mt["coff"])); self._ds(g,"comment_count",_u32(mt["ccnt"]))

        if du["wt"]:
            g=_g("data_used")
            self._ds(g,"wave_type",_u8(du["wt"]),enum_map=DATA_USED_WAVE_TYPE)
            self._ds(g,"station_count",_i32(du["sc"])); self._ds(g,"component_count",_i32(du["cc"]))
            self._ds(g,"shortest_period",_f64(du["sp"])); self._ds(g,"longest_period",_f64(du["lp"]))

        wf.write(_g("waveform_ids"),self._CS)
        ci.write(_g("creation_info"),self._C,self._CS)
        cp.write(_g("comments"),self._C,self._CS)

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

    def read_catalog(self,event_indices=None):
        """Reconstruct an ObsPy Catalog. Pass event_indices to load a subset."""
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

            og=self._grp("origins"); mgg=self._grp("magnitudes")
            smg=self._grp("station_magnitudes"); pkg=self._grp("picks")
            amg=self._grp("amplitudes"); fmg=self._grp("focal_mechanisms")

            for ei in event_indices:
                e=Event()
                e.resource_id=_make_rid(pid[ei])
                e.preferred_origin_id=_make_rid(po[ei])
                e.preferred_magnitude_id=_make_rid(pm[ei])
                e.preferred_focal_mechanism_id=_make_rid(pf[ei])
                e.event_type=_dec(ety[ei],EVENT_TYPE)
                e.event_type_certainty=_dec(ect[ei],EVENT_TYPE_CERTAINTY)
                e.creation_info=self._mk_ci(ci_rows,eci[ei])
                e.event_descriptions=self._rd_descs(int(doff[ei]),int(dcnt[ei]))
                e.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[ei]),int(ccnt[ei]))
                e.origins=self._rd_origins(og,ei,ci_rows,txt,cid,cidx,wf_rows)
                e.magnitudes=self._rd_magnitudes(mgg,ei,ci_rows,txt,cid,cidx)
                e.station_magnitudes=self._rd_sta_mags(smg,ei,ci_rows,wf_rows,txt,cid,cidx)
                e.picks=self._rd_picks(pkg,ei,ci_rows,wf_rows,txt,cid,cidx)
                e.amplitudes=self._rd_amplitudes(amg,ei,ci_rows,wf_rows,txt,cid,cidx)
                e.focal_mechanisms=self._rd_focmecs(fmg,ei,ci_rows,wf_rows,txt,cid,cidx)
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

    def _rd_descs(self,off,cnt):
        g=self._grp("event_descriptions")
        if g is None or cnt==0: return []
        ta=self._rs(g,"text"); ty=self._ra(g,"type")
        return [EventDescription(text=ta[i],type=_dec(ty[i],EVENT_DESC_TYPE))
                for i in range(off,off+cnt)]

    def _rd_origins(self,g,ei,ci_rows,txt,cid,cidx,wf_rows):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        def f64(k): return g[k][()]
        def str_(k): return self._rs(g,k)
        tv=f64("time_value"); tu=f64("time_uncertainty"); tlo=f64("time_lower_unc")
        thi=f64("time_upper_unc"); tcf=f64("time_conf")
        lav=f64("lat_value"); lau=f64("lat_uncertainty")
        lalo=f64("lat_lower_unc"); lahi=f64("lat_upper_unc"); lacf=f64("lat_conf")
        lov=f64("lon_value"); lou=f64("lon_uncertainty")
        lolo=f64("lon_lower_unc"); lohi=f64("lon_upper_unc"); locf=f64("lon_conf")
        dpv=f64("depth_value"); dpu=f64("depth_uncertainty")
        dplo=f64("depth_lower_unc"); dphi=f64("depth_upper_unc"); dpcf=f64("depth_conf")
        dtype=g["depth_type"][()]; tfx=g["time_fixed"][()]; epfx=g["epicenter_fixed"][()]
        rsid=str_("ref_system_id"); mid=str_("method_id"); emid=str_("earth_model_id")
        otype=g["type"][()]; reg=str_("region")
        emode=g["eval_mode"][()]; estat=g["eval_status"][()]; ci_i=g["ci_idx"][()]
        qidx=g["quality_idx"][()]; uidx=g["uncertainty_idx"][()]
        aoff=g["arrival_offset"][()]; acnt=g["arrival_count"][()]
        coff=g["comment_offset"][()]; ccnt=g["comment_count"][()]
        ctoff=g["comptime_offset"][()]; ctcnt=g["comptime_count"][()]
        oqg=self._grp("origin_quality"); oug=self._grp("origin_uncertainty")
        ceg=self._grp("confidence_ellipsoids"); arg=self._grp("arrivals")
        ctg=self._grp("composite_times")
        out=[]
        for i in idxs:
            o=Origin()
            o.resource_id=_make_rid(str_("public_id")[i])
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
            o.quality=self._rd_oq(oqg,int(qidx[i]))
            o.origin_uncertainty=self._rd_ou(oug,ceg,int(uidx[i]))
            o.arrivals=self._rd_arrivals(arg,int(aoff[i]),int(acnt[i]),ci_rows,txt,cid,cidx)
            o.composite_times=self._rd_ct(ctg,int(ctoff[i]),int(ctcnt[i]))
            o.comments=self._mk_comments(txt,cid,cidx,ci_rows,int(coff[i]),int(ccnt[i]))
            out.append(o)
        return out

    def _rd_oq(self,g,idx):
        if g is None or idx<0: return None
        def gi(k): v=int(g[k][idx]); return None if v==-1 else v
        def gf(k): return _nn(float(g[k][idx]))
        return OriginQuality(associated_phase_count=gi("assoc_phase_count"),
            used_phase_count=gi("used_phase_count"),
            associated_station_count=gi("assoc_sta_count"),
            used_station_count=gi("used_sta_count"),
            depth_phase_count=gi("depth_phase_count"),
            standard_error=gf("standard_error"),azimuthal_gap=gf("azimuthal_gap"),
            secondary_azimuthal_gap=gf("sec_azimuthal_gap"),
            ground_truth_level=_sv(g["ground_truth_level"][idx]) or None,
            minimum_distance=gf("minimum_distance"),maximum_distance=gf("maximum_distance"),
            median_distance=gf("median_distance"))

    def _rd_ou(self,oug,ceg,idx):
        if oug is None or idx<0: return None
        def gf(k): return _nn(float(oug[k][idx]))
        eidx=int(oug["ellipsoid_idx"][idx]); el=None
        if ceg is not None and eidx>=0:
            def cf(k): return _nn(float(ceg[k][eidx]))
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
            preferred_description=_dec(int(oug["preferred_description"][idx]),ORIGIN_UNCERTAINTY_DESC),
            confidence_level=gf("confidence_level"),confidence_ellipsoid=el)

    def _rd_arrivals(self,g,off,cnt,ci_rows,txt,cid,cidx):
        if g is None or cnt==0: return []
        out=[]
        for i in range(off,off+cnt):
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
            ta=gf("takeoff_value")
            ar=Arrival(resource_id=_make_rid(gs("public_id")),
                pick_id=_make_rid(gs("pick_id")),phase=gs("phase"),
                time_correction=gf("time_correction"),azimuth=gf("azimuth"),
                distance=gf("distance"),takeoff_angle=ta,
                time_residual=gf("time_residual"),
                horizontal_slowness_residual=gf("hslow_residual"),
                backazimuth_residual=gf("baz_residual"),
                time_weight=gf("time_weight"),horizontal_slowness_weight=gf("hslow_weight"),
                backazimuth_weight=gf("baz_weight"),earth_model_id=_make_rid(gs("earth_model_id")),
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            if ta: ar.takeoff_angle_errors=QuantityError(uncertainty=gf("takeoff_uncertainty"))
            ar.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            out.append(ar)
        return out

    def _rd_ct(self,g,off,cnt):
        if g is None or cnt==0: return []
        out=[]
        for i in range(off,off+cnt):
            def gi(k):
                v=int(g[k][i]); return None if v==-1 else v
            def gf(k):
                return _nn(float(g[k][i]))
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

    def _rd_magnitudes(self,g,ei,ci_rows,txt,cid,cidx):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        scg=self._grp("station_mag_contributions"); out=[]
        for i in idxs:
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
            m=Magnitude(resource_id=_make_rid(gs("public_id")),mag=gf("mag_value"),
                magnitude_type=gs("type"),origin_id=_make_rid(gs("origin_id")),
                method_id=_make_rid(gs("method_id")),
                station_count=_ni(int(g["station_count"][i])),
                azimuthal_gap=gf("azimuthal_gap"),
                evaluation_mode=_dec(int(g["eval_mode"][i]),EVALUATION_MODE),
                evaluation_status=_dec(int(g["eval_status"][i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            m.mag_errors=QuantityError(uncertainty=gf("mag_uncertainty"),
                lower_uncertainty=gf("mag_lower_unc"),upper_uncertainty=gf("mag_upper_unc"),
                confidence_level=gf("mag_conf"))
            m.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            m.station_magnitude_contributions=self._rd_sc(scg,
                int(g["contrib_offset"][i]),int(g["contrib_count"][i]))
            out.append(m)
        return out

    def _rd_sc(self,g,off,cnt):
        if g is None or cnt==0: return []
        out=[]
        for i in range(off,off+cnt):
            smid=_sv(g["station_magnitude_id"][i])
            out.append(StationMagnitudeContribution(station_magnitude_id=_make_rid(smid),
                residual=_nn(float(g["residual"][i])),weight=_nn(float(g["weight"][i]))))
        return out

    def _rd_sta_mags(self,g,ei,ci_rows,wf_rows,txt,cid,cidx):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        out=[]
        for i in idxs:
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
            wfidx=int(g["waveform_idx"][i])
            sm=StationMagnitude(resource_id=_make_rid(gs("public_id")),
                origin_id=_make_rid(gs("origin_id")),mag=gf("mag_value"),
                station_magnitude_type=gs("type"),amplitude_id=_make_rid(gs("amplitude_id")),
                method_id=_make_rid(gs("method_id")),
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            sm.mag_errors=QuantityError(uncertainty=gf("mag_uncertainty"),
                lower_uncertainty=gf("mag_lower_unc"),upper_uncertainty=gf("mag_upper_unc"))
            sm.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            out.append(sm)
        return out

    def _rd_picks(self,g,ei,ci_rows,wf_rows,txt,cid,cidx):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        out=[]
        for i in idxs:
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
            tv=gf("time_value"); wfidx=int(g["waveform_idx"][i])
            hsv=gf("hslow_value"); bvz=gf("baz_value")
            p=Pick(resource_id=_make_rid(gs("public_id")),
                time=_from_ts(tv) if tv else None,
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                filter_id=_make_rid(gs("filter_id")),method_id=_make_rid(gs("method_id")),
                horizontal_slowness=hsv,
                backazimuth=bvz,
                slowness_method_id=_make_rid(gs("slowness_method_id")),
                onset=_dec(int(g["onset"][i]),PICK_ONSET),
                phase_hint=gs("phase_hint"),
                polarity=_dec(int(g["polarity"][i]),PICK_POLARITY),
                evaluation_mode=_dec(int(g["eval_mode"][i]),EVALUATION_MODE),
                evaluation_status=_dec(int(g["eval_status"][i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            p.time_errors=QuantityError(uncertainty=gf("time_uncertainty"),
                lower_uncertainty=gf("time_lower_unc"),upper_uncertainty=gf("time_upper_unc"),
                confidence_level=gf("time_conf"))
            if hsv: p.horizontal_slowness_errors=QuantityError(uncertainty=gf("hslow_uncertainty"))
            if bvz: p.backazimuth_errors=QuantityError(uncertainty=gf("baz_uncertainty"))
            p.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            out.append(p)
        return out

    def _rd_amplitudes(self,g,ei,ci_rows,wf_rows,txt,cid,cidx):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        twg=self._grp("time_windows"); out=[]
        for i in idxs:
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
            wfidx=int(g["waveform_idx"][i]); twidx=int(g["time_window_idx"][i])
            tw=None
            if twg is not None and twidx>=0:
                tw=TimeWindow(begin=float(twg["begin"][twidx]),end=float(twg["end"][twidx]),
                              reference=_from_ts(float(twg["reference"][twidx])))
            perv=gf("period_value"); stv=gf("scaling_time_value")
            a=Amplitude(resource_id=_make_rid(gs("public_id")),
                generic_amplitude=gf("amp_value"),
                type=gs("type"),category=_dec(int(g["category"][i]),AMPLITUDE_CATEGORY),
                unit=_dec(int(g["unit"][i]),AMPLITUDE_UNIT),method_id=_make_rid(gs("method_id")),
                period=perv,
                snr=gf("snr"),time_window=tw,pick_id=_make_rid(gs("pick_id")),
                waveform_id=wf_rows[wfidx] if wfidx<len(wf_rows) else None,
                filter_id=_make_rid(gs("filter_id")),
                scaling_time=_from_ts(stv) if stv else None,
                magnitude_hint=gs("magnitude_hint"),
                evaluation_mode=_dec(int(g["eval_mode"][i]),EVALUATION_MODE),
                evaluation_status=_dec(int(g["eval_status"][i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            a.generic_amplitude_errors=QuantityError(uncertainty=gf("amp_uncertainty"),
                lower_uncertainty=gf("amp_lower_unc"),upper_uncertainty=gf("amp_upper_unc"),
                confidence_level=gf("amp_conf"))
            if perv: a.period_errors=QuantityError(uncertainty=gf("period_uncertainty"))
            if stv:  a.scaling_time_errors=QuantityError(uncertainty=gf("scaling_time_unc"))
            a.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            out.append(a)
        return out

    def _rd_focmecs(self,g,ei,ci_rows,wf_rows,txt,cid,cidx):
        idxs=self._eidx(g,ei)
        if not len(idxs): return []
        mtg=self._grp("moment_tensors"); dug=self._grp("data_used")
        wfpool=self._ra(g,"waveform_pool"); out=[]
        for i in idxs:
            def gf(k): return _nn(float(g[k][i]))
            def gs(k): return _sv(g[k][i]) or None
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
                pp=int(g["preferred_plane"][i])
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
            wpoff=int(g["waveform_pool_offset"][i]); wpcnt=int(g["waveform_pool_count"][i])
            fm_wf=[wf_rows[int(wfpool[j])] for j in range(wpoff,wpoff+wpcnt)
                   if wfpool is not None and int(wfpool[j])<len(wf_rows)]
            # moment tensor
            mtidx=int(g["mt_idx"][i])
            mt=self._rd_mt(mtg,dug,mtidx,ci_rows,txt,cid,cidx) if mtidx>=0 else None
            fm=FocalMechanism(resource_id=_make_rid(gs("public_id")),
                triggering_origin_id=_make_rid(gs("triggering_origin_id")),
                nodal_planes=nps,principal_axes=pa,
                azimuthal_gap=gf("azimuthal_gap"),
                station_polarity_count=_ni(int(g["station_polarity_count"][i])),
                misfit=gf("misfit"),station_distribution_ratio=gf("station_dist_ratio"),
                method_id=_make_rid(gs("method_id")),
                evaluation_mode=_dec(int(g["eval_mode"][i]),EVALUATION_MODE),
                evaluation_status=_dec(int(g["eval_status"][i]),EVALUATION_STATUS),
                creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][i])))
            fm.waveform_id=fm_wf
            if mt: fm.moment_tensor=mt
            fm.comments=self._mk_comments(txt,cid,cidx,ci_rows,
                int(g["comment_offset"][i]),int(g["comment_count"][i]))
            out.append(fm)
        return out

    def _rd_mt(self,g,dug,idx,ci_rows,txt,cid,cidx):
        if g is None or idx<0: return None
        def gf(k): return _nn(float(g[k][idx]))
        def gs(k): return _sv(g[k][idx]) or None
        def rq2(vk,uk):
            v=gf(vk); return v  # flat scalar; caller sets *_errors separately
        # tensor
        comps={s:rq2(f"{s}_value",f"{s}_unc") for s in ("rr","tt","pp","rt","rp","tp")}
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
        stft=_dec(int(g["stf_type"][idx]),SOURCE_TIME_FUNC_TYPE)
        stfd=gf("stf_duration")
        stf=SourceTimeFunction(type=stft,duration=stfd,rise_time=gf("stf_rise_time"),
            decay_time=gf("stf_decay_time")) if (stft or stfd) else None
        # data used
        duoff=int(g["data_used_offset"][idx]); ducnt=int(g["data_used_count"][idx])
        data_used=[]
        if dug is not None:
            for j in range(duoff,duoff+ducnt):
                data_used.append(DataUsed(
                    wave_type=_dec(int(dug["wave_type"][j]),DATA_USED_WAVE_TYPE),
                    station_count=_ni(int(dug["station_count"][j])),
                    component_count=_ni(int(dug["component_count"][j])),
                    shortest_period=_nn(float(dug["shortest_period"][j])),
                    longest_period=_nn(float(dug["longest_period"][j]))))
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
            category=_dec(int(g["category"][idx]),MT_CATEGORY),
            inversion_type=_dec(int(g["inversion_type"][idx]),MT_INVERSION_TYPE),
            creation_info=self._mk_ci(ci_rows,int(g["ci_idx"][idx])))
        if sm_v is not None: mt.scalar_moment_errors=QuantityError(uncertainty=sm_u)
        mt.data_used=data_used
        mt.comments=self._mk_comments(txt,cid,cidx,ci_rows,
            int(g["comment_offset"][idx]),int(g["comment_count"][idx]))
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
        inverse : bool
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

        n_orig=len(qidx)
        mask=np.zeros(n_orig,dtype=bool)
        for oi in range(n_orig):
            qi=int(qidx[oi])
            if qi<0: continue
            v=int(upc[qi])
            if v<0: continue
            if v<min_count: continue
            if max_count is not None and v>max_count: continue
            mask[oi]=True
        return np.where(mask)[0]

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


# ---------------------------------------------------------------------------
# Module-level convenience API — mirrors ObsPy's ergonomic style.
#
#   cat = qmlh5.read_catalog("incat.h5")
#   qmlh5.write_catalog(cat, "outcat.h5")
#   cat.write_catalog("outcat.h5")        # method patched onto Catalog below
#
# ---------------------------------------------------------------------------
def read_catalog(path, event_indices=None):
    """Read a QuakeML/HDF5 file and return an ObsPy :class:`Catalog`.

    Parameters
    ----------
    path : str
        Path to a qmlh5 file written by :func:`write_catalog` or
        :class:`QMLH5`.
    event_indices : iterable of int, optional
        Subset of event row indices to load. If ``None`` (default), the
        entire catalog is loaded.

    Returns
    -------
    obspy.core.event.Catalog
    """
    with QMLH5(path, "r") as q:
        return q.read_catalog(event_indices=event_indices)


def write_catalog(catalog, path):
    """Write an ObsPy :class:`Catalog` to a qmlh5 (HDF5) file.

    Parameters
    ----------
    catalog : obspy.core.event.Catalog
        The catalog to serialize.
    path : str
        Destination file path. Will be created or overwritten.
    """
    with QMLH5(path, "w") as q:
        q.write_catalog(catalog)


# Attach `write_catalog` as a method on ObsPy's Catalog so the user can write
#     cat.write_catalog("out.h5")
# in addition to the module-level form. We deliberately don't override the
# built-in `Catalog.write` (which dispatches by `format=...` to ObsPy's I/O
# plugins); this is a sibling, not a replacement.
if OBSPY_AVAILABLE:
    def _catalog_write_catalog(self, path):
        """Write this catalog to a qmlh5 (HDF5) file. See :func:`qmlh5.write_catalog`."""
        write_catalog(self, path)
    Catalog.write_catalog = _catalog_write_catalog

