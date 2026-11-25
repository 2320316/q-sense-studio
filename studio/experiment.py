import itertools
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .forecast_data import rainfall, stage, windows
from .library import data

ARMS = ("qt", "classical", "compact", "softmax")
class Event(BaseModel):
    name: str
    start: datetime
    end: datetime


class Experiment(BaseModel):
    name: str = Field(default="Batangas rainfall pilot", min_length=1, max_length=120)
    dataset: str
    mode: Literal["rainfall","hourly","local"] = "rainfall"
    site: str = "PH04010"
    features: list[str] = []
    arms: list[Literal["qt","classical","compact","softmax"]] = ["qt","classical"]
    lookback: int = Field(default=12, ge=2, le=144)
    horizon: Literal[1,3,6] = 1
    hidden: int = Field(default=16, ge=2, le=128)
    depth: int = Field(default=2, ge=1, le=8)
    width: int = Field(default=24, ge=2, le=128)
    epochs: int = Field(default=40, ge=1, le=2000)
    batch: int = Field(default=64, ge=1, le=2048)
    lr: float = Field(default=.003, gt=0, le=1)
    optimizer: Literal["Adam","AdamW","SGD"] = "Adam"
    patience: int = Field(default=0, ge=0)
    seeds: list[int] = [42]
    device: Literal["cpu","mps","cuda"] = "cpu"
    val_start: str = ""
    test_start: str = ""
    folds: int = Field(default=1,ge=1,le=10)
    events: list[Event] = []
    threshold: float | None = None
    warm_dataset: str = ""
    warm_site: str = ""
    @model_validator(mode="after")
    def check(self):
        if not self.arms or not self.seeds or len(self.seeds)>20 or len(set(self.seeds))!=len(self.seeds) or len(set(self.arms))!=len(self.arms):
            raise ValueError("Choose distinct arms and one to twenty distinct seeds")
        if bool(self.val_start)!=bool(self.test_start):
            raise ValueError("Set both validation and test dates, or leave both blank")
        if self.mode=="local" and self.site not in ("C1","C2","C3","C4"):
            raise ValueError("Local telemetry sites are C1, C2, C3 or C4")
        if self.warm_dataset and (self.mode!="local" or not self.warm_site):
            raise ValueError("Transfer requires local mode and an hourly gauge ID")
        ordered=sorted(self.events,key=lambda e:e.start)
        if any(e.end<e.start for e in ordered) or any(b.start<=a.end for a,b in zip(ordered,ordered[1:])) or len({e.name for e in ordered})!=len(ordered):
            raise ValueError("Verified events need unique names and non-overlapping intervals")
        return self


def source(c, warm=False):
    key=c.warm_dataset if warm else c.dataset
    path,info=data(key)
    if c.mode=="rainfall":
        if info["kind"]!="rainfall":
            raise ValueError("Rainfall pilot needs Batangas dekadal rainfall")
        dates,ticks,x,names=rainfall(path)
        y=x[:,0]
    else:
        if info["kind"]!="stage":
            raise ValueError("Import or map measured stage, interval rainfall, site, timestamp and QA columns first")
        if info.get("provenance")=="synthetic" and not warm:
            raise ValueError("Synthetic stage is for historical warm starts only")
        dates,ticks,x,names,y=stage(path,"hourly" if warm else c.mode,c.warm_site if warm else c.site)
    if not len(ticks) or (np.diff(ticks)<=0).any():
        raise ValueError("No distinct ordered observations for this site")
    return dates,ticks,x,names,y


def partition(c, dates):
    n=len(dates)
    idx=lambda s:int(np.searchsorted(dates,datetime.fromisoformat(s)))
    val=idx(c.val_start) if c.val_start else int(.55*n if c.folds>1 else .7*n)
    test=idx(c.test_start) if c.test_start else int(.7*n if c.folds>1 else .85*n)
    def boundary(i):
        for e in c.events:
            if i<n and e.start<dates[i]<=e.end:
                return int(np.searchsorted(dates,e.start))
        return i
    edges=[boundary(i) for i in np.linspace(test,n,c.folds+1,dtype=int)]
    folds=[]
    for i in range(c.folds):
        a=boundary(val) if i==0 else edges[i-1]
        bounds=(0,a,edges[i],edges[i+1])
        if min(np.diff(bounds))<=0:
            raise ValueError("Chosen split dates or storm blocks produce an empty fold")
        folds.append(bounds)
    return folds


def shape_info(c, inputs):
    def count(h):
        return 4*h*(inputs+h+2)+h+1
    weights=count(c.hidden)
    qubits=(weights-1).bit_length()
    budget=c.depth*qubits+c.width*(qubits+3)+1
    compact=min(range(1,c.hidden+1),key=lambda h:abs(count(h)-budget))
    return dict(weights=weights,qubits=qubits,qt_trainable=budget,compact_hidden=compact,
                compact_trainable=count(compact),compact_difference=count(compact)-budget)


def prepare(c, fold=0, warm=False):
    dates,ticks,x,available,y=source(c,warm)
    names=c.features or (source(c)[3] if warm else available)
    if len(set(names))!=len(names):
        raise ValueError("Duplicate features")
    if not warm and not set(names)<=set(available):
        raise ValueError("Unknown feature. Available: "+", ".join(available))
    if warm and not set(names).intersection(available):
        raise ValueError("Transfer requires at least one shared hourly feature")
    xx=np.column_stack([x[:,available.index(k)] if k in available else np.zeros(len(x)) for k in names])
    if warm:
        stop=np.searchsorted(dates,source(c)[0][0])
        dates,ticks,xx,y=dates[:stop],ticks[:stop],xx[:stop],y[:stop]
        bounds=(0,int(stop*.7),int(stop*.85),stop)
    else:
        bounds=partition(c,dates)[fold]
    warmup=143 if c.mode=="local" and not warm else 0
    horizon=(c.horizon*(6 if c.mode=="local" and not warm else 1)) if c.mode!="rainfall" else 1
    raw=[windows(ticks,xx,y,c.lookback,horizon,a+warmup,b) for a,b in zip(bounds,bounds[1:])]
    # Baselines use measured history even if the model omits stage from its predictors.
    origin=[x[ids-horizon,0] for _,_,ids in raw]
    return dict(raw=raw,dates=dates,names=names,ticks=ticks,y=y,horizon=horizon,origin=origin,available=available,
                bounds=[dates[i].isoformat() if i<len(dates) else dates[-1].isoformat() for i in bounds])


def preview(c):
    p=prepare(c)
    folds=[]
    for i in range(c.folds):
        seq=prepare(c,i) if i else p
        folds.append(dict(boundaries=seq["bounds"],windows=[len(y) for _,y,_ in seq["raw"]]))
    init=["scratch","transfer"] if c.warm_dataset else ["scratch"]
    if c.warm_dataset:
        prepare(c,warm=True)
    specs=list(itertools.product(range(c.folds),c.seeds,c.arms,init))
    return dict(folds=folds,features=p["names"],available=p["available"],jobs=len(specs),
                specs=specs,**shape_info(c,len(p["names"])))
