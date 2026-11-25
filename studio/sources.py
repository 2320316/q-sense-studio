import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen

from .library import LAYERS,SIDECARS,inspect_files

SOURCES=[
 dict(id="hdx",name="Philippines rainfall",provider="WFP / HDX",area="Philippines · Batangas filter",kind="Dekadal rainfall",url="https://data.humdata.org/dataset/phl-rainfall-subnational",action="Collect"),
 dict(id="noah",name="Batangas flood hazards",provider="Project NOAH · BetterGovPH archive",area="Batangas",kind="100-year hazard polygons",url="https://huggingface.co/datasets/bettergovph/project-noah-hazard-maps/tree/main/Flood",action="Collect"),
 dict(id="pagasa",name="Ambulong station observations",provider="DOST-PAGASA CADS",area="Tanauan, Batangas",kind="Hourly rainfall · academic request",url="https://pagasa.dost.gov.ph/climate/climate-data",action="Open request"),
 dict(id="stage",name="Basin gauge observations",provider="DOST-PAGASA Hydrometeorology",area="Request by gauge and dates",kind="Hourly measured stage",url="https://ihdrms.app/",action="Open request"),
 dict(id="lipad",name="Elevation and terrain",provider="UP LiPAD",area="Request by area of interest",kind="DEM / DTM",url="https://lipad.dream.upd.edu.ph/",action="Open request"),
 dict(id="fmc",name="Flood map layers",provider="Phil-LiDAR / LiPAD FMC",area="Philippine river basins",kind="GIS layers · provider download",url="https://lipad-fmc.dream.upd.edu.ph/layers/",action="Open catalog")]


def fetch(url):
    return urlopen(Request(url,headers={"User-Agent":"Q-Sense-Studio/1.0 (research data collection)"}),timeout=45)


def collect(key):
    if key=="hdx":
        with fetch("https://data.humdata.org/api/3/action/package_show?id=phl-rainfall-subnational") as f:
            resources=json.load(f)["result"]["resources"]
        res=next((r for r in resources if "phl-rainfall-subnat-full.csv" in (r.get("url","")+r.get("name",""))),None)
        if not res:
            raise ValueError("HDX did not list the full historical CSV. Download it from the provider and import it.")
        url=res["url"]
        name="phl-rainfall-subnat-full.csv"
    elif key=="noah":
        repo="https://huggingface.co"
        api=repo+"/api/datasets/bettergovph/project-noah-hazard-maps/tree/main/"
        with fetch(api+"Flood") as f:
            groups=json.load(f)
        group=next((v["path"] for v in groups if v["type"]=="directory" and "100" in v["path"]),None)
        if not group:
            raise ValueError("The archive has changed. Open the provider catalog.")
        url=repo+"/datasets/bettergovph/project-noah-hazard-maps/resolve/main/"+quote(group)+"/Batangas.zip"
        name="Batangas.zip"
    else:
        raise ValueError("Use the provider request route for this dataset")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with fetch(url) as response,(root/name).open("wb") as out:
            shutil.copyfileobj(response,out)
        if key=="noah":
            with zipfile.ZipFile(root/name) as archive:
                for member in archive.infolist():
                    p=Path(member.filename)
                    if "__MACOSX" in p.parts or p.name.startswith(".") or p.suffix.lower() not in LAYERS|SIDECARS:
                        continue
                    with archive.open(member) as src,(root/p.name).open("wb") as out:
                        shutil.copyfileobj(src,out)
        return inspect_files(root,url)
