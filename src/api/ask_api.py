from __future__ import annotations

from datetime import datetime
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..retrieval.graphrag import retrieve_clusters
from ..common.utils import data_dir


class AskRequest(BaseModel):
    query: str
    top_n: int | None = 5


app = FastAPI(title="Customer Journey GraphRAG API")


@app.post("/ask")
def ask(req: AskRequest):
    results = retrieve_clusters(req.query, top_n=req.top_n or 5)
    return {"query": req.query, "results": results}


# ----- Raw data schemas -----
class UserIn(BaseModel):
    id: int
    email: str
    phone: str
    name: str
    location: str
    last_active_at: datetime


class EventIn(BaseModel):
    event_id: str
    muid: int
    session_id: str | None = None
    event_name: str
    event_time: datetime
    device_os: str
    channel: str
    traffic_source: str
    category: str
    product_id: str
    page_url: str


class UsersBatch(BaseModel):
    users: List[UserIn]


class EventsBatch(BaseModel):
    events: List[EventIn]


# ----- Raw data endpoints -----
@app.get("/raw/users", tags=["Raw Data"])
def list_raw_users(limit: int = 20):
    users_path = data_dir("raw", "users.parquet")
    if not users_path.exists():
        raise HTTPException(status_code=404, detail="Raw users parquet not found. Run prep or POST /raw/users.")
    df = pd.read_parquet(users_path).head(max(1, int(limit)))
    return {"users": df.to_dict(orient="records")}


@app.get("/raw/events", tags=["Raw Data"])
def list_raw_events(limit: int = 20):
    events_path = data_dir("raw", "events.parquet")
    if not events_path.exists():
        raise HTTPException(status_code=404, detail="Raw events parquet not found. Run prep or POST /raw/events.")
    df = pd.read_parquet(events_path).head(max(1, int(limit)))
    return {"events": df.to_dict(orient="records")}


@app.post("/raw/users", tags=["Raw Data"])  # add or create raw users parquet
def add_raw_users(batch: UsersBatch):
    users_path = data_dir("raw", "users.parquet")
    new_df = pd.DataFrame([u.model_dump() for u in batch.users])
    if "last_active_at" in new_df.columns:
        new_df["last_active_at"] = pd.to_datetime(new_df["last_active_at"], utc=False)
    if users_path.exists():
        existing = pd.read_parquet(users_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # de-duplicate by user id, keep last occurrence
        combined = combined.drop_duplicates(subset=["id"], keep="last")
        combined.to_parquet(users_path, index=False)
        return {"inserted": len(new_df), "total": len(combined)}
    new_df.to_parquet(users_path, index=False)
    return {"inserted": len(new_df), "total": len(new_df)}


@app.post("/raw/events", tags=["Raw Data"])  # add or create raw events parquet
def add_raw_events(batch: EventsBatch):
    events_path = data_dir("raw", "events.parquet")
    new_df = pd.DataFrame([e.model_dump() for e in batch.events])
    if "event_time" in new_df.columns:
        new_df["event_time"] = pd.to_datetime(new_df["event_time"], utc=False)
    if events_path.exists():
        existing = pd.read_parquet(events_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # de-duplicate by event_id, keep last occurrence
        combined = combined.drop_duplicates(subset=["event_id"], keep="last")
        combined.to_parquet(events_path, index=False)
        return {"inserted": len(new_df), "total": len(combined)}
    new_df.to_parquet(events_path, index=False)
    return {"inserted": len(new_df), "total": len(new_df)}


