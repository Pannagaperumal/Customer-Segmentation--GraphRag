from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

from ..common.utils import data_dir, ensure_directory


def _random_datetime_between(start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def generate_synthetic_data(
    num_users: int = 5000,
    min_events_per_user: int = 5,
    max_events_per_user: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    device_oses = ["Android", "iOS", "Windows", "MacOS", "Linux"]
    channels = ["email", "push", "organic", "paid", "referral"]
    traffic_sources = ["google", "facebook", "twitter", "newsletter", "direct"]
    categories = ["electronics", "fashion", "home", "sports", "beauty", "toys"]
    event_names = [
        "page_view",
        "product_view",
        "add_to_cart",
        "checkout_start",
        "purchase",
    ]

    # Users
    users = []
    start_date = datetime.utcnow() - timedelta(days=120)
    end_date = datetime.utcnow()

    locations = ["US", "CA", "UK", "DE", "IN", "BR", "AU", "SG"]

    for uid in range(1, num_users + 1):
        name = f"User {uid:04d}"
        email = f"user{uid}@example.com"
        phone = f"+1-555-{uid:07d}"[-12:]
        location = random.choice(locations)
        last_active_at = _random_datetime_between(start_date, end_date)
        users.append(
            {
                "id": uid,
                "email": email,
                "phone": phone,
                "name": name,
                "location": location,
                "last_active_at": last_active_at,
            }
        )

    users_df = pd.DataFrame(users)

    # Events
    events = []
    for uid in users_df["id"].tolist():
        num_events = random.randint(min_events_per_user, max_events_per_user)
        session_id = str(uuid.uuid4())
        last_time = _random_datetime_between(start_date, end_date)
        for _ in range(num_events):
            last_time += timedelta(minutes=random.randint(1, 180))
            event_name = random.choices(
                event_names,
                weights=[40, 30, 15, 10, 5],
                k=1,
            )[0]
            category = random.choice(categories)
            product_id = f"{category[:3]}-{random.randint(1, 5000):05d}"
            page_url = f"/shop/{category}/{product_id}"
            events.append(
                {
                    "event_id": str(uuid.uuid4()),
                    "muid": uid,
                    "session_id": session_id,
                    "event_name": event_name,
                    "event_time": last_time,
                    "device_os": random.choice(device_oses),
                    "channel": random.choice(channels),
                    "traffic_source": random.choice(traffic_sources),
                    "category": category,
                    "product_id": product_id,
                    "page_url": page_url,
                }
            )

    events_df = pd.DataFrame(events)

    # Save raw
    raw_dir = data_dir("raw")
    ensure_directory(raw_dir)
    users_df.to_parquet(raw_dir / "users.parquet", index=False)
    events_df.to_parquet(raw_dir / "events.parquet", index=False)

    return users_df, events_df


def _compute_funnel_flags(events_df: pd.DataFrame) -> pd.DataFrame:
    has_view = (
        events_df[events_df["event_name"] == "product_view"].groupby("muid")["event_id"].count() > 0
    )
    has_add = (
        events_df[events_df["event_name"] == "add_to_cart"].groupby("muid")["event_id"].count() > 0
    )
    has_checkout = (
        events_df[events_df["event_name"] == "checkout_start"].groupby("muid")["event_id"].count() > 0
    )
    has_purchase = (
        events_df[events_df["event_name"] == "purchase"].groupby("muid")["event_id"].count() > 0
    )

    flags = pd.DataFrame(
        {
            "viewed_not_added": has_view & ~has_add,
            "added_not_purchased": has_add & ~has_purchase,
            "checkout_abandoned": has_checkout & ~has_purchase,
        }
    ).reset_index().rename(columns={"index": "muid"})
    for col in ["viewed_not_added", "added_not_purchased", "checkout_abandoned"]:
        flags[col] = flags[col].astype(bool)
    return flags


def build_user_features(users_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    # Base counts
    event_counts = events_df.groupby("muid").size().rename("event_count").reset_index()
    session_counts = events_df.groupby(["muid", "session_id"]).size().groupby("muid").size().rename("session_count").reset_index()
    unique_pages = events_df.groupby("muid")["page_url"].nunique().rename("unique_pages").reset_index()
    unique_categories = events_df.groupby("muid")["category"].nunique().rename("unique_categories").reset_index()
    unique_products = events_df.groupby("muid")["product_id"].nunique().rename("unique_products").reset_index()

    # Device OS distribution
    device_dist = (
        events_df.pivot_table(index="muid", columns="device_os", values="event_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    # Normalize per user to proportions
    device_cols = [c for c in device_dist.columns if c != "muid"]
    device_dist[device_cols] = device_dist[device_cols].div(device_dist[device_cols].sum(axis=1).replace(0, 1), axis=0)
    device_dist = device_dist.rename(columns={c: f"device_{c.lower()}_share" for c in device_cols})

    # Category distribution (per user proportion across event categories)
    category_dist = (
        events_df.pivot_table(index="muid", columns="category", values="event_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    category_cols = [c for c in category_dist.columns if c != "muid"]
    if category_cols:
        category_dist[category_cols] = category_dist[category_cols].div(
            category_dist[category_cols].sum(axis=1).replace(0, 1), axis=0
        )
    category_dist = category_dist.rename(columns={c: f"category_{c.lower()}_share" for c in category_cols})

    # Funnel flags
    flags = _compute_funnel_flags(events_df)

    # First/last activity
    activity = events_df.groupby("muid")["event_time"].agg(["min", "max"]).reset_index().rename(columns={"min": "first_activity", "max": "last_activity"})

    # Join all
    feats = (
        users_df.rename(columns={"id": "muid"})[["muid", "location", "last_active_at"]]
        .merge(event_counts, on="muid", how="left")
        .merge(session_counts, on="muid", how="left")
        .merge(unique_pages, on="muid", how="left")
        .merge(unique_categories, on="muid", how="left")
        .merge(unique_products, on="muid", how="left")
        .merge(device_dist, on="muid", how="left")
        .merge(flags, on="muid", how="left")
        .merge(activity, on="muid", how="left")
    )
    # Merge category distribution if present
    feats = feats.merge(category_dist, on="muid", how="left")
    feats = feats.fillna({
        "event_count": 0,
        "session_count": 0,
        "unique_pages": 0,
        "unique_categories": 0,
        "unique_products": 0,
        "viewed_not_added": False,
        "added_not_purchased": False,
        "checkout_abandoned": False,
    })

    # Derived time features
    # Use tz-naive 'now' to match tz-naive event timestamps
    now = pd.Timestamp.now(tz=None)
    feats["days_since_first"] = (now - pd.to_datetime(feats["first_activity"])) / pd.Timedelta(days=1)
    feats["days_since_last"] = (now - pd.to_datetime(feats["last_activity"])) / pd.Timedelta(days=1)
    feats["lifetime_days"] = (pd.to_datetime(feats["last_activity"]) - pd.to_datetime(feats["first_activity"])) / pd.Timedelta(days=1)

    # Save processed
    processed_path = data_dir("processed", "user_features.parquet")
    ensure_directory(processed_path.parent)
    feats.to_parquet(processed_path, index=False)

    return feats


def main() -> None:
    users_df, events_df = generate_synthetic_data()
    build_user_features(users_df, events_df)
    print("Saved processed user features to", data_dir("processed", "user_features.parquet"))


if __name__ == "__main__":
    main()


