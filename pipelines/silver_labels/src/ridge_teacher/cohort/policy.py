"""Only confirmed ERMS / ARMS subtypes may enter any training dataset."""
import re

POLICY_VERSION = "20261117-erms-arms-only-training"


def normalized_subtype(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def excluded_subtype(value):
    value = normalized_subtype(value)
    # A whitelist also excludes missing / unknown and newly encountered subtypes.
    return value not in {"erms", "arms"}


def training_excluded(row):
    # Include every observed sample's subtype, not just the representative array.
    subtypes = str(row.get("observed_subtypes", row.get("subtype", ""))).split("|")
    blocked = any(excluded_subtype(s) for s in subtypes + [row.get("subtype", "")])
    return blocked or str(row.get("training_excluded", "False")).lower() in {"true", "1"}


def assert_training_policy(patients):
    if "subtype" not in patients.columns:
        raise ValueError("Subtype metadata is required to enforce training exclusions")
    blocked = patients.apply(training_excluded, axis=1)
    training = patients.teacher_train | patients.split.isin(["train_gold", "train_silver_candidate"])
    if (blocked & training).any():
        raise ValueError("Subtype exclusion: only ERMS / ARMS may enter any training dataset")
