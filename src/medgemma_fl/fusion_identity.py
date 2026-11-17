"""Canonical patient identity checks shared by export and evaluation tools."""


def canonical_uid(row):
    uid = str(row.get("canonical_uid", "")).strip()
    # Older native preparation already used UID.Subject as patient_id. Never
    # apply this fallback to fixed-panel sample IDs or Comp Bio ID prefixes.
    if not uid and row.get("patient_id_source") == "RMSs_meta UID.Subject":
        value = row.get("patient_id", "")
        if value.startswith("COMET:"):
            uid = value[len("COMET:"):].strip()
    if not uid:
        raise ValueError("Missing canonical_uid: relink the record to UID.Subject before export")
    return uid


def validate_patient_rows(rows):
    """Require unique record IDs and one split and fusion label per patient."""
    records, patients = {}, {}
    for row in rows:
        rid, uid = row["patient_id"], canonical_uid(row)
        if rid in records:
            raise ValueError("Duplicate record ID")
        state = (row["split"], row["diagnosis"])
        if state[0] not in ("train", "validation", "test") or state[1] not in ("NEG", "POS"):
            raise ValueError("Invalid split or fusion label")
        if uid in patients and patients[uid] != state:
            raise ValueError("Canonical patient has cross-split overlap or conflicting fusion labels")
        patients[uid] = state
        records[rid] = (uid, *state)
    if not records:
        raise ValueError("Empty cohort")
    return records
