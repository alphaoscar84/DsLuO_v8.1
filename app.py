import pandas as pd
import math
import traceback
import time
import csv
import random
import io
import streamlit as st
import pulp


# =========================================================
# DEFAULT SETTINGS
# =========================================================
DEFAULTS = {
    "SALARY_CAP": 100000,
    "LINEUP_COUNT": 150,
    "REQ_DEF": 2,
    "REQ_MID": 4,
    "REQ_RK": 1,
    "REQ_FWD": 2,
    "MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS": 4,
    "POSITIONAL_DIFFERENCE_PATTERN": [
        {"DEF": 1, "MID": 2, "FWD": 1, "RK": 0},
        {"DEF": 1, "MID": 1, "FWD": 1, "RK": 1},
    ],
    "POSITION_START_CYCLE": ["DEF", "MID", "FWD", "RK"],
    "RANDOM_SEED": 42,
    "TIEBREAK_EPSILON": 1e-12,
    "MIN_TOP3_BUCKET_SHARE": 0.25,
    "TOP3_BUCKET_RANDOM_SEED": 123,
    "MAX_PLAYER_LINEUP_SHARE": 0.7,
    "MAX_PLAYERS_PER_TEAM": 6,
    "SOLVER_TIME_LIMIT": None,
    "PLAYING_STATUS_REQUIRED_TEXT": "IN TEAM TO PLAY",
    "FALLBACK_PROJECTION": 47,
    "DRAFTSTARS_START_ROW": 2,
    "DRAFTSTARS_START_COL": 4,
}

DRAFTSTARS_POSITION_ORDER = [
    "FWD1", "FWD2",
    "MID1", "MID2", "MID3", "MID4",
    "DEF1", "DEF2",
    "RK1"
]

TEAM_ALIASES = {
    "adelaide": "Crows",
    "crows": "Crows",
    "brisbane": "Lions",
    "brisbane lions": "Lions",
    "lions": "Lions",
    "carlton": "Blues",
    "blues": "Blues",
    "collingwood": "Magpies",
    "magpies": "Magpies",
    "essendon": "Bombers",
    "bombers": "Bombers",
    "fremantle": "Dockers",
    "dockers": "Dockers",
    "geelong": "Cats",
    "cats": "Cats",
    "gold coast": "Suns",
    "gold coast suns": "Suns",
    "suns": "Suns",
    "greater western sydney": "Giants",
    "gws": "Giants",
    "giants": "Giants",
    "hawthorn": "Hawks",
    "hawks": "Hawks",
    "melbourne": "Demons",
    "demons": "Demons",
    "north melbourne": "Kangaroos",
    "kangaroos": "Kangaroos",
    "port adelaide": "Power",
    "power": "Power",
    "richmond": "Tigers",
    "tigers": "Tigers",
    "st kilda": "Saints",
    "saints": "Saints",
    "sydney": "Swans",
    "swans": "Swans",
    "west coast": "Eagles",
    "eagles": "Eagles",
    "western bulldogs": "Bulldogs",
    "bulldogs": "Bulldogs",
}

POS_PREFIX = {"DEF": "D", "MID": "M", "FWD": "F", "RK": "R"}


# =========================================================
# HELPERS
# =========================================================
def find_first_existing_column(df, candidates, required=True):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def normalise_team(team):
    if pd.isna(team):
        return ""
    key = str(team).strip().lower()
    return TEAM_ALIASES.get(key, str(team).strip())


def parse_positions(pos_value):
    if pd.isna(pos_value):
        return []
    s = str(pos_value).upper().replace(" ", "")
    s = s.replace("RUC", "RK")
    parts = [p for p in s.replace("/", ",").split(",") if p]
    out = []
    for p in parts:
        if p in {"DEF", "MID", "FWD", "RK"}:
            out.append(p)
    return list(dict.fromkeys(out))


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def format_score(x):
    if pd.isna(x):
        return ""
    return round(float(x), 2)


def rotated_position_order_for_lineup(lineup_idx, position_start_cycle):
    start_idx = (lineup_idx - 1) % len(position_start_cycle)
    return position_start_cycle[start_idx:] + position_start_cycle[:start_idx]


def build_tiebreak_bonus_map(df, lineup_idx, random_seed, position_start_cycle):
    rng = random.Random(random_seed + lineup_idx)
    pos_order = rotated_position_order_for_lineup(lineup_idx, position_start_cycle)
    pos_weight = {pos: len(pos_order) - i for i, pos in enumerate(pos_order)}

    bonuses = {}
    for pos in pos_order:
        pos_rows = df.loc[df["Position"] == pos, "RowID"].tolist()
        rng.shuffle(pos_rows)
        for j, rid in enumerate(pos_rows):
            bonuses[rid] = pos_weight[pos] * 1_000_000 + (len(pos_rows) - j)

    return bonuses, pos_order


def load_inputs(players_file_obj, projection_file_obj):
    players_name = players_file_obj.name.lower()
    projection_name = projection_file_obj.name.lower()

    if not players_name.endswith(".csv"):
        raise ValueError("Players file must be a CSV.")

    if not projection_name.endswith(".csv"):
        raise ValueError("Pre-game players file must be a CSV.")

    players = pd.read_csv(players_file_obj)
    projections = pd.read_csv(projection_file_obj)

    return players, projections


# =========================================================
# DRAFTSTARS HELPERS
# =========================================================
def load_draftstars_name_id_lookup_from_upload(draftstars_file_obj):
    raw_bytes = draftstars_file_obj.getvalue()

    for skip in range(0, 20):
        try:
            test_df = pd.read_csv(io.BytesIO(raw_bytes), skiprows=skip)

            name_col = None
            id_col = None

            for c in test_df.columns:
                cl = str(c).strip().lower()
                if cl == "name":
                    name_col = c
                elif cl in {"id", "player id", "playerid"}:
                    id_col = c

            if name_col is not None and id_col is not None:
                lookup_df = test_df[[name_col, id_col]].copy()
                lookup_df[name_col] = lookup_df[name_col].astype(str).str.strip()
                lookup_df[id_col] = lookup_df[id_col].astype(str).str.strip()

                lookup_df = lookup_df[
                    (lookup_df[name_col] != "") &
                    (lookup_df[id_col] != "") &
                    (lookup_df[name_col].str.lower() != "nan") &
                    (lookup_df[id_col].str.lower() != "nan")
                ].copy()

                return dict(zip(lookup_df[name_col], lookup_df[id_col])), skip, raw_bytes

        except Exception:
            continue

    return None, None, raw_bytes


def build_updated_draftstars_csv(draftstars_file_obj, lineups_export, start_row, start_col):
    name_to_id, detected_skiprows, raw_bytes = load_draftstars_name_id_lookup_from_upload(draftstars_file_obj)

    if not name_to_id:
        raise ValueError("Could not detect Draftstars Name/ID columns automatically.")

    all_rows = list(csv.reader(io.StringIO(raw_bytes.decode("utf-8"))))

    missing_lineup_cols = [c for c in DRAFTSTARS_POSITION_ORDER if c not in lineups_export.columns]
    if missing_lineup_cols:
        raise ValueError(f"Expected lineup columns not found: {missing_lineup_cols}")

    inserted_count = 0
    missing_id_names = []
    truncated = False

    for lineup_idx, (_, lineup_row) in enumerate(lineups_export.iterrows()):
        row_index = start_row + lineup_idx

        if row_index >= len(all_rows):
            truncated = True
            break

        player_cells = []

        for col in DRAFTSTARS_POSITION_ORDER:
            raw_name = lineup_row.get(col, "")
            if pd.isna(raw_name) or str(raw_name).strip() == "":
                player_cells.append("")
                continue

            name = str(raw_name).strip()
            player_id = name_to_id.get(name)

            if player_id:
                player_cells.append(f"{name} ({player_id})")
            else:
                player_cells.append(name)
                missing_id_names.append(name)

        needed_len = start_col + len(player_cells)
        if len(all_rows[row_index]) < needed_len:
            all_rows[row_index].extend([""] * (needed_len - len(all_rows[row_index])))

        for i, val in enumerate(player_cells):
            all_rows[row_index][start_col + i] = val

        inserted_count += 1

    out = io.StringIO()
    writer = csv.writer(out, quoting=csv.QUOTE_ALL)
    writer.writerows(all_rows)

    meta = {
        "detected_skiprows": detected_skiprows,
        "inserted_count": inserted_count,
        "missing_id_names": sorted(set(missing_id_names)),
        "truncated": truncated,
    }

    return out.getvalue().encode("utf-8"), meta


# =========================================================
# LOAD + PREP
# =========================================================
def prepare_players_df(players, required_status_text):
    name_col = find_first_existing_column(players, ["Name"])
    team_col = find_first_existing_column(players, ["Team"])
    salary_col = find_first_existing_column(players, ["Salary"])
    pos_col = find_first_existing_column(players, ["Position", "Pos"])

    player_id_col = find_first_existing_column(players, ["Player ID", "PlayerID", "ID"], required=False)
    opponent_col = find_first_existing_column(players, ["Opponent"], required=False)
    score_col = find_first_existing_column(players, ["Score", "Pts"], required=False)
    form_col = find_first_existing_column(players, ["Form"], required=False)
    playing_status_col = find_first_existing_column(players, ["Playing Status", "Status"], required=False)

    df = players.copy()
    df = df.rename(columns={
        name_col: "Name",
        team_col: "Team",
        salary_col: "Salary",
        pos_col: "PositionRaw",
    })

    df["Player ID"] = df[player_id_col] if player_id_col else ""
    df["Opponent"] = df[opponent_col] if opponent_col else ""
    df["Score"] = df[score_col] if score_col else pd.NA
    df["Form"] = df[form_col] if form_col else pd.NA
    df["Playing Status"] = df[playing_status_col] if playing_status_col else ""

    df["Name"] = (
        df["Name"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["Team"] = df["Team"].astype(str).str.strip()
    df["TeamNick"] = df["Team"].map(normalise_team)
    df["Salary"] = df["Salary"].apply(safe_int)
    df["Score"] = df["Score"].apply(lambda x: safe_float(x, default=math.nan))
    df["Form"] = df["Form"].apply(lambda x: safe_float(x, default=math.nan))
    df["Playing Status"] = df["Playing Status"].astype(str).str.strip()
    df["EligiblePositions"] = df["PositionRaw"].map(parse_positions)

    df = df[
        df["Playing Status"].str.contains(required_status_text, case=False, na=False)
    ].copy()

    df = df[df["EligiblePositions"].map(len) > 0].copy()
    return df


def prepare_projection_df(projections):
    player_col = find_first_existing_column(projections, ["Player", "Name"])
    team_col = find_first_existing_column(projections, ["Team"], required=False)
    fppg_col = find_first_existing_column(projections, ["FPPG"])

    df = projections.copy()
    rename_map = {
        player_col: "Player",
        fppg_col: "ProjectionAverage",
    }
    if team_col:
        rename_map[team_col] = "Team"

    df = df.rename(columns=rename_map)

    if "Team" not in df.columns:
        df["Team"] = ""

    df["Player"] = (
        df["Player"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["Team"] = df["Team"].astype(str).str.strip()
    df["TeamNick"] = df["Team"].map(normalise_team)
    df["ProjectionAverage"] = df["ProjectionAverage"].apply(safe_float)

    return df[["Player", "Team", "TeamNick", "ProjectionAverage"]].copy()


def match_players(players_df, projection_df, fallback_projection):
    out = players_df.merge(
        projection_df,
        left_on="Name",
        right_on="Player",
        how="left",
        suffixes=("_players", "_proj")
    )

    out["MatchMethod"] = out["ProjectionAverage"].notna().map(lambda x: "full-name" if x else "")
    unmatched_mask = out["ProjectionAverage"].isna()

    if unmatched_mask.any():
        fallback = players_df.merge(
            projection_df,
            left_on=["Name", "TeamNick"],
            right_on=["Player", "TeamNick"],
            how="left",
            suffixes=("_players", "_proj")
        )

        out.loc[unmatched_mask, "ProjectionAverage"] = fallback.loc[unmatched_mask, "ProjectionAverage"].values

        fallback_match_mask = pd.notna(fallback.loc[unmatched_mask, "ProjectionAverage"]).values
        out.loc[unmatched_mask, "MatchMethod"] = [
            "full-name+team" if matched else ""
            for matched in fallback_match_mask
        ]

    if "Team_players" in out.columns:
        out["Team"] = out["Team_players"]
    elif "Team_x" in out.columns:
        out["Team"] = out["Team_x"]

    if "TeamNick_players" in out.columns:
        out["TeamNick"] = out["TeamNick_players"]
    elif "TeamNick_x" in out.columns:
        out["TeamNick"] = out["TeamNick_x"]

    out["ProjectedAverage"] = out["ProjectionAverage"]
    out["ProjectedAverage"] = out["ProjectedAverage"].fillna(fallback_projection)

    expanded_rows = []
    unique_names = sorted(out["Name"].unique())
    name_to_bit = {name: i for i, name in enumerate(unique_names)}

    for _, row in out.iterrows():
        for pos in row["EligiblePositions"]:
            rec = row.to_dict()
            rec["Position"] = pos
            rec["PlayerKey"] = rec["Name"]
            rec["PlayerInternalID"] = name_to_bit[rec["Name"]]
            expanded_rows.append(rec)

    expanded = pd.DataFrame(expanded_rows)

    expanded["PosRankNum"] = expanded.groupby("Position")["ProjectedAverage"].rank(
        method="first",
        ascending=False
    ).astype(int)

    expanded["RankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['PosRankNum'])}",
        axis=1
    )

    expanded["ScoreRankNum"] = expanded.groupby("Position")["Score"].rank(
        method="first",
        ascending=False
    )

    expanded["ScoreRankLabel"] = expanded.apply(
        lambda r: f"{POS_PREFIX.get(r['Position'], r['Position'][0])}{int(r['ScoreRankNum'])}"
        if pd.notna(r["ScoreRankNum"]) else "",
        axis=1
    )

    expanded = expanded.reset_index(drop=True)
    expanded["RowID"] = expanded.index

    return expanded


def build_top3_bucket_playerkeys(expanded):
    top3 = {}

    for pos in ["DEF", "MID", "RK", "FWD"]:
        pos_df = expanded[expanded["Position"] == pos].copy()
        pos_df = pos_df.sort_values(["PosRankNum", "ProjectedAverage"], ascending=[True, False])
        pos_df = pos_df.drop_duplicates(subset=["PlayerKey"], keep="first")
        top3[pos] = pos_df.head(3)["PlayerKey"].tolist()

    return top3


# =========================================================
# EXPORTS
# =========================================================
def build_ranked_players_export(expanded):
    ranked = expanded.copy()
    ranked["Actual_or_Contest_Score"] = ranked["Score"]

    cols = [
        "Player ID", "Name", "Team", "Opponent", "Position",
        "RankLabel", "PosRankNum", "Salary", "ProjectedAverage", "ProjectionAverage",
        "Actual_or_Contest_Score", "Form", "Playing Status", "MatchMethod"
    ]
    cols = [c for c in cols if c in ranked.columns]
    ranked = ranked[cols].sort_values(["Position", "PosRankNum", "Salary", "Name"]).reset_index(drop=True)
    return ranked


def build_lineups_export(selected):
    rows = []

    for lineup_no, lineup in enumerate(selected, start=1):
        players = lineup["players"]
        by_pos = {"DEF": [], "MID": [], "RK": [], "FWD": []}
        for p in players:
            by_pos[p["Position"]].append(p)

        for pos in by_pos:
            by_pos[pos] = sorted(by_pos[pos], key=lambda x: safe_float(x["ProjectedAverage"]), reverse=True)

        row = {
            "Lineup_No": lineup_no,
            "Forced_Top3_Bucket": lineup.get("forced_top3_bucket", ""),
            "Total_Salary": lineup["salary"],
            "Total_Expected_Score": round(lineup["proj"], 2),
            "Average_Player_Avg": round(lineup["proj"] / 9, 2),
            "Actual_Score": round(lineup["actual"], 2) if lineup["actual"] > 0 else "",
        }

        for pos in ["DEF", "MID", "RK", "FWD"]:
            for idx, p in enumerate(by_pos[pos], start=1):
                base = f"{pos}{idx}"
                row[base] = p["Name"]
                row[f"{base}_Rank"] = p["RankLabel"]
                row[f"{base}_Avg"] = round(safe_float(p["ProjectedAverage"]), 2)
                row[f"{base}_Salary"] = safe_int(p["Salary"])
                row[f"{base}_Score"] = format_score(p["Score"])

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# SOLVER
# =========================================================
def solve_lineups_exact(expanded, settings):
    df = expanded.copy().reset_index(drop=True)

    row_ids = df["RowID"].tolist()
    row_to_record = df.set_index("RowID").to_dict("index")

    pos_to_rows = {
        "DEF": df.loc[df["Position"] == "DEF", "RowID"].tolist(),
        "MID": df.loc[df["Position"] == "MID", "RowID"].tolist(),
        "RK":  df.loc[df["Position"] == "RK", "RowID"].tolist(),
        "FWD": df.loc[df["Position"] == "FWD", "RowID"].tolist(),
    }

    player_to_rows = df.groupby("PlayerKey")["RowID"].apply(list).to_dict()
    team_to_rows = df.groupby("TeamNick")["RowID"].apply(list).to_dict()
    top3_bucket_playerkeys = build_top3_bucket_playerkeys(df)

    required_top3_count = math.ceil(settings["LINEUP_COUNT"] * settings["MIN_TOP3_BUCKET_SHARE"])
    rng = random.Random(settings["TOP3_BUCKET_RANDOM_SEED"])

    forced_bucket_schedule = {}
    if required_top3_count > 0:
        lineup_slots = list(range(1, settings["LINEUP_COUNT"] + 1))
        rng.shuffle(lineup_slots)
        forced_slots = sorted(lineup_slots[:required_top3_count])
        bucket_choices = ["DEF", "MID", "RK", "FWD"]
        for slot in forced_slots:
            forced_bucket_schedule[slot] = rng.choice(bucket_choices)

    if settings["USE_MANUAL_MAX_PLAYER_SELECTIONS"]:
        max_player_lineups = settings["MANUAL_MAX_PLAYER_SELECTIONS"]
    else:
        max_player_lineups = math.floor(settings["LINEUP_COUNT"] * settings["MAX_PLAYER_LINEUP_SHARE"])

    previous_lineups_playerkeys = []
    previous_lineups_by_pos_playerkeys = []
    solved_lineups = []
    player_lineup_counts = {player_key: 0 for player_key in player_to_rows.keys()}

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=settings["SOLVER_TIME_LIMIT"])

    progress = st.progress(0.0, text="Solving lineups...")

    for lineup_idx in range(1, settings["LINEUP_COUNT"] + 1):
        prob = pulp.LpProblem(f"Lineup_ProjectedAverage_{lineup_idx}", pulp.LpMaximize)
        x = {rid: pulp.LpVariable(f"x_{rid}", cat="Binary") for rid in row_ids}

        row_bonus, pos_order = build_tiebreak_bonus_map(
            df,
            lineup_idx,
            settings["RANDOM_SEED"],
            settings["POSITION_START_CYCLE"]
        )

        prob += pulp.lpSum(
            safe_float(row_to_record[rid]["ProjectedAverage"]) * x[rid] +
            settings["TIEBREAK_EPSILON"] * row_bonus.get(rid, 0) * x[rid]
            for rid in row_ids
        )

        prob += pulp.lpSum(safe_int(row_to_record[rid]["Salary"]) * x[rid] for rid in row_ids) <= settings["SALARY_CAP"]

        req_by_pos = {
            "DEF": settings["REQ_DEF"],
            "MID": settings["REQ_MID"],
            "FWD": settings["REQ_FWD"],
            "RK": settings["REQ_RK"],
        }
        for pos in pos_order:
            prob += pulp.lpSum(x[rid] for rid in pos_to_rows[pos]) == req_by_pos[pos]

        for _, rids in player_to_rows.items():
            prob += pulp.lpSum(x[rid] for rid in rids) <= 1

        if max_player_lineups >= 0:
            for player_key, used_count in player_lineup_counts.items():
                if used_count >= max_player_lineups:
                    rids = player_to_rows[player_key]
                    prob += pulp.lpSum(x[rid] for rid in rids) == 0

        if settings["MAX_PLAYERS_PER_TEAM"] is not None:
            team_names = list(team_to_rows.keys())
            random.Random(settings["RANDOM_SEED"] + lineup_idx + 999).shuffle(team_names)
            for team_name in team_names:
                rids = team_to_rows[team_name]
                prob += pulp.lpSum(x[rid] for rid in rids) <= settings["MAX_PLAYERS_PER_TEAM"]

        forced_bucket = forced_bucket_schedule.get(lineup_idx)
        if forced_bucket is not None:
            forced_playerkeys = top3_bucket_playerkeys.get(forced_bucket, [])
            forced_rows = []
            for player_key in forced_playerkeys:
                forced_rows.extend(player_to_rows.get(player_key, []))
            if forced_rows:
                prob += pulp.lpSum(x[rid] for rid in forced_rows) >= 1

        for prev_player_keys in previous_lineups_playerkeys:
            prev_rows_all_positions = []
            for player_key in prev_player_keys:
                prev_rows_all_positions.extend(player_to_rows[player_key])

            prob += pulp.lpSum(x[rid] for rid in prev_rows_all_positions) <= 8

            if settings["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"] > 1:
                prob += pulp.lpSum(x[rid] for rid in prev_rows_all_positions) <= 9 - settings["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"]

        if previous_lineups_by_pos_playerkeys:
            prev_by_pos = previous_lineups_by_pos_playerkeys[-1]
            pattern = settings["POSITIONAL_DIFFERENCE_PATTERN"][(lineup_idx - 2) % len(settings["POSITIONAL_DIFFERENCE_PATTERN"])]

            for pos, required_diff in pattern.items():
                if required_diff <= 0:
                    continue

                prev_pos_playerkeys = prev_by_pos.get(pos, [])
                if not prev_pos_playerkeys:
                    continue

                prev_pos_rows_all_positions = []
                for player_key in prev_pos_playerkeys:
                    prev_pos_rows_all_positions.extend(player_to_rows[player_key])

                max_overlap_allowed = len(prev_pos_playerkeys) - required_diff
                if max_overlap_allowed < 0:
                    max_overlap_allowed = 0

                prob += pulp.lpSum(x[rid] for rid in prev_pos_rows_all_positions) <= max_overlap_allowed

        status = prob.solve(solver)
        status_name = pulp.LpStatus[status]

        progress.progress(min(lineup_idx / settings["LINEUP_COUNT"], 1.0), text=f"Solving lineup {lineup_idx} of {settings['LINEUP_COUNT']}")

        if status_name != "Optimal":
            break

        selected_rids = [rid for rid in row_ids if pulp.value(x[rid]) > 0.5]

        if len(selected_rids) != 9:
            break

        players = [row_to_record[rid] for rid in selected_rids]
        selected_player_keys = [p["PlayerKey"] for p in players]

        prev_by_pos = {"DEF": [], "MID": [], "RK": [], "FWD": []}
        for p in players:
            prev_by_pos[p["Position"]].append(p["PlayerKey"])

        previous_lineups_playerkeys.append(selected_player_keys)
        previous_lineups_by_pos_playerkeys.append(prev_by_pos)

        for player_key in set(selected_player_keys):
            player_lineup_counts[player_key] += 1

        salary = sum(safe_int(p["Salary"]) for p in players)
        proj = sum(safe_float(p["ProjectedAverage"]) for p in players)
        actual = sum(safe_float(p["Score"], 0.0) for p in players if not pd.isna(p["Score"]))

        solved_lineups.append({
            "salary": salary,
            "proj": proj,
            "actual": actual,
            "players": players,
            "row_ids": selected_rids,
            "forced_top3_bucket": forced_bucket if forced_bucket is not None else "",
        })

    progress.empty()
    return solved_lineups


# =========================================================
# STREAMLIT GUI
# =========================================================
st.set_page_config(page_title="Enhanced Lineup Generator", layout="wide")
st.title("FPPG Lineup Generator")
st.caption("Upload your main players CSV, pre-game players CSV (used for FPPG), and optionally a Draftstars batch-edit CSV.")

with st.sidebar:
    st.header("Inputs")
    players_file = st.file_uploader("Main players CSV", type=["csv"])
    projection_file = st.file_uploader("Pre-game players CSV (uses FPPG)", type=["csv"])
    draftstars_file = st.file_uploader("Draftstars CSV (optional)", type=["csv"])

    st.header("Outputs")
    output_ranked_players = st.text_input("Ranked players filename", value="r4_col_bne_player_rankings.csv")
    output_lineups = st.text_input("Lineups filename", value="R4_col_bne4_6e.csv")
    output_draftstars = st.text_input("Updated Draftstars filename", value="Draftstars_UPDATED.csv")

    st.header("Core settings")
    salary_cap = st.number_input("Salary cap", value=DEFAULTS["SALARY_CAP"], step=1000)
    lineup_count = st.number_input("Lineup count", value=DEFAULTS["LINEUP_COUNT"], step=1, min_value=1)

    req_def = st.number_input("DEF required", value=DEFAULTS["REQ_DEF"], step=1, min_value=0)
    req_mid = st.number_input("MID required", value=DEFAULTS["REQ_MID"], step=1, min_value=0)
    req_rk = st.number_input("RK required", value=DEFAULTS["REQ_RK"], step=1, min_value=0)
    req_fwd = st.number_input("FWD required", value=DEFAULTS["REQ_FWD"], step=1, min_value=0)

    min_diff = st.number_input("Min different players from previous", value=DEFAULTS["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"], step=1, min_value=0)
    max_players_per_team = st.number_input("Max players per team", value=DEFAULTS["MAX_PLAYERS_PER_TEAM"], step=1, min_value=1)

    min_top3_bucket_share = st.number_input("Min top-3 bucket share", value=float(DEFAULTS["MIN_TOP3_BUCKET_SHARE"]), min_value=0.0, max_value=1.0, step=0.05)
    max_player_lineup_share = st.number_input("Max player lineup share", value=float(DEFAULTS["MAX_PLAYER_LINEUP_SHARE"]), min_value=0.0, max_value=1.0, step=0.05)

    playing_status_required_text = st.text_input(
        "Required Playing Status text",
        value=DEFAULTS["PLAYING_STATUS_REQUIRED_TEXT"]
    )

    st.header("Exposure toggle")
    use_manual_max_player_selections = st.checkbox("Use manual max lineups per player", value=False)
    manual_max_player_selections = st.number_input("Manual max lineups per player", value=40, step=1, min_value=1)

run_button = st.button("Generate lineups", type="primary")

if run_button:
    try:
        if players_file is None or projection_file is None:
            st.error("Please upload both the main players file and the pre-game players file.")
            st.stop()

        settings = DEFAULTS.copy()
        settings["SALARY_CAP"] = int(salary_cap)
        settings["LINEUP_COUNT"] = int(lineup_count)
        settings["REQ_DEF"] = int(req_def)
        settings["REQ_MID"] = int(req_mid)
        settings["REQ_RK"] = int(req_rk)
        settings["REQ_FWD"] = int(req_fwd)
        settings["MIN_DIFFERENT_PLAYERS_FROM_PREVIOUS"] = int(min_diff)
        settings["MAX_PLAYERS_PER_TEAM"] = int(max_players_per_team)
        settings["MIN_TOP3_BUCKET_SHARE"] = float(min_top3_bucket_share)
        settings["MAX_PLAYER_LINEUP_SHARE"] = float(max_player_lineup_share)
        settings["PLAYING_STATUS_REQUIRED_TEXT"] = str(playing_status_required_text).strip()
        settings["USE_MANUAL_MAX_PLAYER_SELECTIONS"] = bool(use_manual_max_player_selections)
        settings["MANUAL_MAX_PLAYER_SELECTIONS"] = int(manual_max_player_selections)

        t0 = time.time()

        st.info("Loading files...")
        players, projections = load_inputs(players_file, projection_file)

        st.info("Preparing player data...")
        players_df = prepare_players_df(players, settings["PLAYING_STATUS_REQUIRED_TEXT"])
        st.write(f"Players remaining after Playing Status filter: {len(players_df):,}")

        projection_df = prepare_projection_df(projections)
        expanded = match_players(players_df, projection_df, fallback_projection=settings["FALLBACK_PROJECTION"])

        unmatched = expanded[expanded["MatchMethod"] == ""]
        if len(unmatched) > 0:
            st.warning(
                f"{len(unmatched.drop_duplicates(subset=['Name']))} players did not match FPPG exactly. "
                f"They were assigned the fallback projected score of {settings['FALLBACK_PROJECTION']}."
            )

        st.info("Building ranked players export...")
        ranked_export = build_ranked_players_export(expanded)

        st.info("Solving lineups...")
        selected_projected = solve_lineups_exact(expanded, settings)

        if not selected_projected:
            st.error("No projected lineups were solved.")
            st.stop()

        lineups_export = build_lineups_export(selected_projected)

        st.success(f"Generated {len(lineups_export):,} lineups in {time.time() - t0:,.1f}s")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ranked players preview")
            st.dataframe(ranked_export.head(50), use_container_width=True)
        with col2:
            st.subheader("Lineups preview")
            st.dataframe(lineups_export.head(50), use_container_width=True)

        ranked_csv = ranked_export.to_csv(index=False).encode("utf-8")
        lineups_csv = lineups_export.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download ranked players CSV",
            data=ranked_csv,
            file_name=output_ranked_players,
            mime="text/csv"
        )

        st.download_button(
            "Download lineups CSV",
            data=lineups_csv,
            file_name=output_lineups,
            mime="text/csv"
        )

        if draftstars_file is not None:
            st.info("Building updated Draftstars CSV...")
            updated_draftstars_bytes, ds_meta = build_updated_draftstars_csv(
                draftstars_file_obj=draftstars_file,
                lineups_export=lineups_export,
                start_row=DEFAULTS["DRAFTSTARS_START_ROW"],
                start_col=DEFAULTS["DRAFTSTARS_START_COL"],
            )

            st.download_button(
                "Download updated Draftstars CSV",
                data=updated_draftstars_bytes,
                file_name=output_draftstars,
                mime="text/csv"
            )

            st.write(f"Draftstars Name/ID header detected using skiprows = {ds_meta['detected_skiprows']}")
            st.write(f"Draftstars lineups inserted: {ds_meta['inserted_count']:,}")

            if ds_meta["truncated"]:
                st.warning("Draftstars file ran out of rows before all lineups could be inserted.")

            if ds_meta["missing_id_names"]:
                st.warning(f"No Draftstars ID found for {len(ds_meta['missing_id_names'])} player(s).")
                st.dataframe(pd.DataFrame({"Missing Draftstars IDs": ds_meta["missing_id_names"]}), use_container_width=True)

    except Exception as e:
        st.error(f"ERROR: {e}")
        st.code(traceback.format_exc())
