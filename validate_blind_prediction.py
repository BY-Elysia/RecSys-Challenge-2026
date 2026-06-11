"""Validate a Blind A prediction JSON before Codabench submission."""

from __future__ import annotations

import argparse
import json
import zipfile

from omegaconf import OmegaConf

from mcrs.db_item import MusicCatalogDB


FORBIDDEN_RESPONSE_WORDS = (
    "metadata",
    "ranking",
    "system",
    "tool",
    "sorry",
    "apolog",
)


def first_value(value):
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def validate_prediction(path: str, tid: str) -> dict:
    config = OmegaConf.load(f"config/{tid}.yaml")
    prediction = json.load(open(path, encoding="utf-8"))
    item_db = MusicCatalogDB(config.item_db_name, config.track_split_types, config.corpus_types)

    missing_title_or_artist = []
    forbidden = []
    bad_topk = []
    empty_response = []
    not_question = []
    word_counts = []
    seen_track_ids = set()

    for index, item in enumerate(prediction):
        track_ids = item.get("predicted_track_ids") or []
        if len(track_ids) != 20:
            bad_topk.append((index, len(track_ids)))
        seen_track_ids.update(track_ids)
        response = str(item.get("predicted_response") or "").strip()
        if not response:
            empty_response.append(index)
        if response and not response.endswith("?"):
            not_question.append(index)
        word_counts.append(len(response.split()))
        lowered = response.lower()
        if any(word in lowered for word in FORBIDDEN_RESPONSE_WORDS):
            forbidden.append(index)

        if track_ids:
            track = item_db.metadata_dict[track_ids[0]]
            title = first_value(track.get("track_name")).lower()
            artist = first_value(track.get("artist_name")).lower()
            if title and title not in lowered:
                missing_title_or_artist.append((index, "title", title))
            if artist and artist not in lowered:
                missing_title_or_artist.append((index, "artist", artist))

    return {
        "rows": len(prediction),
        "all_top20": not bad_topk,
        "bad_topk": bad_topk[:20],
        "empty_response": empty_response[:20],
        "not_question": not_question[:20],
        "missing_title_or_artist": missing_title_or_artist[:20],
        "missing_title_or_artist_count": len(missing_title_or_artist),
        "forbidden_response_word_indices": forbidden[:20],
        "forbidden_response_word_count": len(forbidden),
        "unique_recommended_tracks": len(seen_track_ids),
        "catalog_diversity_if_all_tracks_47071": len(seen_track_ids) / 47071,
        "word_count_min": min(word_counts) if word_counts else 0,
        "word_count_avg": sum(word_counts) / len(word_counts) if word_counts else 0,
        "word_count_max": max(word_counts) if word_counts else 0,
    }


def validate_zip(path: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return {
        "zip_entries": names,
        "has_prediction_json": names == ["prediction.json"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_path")
    parser.add_argument("--tid", default="bm25_tags_doubao_blindset_A")
    parser.add_argument("--zip_path", default=None)
    args = parser.parse_args()

    result = validate_prediction(args.prediction_path, args.tid)
    if args.zip_path:
        result.update(validate_zip(args.zip_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
