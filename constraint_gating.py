"""High-precision request-constraint detection and locked ranking merge helpers."""

from __future__ import annotations

import re

from train_ltr_ranker import normalize_text


CONSTRAINT_CATEGORIES = (
    "year",
    "same_artist",
    "different_artist",
    "same_album",
    "instrumental",
    "live",
    "remix",
)


def _contains_any(query: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in query for phrase in phrases)


def strict_constraint_categories(request: str) -> set[str]:
    """Return only constraints expressed by unambiguous recommendation phrases."""
    query = f" {normalize_text(request)} "
    categories = set()
    if re.search(r"\b(?:19|20)\d{2}\b", query) or re.search(r"\b(?:19|20)\d0s\b", query):
        categories.add("year")
    if _contains_any(query, (
        " same artist ", " same band ", " same singer ", " more by ",
        " another by ", " by them ", " songs by ", " tracks by ",
    )):
        categories.add("same_artist")
    if _contains_any(query, (
        " different artist ", " different band ", " different singer ",
        " completely different artist ", " other artists ", " other bands ",
        " new artists ", " no more ",
    )):
        categories.add("different_artist")
    explicit_same_album = _contains_any(query, (
        " same album ", " another track from ", " another song from ",
        " deep cuts are there from ",
    ))
    album_with_recommendation_intent = (
        _contains_any(query, (" this album ", " that album ", " from the album ", " on the album "))
        and _contains_any(query, (
            " recommend ", " suggest ", " keep them coming ", " how about ",
            " play another ", " give me another ",
        ))
    )
    if explicit_same_album or album_with_recommendation_intent:
        categories.add("same_album")
    if _contains_any(query, (
        " instrumental ", " instrumentals ", " no vocals ", " without vocals ",
        " soundscape without any beats ",
    )):
        categories.add("instrumental")
    if _contains_any(query, (
        " live version ", " live recording ", " concert recording ",
        " performed live ", " recorded live ",
    )):
        categories.add("live")
    if _contains_any(query, (" remix ", " remixed ", " remix version ")):
        categories.add("remix")
    return categories


def locked_prefix_merge(
    base_tracks: list[str],
    candidate_tracks: list[str],
    lock_prefix: int,
    topk: int = 20,
) -> list[str]:
    merged = list(base_tracks[:lock_prefix])
    for track_id in candidate_tracks:
        if track_id not in merged:
            merged.append(track_id)
        if len(merged) == topk:
            break
    for track_id in base_tracks:
        if track_id not in merged:
            merged.append(track_id)
        if len(merged) == topk:
            break
    if len(merged) != topk:
        raise RuntimeError(f"Only produced {len(merged)} tracks, expected {topk}.")
    return merged
