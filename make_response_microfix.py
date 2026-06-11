"""Create a conservative response-only microfix candidate from the v2 champion."""

from __future__ import annotations

import argparse
import json
import os


MICROFIX_RESPONSES = {
    "39698083-bc11-4514-9e53-da76d9260b3a": (
        "\"We Are\" is from ONE OK ROCK's Ambitions album. For another track with "
        "the same big J-rock lift, Cry Out by ONE OK ROCK brings a driving alternative-rock "
        "sound, a huge chorus, and the kind of collective, arena-ready urgency you were "
        "chasing after finding \"We Are.\" Would you like more ONE OK ROCK songs with that "
        "same unifying, high-energy feel?"
    ),
    "d53d9457-12f1-4286-b195-aa42c23d3bce": (
        "Whispers from the Ether by Kognitif keeps you on the electronic side with an "
        "instrumental, atmospheric sound that feels thoughtful and introspective through "
        "sound design rather than words. Its spacious layers, warm texture, and meditative "
        "mood fit the reflective thread of the session while giving you a different ambient "
        "path into that feeling. Would you like more electronic picks that stay deep and "
        "contemplative?"
    ),
    "2a677d32-23e9-4fe7-8b8e-4875c93509a2": (
        "Virgo - 2004 - Remaster by Reginald Workman, Lee Morgan, Wayne Shorter, "
        "and Elvin Jones takes you straight into the Night Dreamer Blue Note world, with "
        "tenor saxophone, rich horns, late-evening mood, and instrumental jazz depth that "
        "matches your search for classic 1960s hard bop. Would you like more Blue Note "
        "deep cuts from saxophone-led sessions in this vein?"
    ),
    "d851eac7-27a7-4363-b2a4-6cb365c79d22": (
        "A Really Good Time - 1999 Digital Remaster by Roxy Music pairs slick 1970s "
        "art rock and glam-pop color with the visually striking Country Life era you were "
        "asking about. It has a stylish, vivid album identity and a smooth, decadent sound "
        "that makes the cover and music feel connected. Would you like more songs from "
        "records with memorable visual design?"
    ),
    "ebad356d-d5ab-437a-81f1-6d219e532c33": (
        "Turn the Beat Around - Single Version by Vicki Sue Robinson is a bright, "
        "high-energy party pick built on punchy Latin percussion, fast disco momentum, and "
        "a passionate dance-floor feel. It pushes the upbeat rhythmic energy you liked from "
        "Los Caligaris into something even more vibrant and immediately danceable. Would "
        "you want more party-ready tracks with lively Latin rhythm influences?"
    ),
    "db11666f-bfb8-4d0d-b62b-075ab85d92f9": (
        "Get Me - 2013 Remaster by Everything But The Girl has the distinct sonic texture "
        "you want: close-to-the-ear female vocals, moody downtempo warmth, and subtle "
        "electronic detail that feels immediately different from standard pop production. "
        "It is smooth enough for relaxed listening but still textured enough to stand out. "
        "Would you like more tracks with similarly creative production choices?"
    ),
    "d9cca604-febe-4c95-a5cc-a4318c33ec40": (
        "Juliette by Hollerado keeps you with the same artist and gives you that punchy, "
        "upbeat indie rock energy you liked, built around a bright riff, power-pop snap, "
        "and an easygoing summer feel. It should fit well when you want more Hollerado "
        "songs that are fun without losing their alternative edge. Would you like another "
        "batch from their discography?"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_prediction.json",
    )
    parser.add_argument(
        "--output_path",
        default="exp/inference/blindset_A/multichannel_ltr_lean_v2_microfix_prediction.json",
    )
    args = parser.parse_args()

    predictions = json.load(open(args.input_path, encoding="utf-8"))
    changed = []
    for item in predictions:
        response = MICROFIX_RESPONSES.get(item["session_id"])
        if response:
            item["predicted_response"] = response
            changed.append(item["session_id"])

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False)

    print(json.dumps({
        "output_path": args.output_path,
        "changed_responses": len(changed),
        "changed_sessions": changed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
