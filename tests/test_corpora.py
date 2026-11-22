import zipfile
from pathlib import Path

from fy_asr.corpora import (
    _speaker_groups,
    bhc_text,
    frisian_transcript,
    textgrid_tier,
)


def segment(recording: str, speaker: str | int, role: str) -> dict:
    return {
        "src_audio_filepath": recording,
        "speaker_id": speaker,
        "speaker_role": role,
    }


def test_shared_speakers_link_recordings_but_coordinator_and_unk_do_not() -> None:
    groups = _speaker_groups(
        [
            segment("a.m4a", 101, "Caregiver"),
            segment("a.m4a", 201, "Patient"),
            segment("b.m4a", 101, "Caregiver"),
            segment("b.m4a", 202, "Patient"),
            segment("c.m4a", 103, "Caregiver"),
            segment("c.m4a", 203, "Patient"),
            segment("a.m4a", "111", "Coordinator"),
            segment("c.m4a", "111", "Coordinator"),
            segment("b.m4a", "unk", "unk"),
            segment("c.m4a", "unk", "Patient"),
        ]
    )
    assert groups["a.m4a"] == groups["b.m4a"]
    assert groups["a.m4a"] != groups["c.m4a"]


def test_frisian_transcript_drops_headers_turn_stamps_and_dutch_translation(
    tmp_path: Path,
) -> None:
    paragraphs = [
        "Transkripsje Petear99 Test",
        "S = Soarchbieder",
        "C = Community manager",
        "00:01 C: Petear trije.",
        "00:04 S:  No,   goeie.<w:br/>00:06 P: Goeiedei.",
        "",
        "Oersetting / Vertaling",
        "00:01 C: Gesprek drie.",
        "00:04 S: Nou, goedendag.",
    ]
    body = "".join(
        f"<w:p><w:r><w:t>{p.replace('<w:br/>', '</w:t><w:br/><w:t>')}</w:t></w:r></w:p>"
        for p in paragraphs
    )
    docx = tmp_path / "t.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr(
            "word/document.xml", f"<w:document><w:body>{body}</w:body></w:document>"
        )
    assert frisian_transcript(docx) == "Petear trije. No, goeie. Goeiedei."


def test_bhc_text_follows_fris_conventions() -> None:
    assert bhc_text("mar at je in tsjer! eh sterke [jongen kom jij]") == (
        "mar at je in tsjer uh sterke jongen kom jij"
    )
    assert bhc_text("frij folle yn ´t Frysk ehm") == "frij folle yn 't Frysk uh"
    assert bhc_text("dat ha ik ... dus") is None
    assert bhc_text("  ") is None


def test_textgrid_tier_reads_multiline_and_quoted_text() -> None:
    grid = (
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n0\n5\n<exists>\n2\n'
        '"IntervalTier"\n"stimulus"\n0\n5\n1\n0\n5\n""\n'
        '"IntervalTier"\n"orthographic"\n0\n5\n2\n0\n1.5\n"no ""ja""\nsa"\n1.5\n5\n""\n'
    )
    assert textgrid_tier(grid, "orthographic") == [
        (0.0, 1.5, 'no "ja"\nsa'),
        (1.5, 5.0, ""),
    ]
