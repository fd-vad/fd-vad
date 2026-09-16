"""Tests for clip→chunk label derivation (docs/labeling.md, S2 scheme)."""
from fd_vad.schema import Utterance
from fd_vad.windowing import WindowConfig
from fd_vad.labeling import clip_chunk_labels, label_sanity

W = WindowConfig(256, 32)  # paper defaults


def _utt(sem, dur, stop):
    return Utterance(id="x", sem_class=sem, text="", source_text="", audio_path="",
                     sample_rate=16000, duration_s=dur, stop_speaking_s=stop, speaker="d")


def test_complete_has_contiguous_stop_tail():
    u = _utt("complete", 3.0, 2.7)   # n=300 frames, stop_frame=270
    labels = clip_chunk_labels(u, W)
    assert labels[-1] == 1 and labels[0] == 0
    assert sum(labels) >= 1
    # contiguous tail: once it turns 1 it stays 1
    first = labels.index(1)
    assert all(v == 1 for v in labels[first:])
    assert label_sanity(u, W) == []


def test_incomplete_all_continue():
    u = _utt("incomplete", 3.0, 3.0)
    labels = clip_chunk_labels(u, W)
    assert set(labels) == {0}
    assert label_sanity(u, W) == []


def test_complete_boundary_near_zero_flags():
    u = _utt("complete", 3.0, 0.05)   # stop_frame=5 -> every chunk Stop
    assert set(clip_chunk_labels(u, W)) == {1}
    assert any("ALL Stop" in e for e in label_sanity(u, W))


def test_complete_boundary_past_end_flags():
    # stop beyond duration -> 0 stop chunks -> flagged
    u = _utt("complete", 3.0, 3.5)
    problems = label_sanity(u, W)
    assert any("0 Stop" in e or "outside" in e for e in problems)


def test_incomplete_with_stray_stop_would_flag():
    # sanity is keyed on sem_class; incomplete must have 0 stop regardless of stop_speaking_s
    u = _utt("incomplete", 3.0, 1.0)
    assert label_sanity(u, W) == []
    assert sum(clip_chunk_labels(u, W)) == 0
