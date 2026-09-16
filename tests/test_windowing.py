"""Unit tests for the spec-critical sliding-window math (paper §2.2, §2.3)."""
import pytest

from fd_vad.windowing import (
    WindowConfig, num_chunks, window_bounds, all_windows,
    seconds_to_fbank_frames, chunk_label, boundary_chunk,
)

PAPER = WindowConfig(window_frames=256, stride_frames=32)


def test_paper_ms_conversions():
    assert PAPER.window_ms == 2560   # 256 frames @ 100Hz
    assert PAPER.stride_ms == 320    # 32 frames  @ 100Hz


def test_config_validation():
    with pytest.raises(ValueError):
        WindowConfig(window_frames=0, stride_frames=32)
    with pytest.raises(ValueError):
        WindowConfig(window_frames=32, stride_frames=64)  # stride > window


def test_num_chunks_ceildiv():
    assert num_chunks(0, PAPER) == 0
    assert num_chunks(1, PAPER) == 1
    assert num_chunks(32, PAPER) == 1
    assert num_chunks(33, PAPER) == 2
    assert num_chunks(100, PAPER) == 4     # ceil(100/32)
    assert num_chunks(256, PAPER) == 8


def test_window_rampup_then_saturate():
    N = 1000
    # c=1..8 ramp up to W_f, then window length stays == W_f
    assert window_bounds(1, N, PAPER) == (0, 32)      # r=32, start=max(0,32-256)=0
    assert window_bounds(8, N, PAPER) == (0, 256)     # r=256, start=0, len=256=W_f
    assert window_bounds(9, N, PAPER) == (32, 288)    # r=288, start=32, len=256
    for c in range(8, 20):
        s, e = window_bounds(c, N, PAPER)
        assert e - s == 256                            # saturated at W_f


def test_last_window_ends_at_N():
    N = 100
    windows = all_windows(N, PAPER)
    assert len(windows) == 4
    assert [w for _, w in windows] == [32, 64, 96, 100]  # r_c clamps to N
    assert windows[-1][1] == N


def test_window_index_errors():
    with pytest.raises(ValueError):
        window_bounds(0, 100, PAPER)   # c must be >= 1
    with pytest.raises(ValueError):
        window_bounds(1, 0, PAPER)     # need frames


def test_seconds_to_frames():
    assert seconds_to_fbank_frames(0.0) == 0
    assert seconds_to_fbank_frames(1.0) == 100   # 100 Hz
    assert seconds_to_fbank_frames(0.32) == 32   # one stride
    with pytest.raises(ValueError):
        seconds_to_fbank_frames(-0.1)


def test_chunk_labels_and_boundary():
    N, stop = 100, 80
    labels = [chunk_label(c, stop, N, PAPER) for c in range(1, num_chunks(N, PAPER) + 1)]
    assert labels == [0, 0, 1, 1]          # r=32,64,96,100 vs stop=80
    assert boundary_chunk(stop, N, PAPER) == 3

    # If stop is at the very end, only the last chunk is Stop.
    assert [chunk_label(c, 100, N, PAPER) for c in range(1, 5)] == [0, 0, 0, 1]
    assert boundary_chunk(100, N, PAPER) == 4

    # If stop==0 (degenerate), every chunk is Stop.
    assert all(chunk_label(c, 0, N, PAPER) == 1 for c in range(1, 5))
    assert boundary_chunk(0, N, PAPER) == 1


def test_160ms_ablation_config():
    # Table 5 ablation: smaller chunk size 160ms => stride 16 frames.
    small = WindowConfig(window_frames=256, stride_frames=16)
    assert small.stride_ms == 160
    assert num_chunks(256, small) == 16
