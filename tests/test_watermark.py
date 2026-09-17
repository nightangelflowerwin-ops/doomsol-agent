import numpy as np

from jevlike.watermark import stamp_frame


def test_stamp_frame_preserves_shape_and_changes_pixels():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    marked = stamp_frame(frame)
    assert marked.shape == frame.shape
    assert marked.dtype == frame.dtype
    assert np.any(marked != frame)
