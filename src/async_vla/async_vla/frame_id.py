#!/usr/bin/env python3
"""Shared frame identifier for sim mode.

On hardware the camera node stamps every frame with a seq_num that both sys1 and sys2
receive, so they agree on which image is which. In sim /cam is a plain
sensor_msgs/Image with no such field, and each node used to fabricate its own counter.
Because sys2 spends ~8 s loading the VLA before it starts counting, the two numberings
drifted apart by roughly IMAGE_BUFFER_SIZE frames; sys1 then looked up sys2's number in
its own buffer, missed every time, and never published an action chunk.

The pairing has to be exact rather than approximate: sys1 combines the hidden state with
the very frame sys2 computed it from, and the delta against the current frame is what
compensates for sys2's inference latency. Pairing the wrong frames would silently
corrupt that correction.
"""


def sim_seq_num(header):
    """Deterministic frame id from an image header, identical across nodes.

    10 ms resolution (frames arrive at ~10 Hz so ids stay unique), wraps every 20000 s,
    and stays well inside the int32 range of ImageWithSeqNum.img_seq_num.
    """
    return (header.stamp.sec % 20000) * 100 + header.stamp.nanosec // 10_000_000
