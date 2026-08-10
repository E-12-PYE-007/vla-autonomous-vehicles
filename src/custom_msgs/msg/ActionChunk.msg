std_msgs/Header header
uint32 seq_num
# Frame this chunk was conditioned on most recently. Equals seq_num for publishers with a
# single observation per chunk (sys2). sys1 conditions on both seq_num's frame and this
# newer one, since it runs faster than sys2.
uint32 curr_img_seq_num
# Newest /cam frame once sys2 finished its forward pass, from AsyncHiddenState. With
# seq_num and curr_img_seq_num this gives the three frames spanning one sys2 inference.
uint32 end_img_seq_num
# Duration of that forward pass, on the node clock. Propagated unchanged by sys1.
float32 sys2_inference_ms
geometry_msgs/Pose2D[] relative_poses
