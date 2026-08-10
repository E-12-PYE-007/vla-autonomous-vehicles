std_msgs/Header header
uint32 seq_num
# Frame this chunk was actually conditioned on most recently. Equals seq_num for
# publishers with a single observation per chunk (sys2). sys1 conditions on both
# seq_num's frame (shared hidden state input) and this newer one, since it runs
# faster than sys2 and a fresher /cam frame is usually available by publish time.
uint32 curr_img_seq_num
geometry_msgs/Pose2D[] relative_poses
