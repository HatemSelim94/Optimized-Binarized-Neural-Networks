mkdir -p /home/hatem/Projects/server/final_repo/final_repo/eval/darts/sub_experiments/road_test_2/bev;
python transform2BEV.py "/home/hatem/Projects/server/final_repo/final_repo/eval/darts/sub_experiments/road_test_2/probs/*.png" \
"/home/hatem/Projects/server/final_repo/final_repo/data/data_road/testing/calib/" \
"/home/hatem/Projects/server/final_repo/final_repo/eval/darts/sub_experiments/road_test_2/bev"