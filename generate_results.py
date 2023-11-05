import os
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--checkpoint", help="", default="")
arg_parser.add_argument("--logname", help="logname", default="logname")
arg_parser.add_argument("--dataset", help="dataset", default="7scenes")#"7scenes")#cambridge
arg_parser.add_argument("--gpu", help="gpu id", default="6")
arg_parser.add_argument("--config_file", help="config_file", default="7scenes_config.json")
arg_parser.add_argument("--is_knn", help="is_knn", type=int, default="0")
args = arg_parser.parse_args()


if args.dataset == "7scenes":
    out_filename = "{}_{}_logs.txt".format(args.dataset, args.logname)
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    #scenes = ['chess']
    for s in scenes:
       cmd = "python main.py --mode=test  --dataset_path=/data/datasets/7Scenes/ --test_dataset_id=7scenes\
                  --test_labels_file=datasets/7Scenes_test_NN/NN_7scenes_{}.csv --config_file={} --checkpoint_path {} --gpu={} --is_knn={} >> {}".format(s,
                                                                                                     args.config_file,
                                                                                                     args.checkpoint,
                                                                                                     args.gpu,
                                                                                                    args.is_knn,
                                                                                                    out_filename)
       os.system(cmd)

else:
    out_filename = "{}_{}_logs.txt".format(args.dataset, args.logname)
    scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

    for s in scenes:
        cmd = "python main.py --mode=test --dataset_path=/data/datasets/CAMBRIDGE_dataset/  --test_dataset_id=Cambridge\
              --test_labels_file=datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv \
              --test_knn_file=datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv_with_netvlads.csv-knn-cambridge_four_scenes.csv_with_netvlads.csv \
              --config_file={} --checkpoint_path {} --gpu={} --is_knn=1 >> {}".format(
              s,s, args.config_file, args.checkpoint, args.gpu, out_filename)

        os.system(cmd)

f = open(out_filename)
lines = f.readlines()
i = 0

for l in lines:
    if "Median pose error: " in l:
        s = scenes[i]
        i += 1
        details = l.rstrip().split("Median pose error: ")[1]
        print("{} - {}".format(s, details))
