import os
dataset = "7scenes"#"7scenes"#cambridge


if dataset == "7scenes":

    checkpoint_path = "out/run_24_01_23_10_03_relformer_final.pth"
    log_name = checkpoint_path.replace("out/", "")
    out_filename = "{}_{}_logs.txt".format(dataset, log_name)
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    if not os.path.exists(out_filename):
        for s in scenes:
            cmd = "python main.py test /media/yoli/WDC-2.0-TB-Hard-/7Scenes ./models/backbones/efficient-net-b0.pth " \
                  "./datasets/7Scenes/7scenes_test_pairs/pairs_test_{}.csv 7scenes_config.json --checkpoint_path {} >> {}".format(s,
                                                                                                     checkpoint_path,
                                                                                                    out_filename)
            os.system(cmd)

else:
    checkpoint_path = "out/run_04_12_22_10_20_checkpoint-550.pth"#"out/run_05_12_22_10_04_checkpoint-100.pth"
    log_name = checkpoint_path.replace("out/", "")
    out_filename = "{}_{}_logs.txt".format(dataset, log_name)
    scenes = ['KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']

    if not os.path.exists(out_filename):
        for s in scenes:
            cmd = 'python main.py test /media/yoli/WDC-2.0-TB-Hard-/CambridgeLandmarks ./models/backbones/efficient-net-b0.pth ' \
                  './datasets/CambridgeLandmarks/pairs_test_{}.csv cambridge_config.json --checkpoint_path {} >> {}'.format(s,
                                                                                                                      checkpoint_path,
                                                                                                                      out_filename)
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
