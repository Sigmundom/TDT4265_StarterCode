from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import csv

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def size_box(bb_list) :

    x_left = bb_list[0]
    x_right = bb_list[2]
    y_top = bb_list[1]
    y_bottom = bb_list[3]
    width = x_right - x_left
    height = y_bottom - y_top

    area = width*height

    return(area)

def aspect_ratio_box(bb_list, scale=True) :

    if scale :
        x_left = bb_list[0] * 1024
        x_right = bb_list[2] * 1024
        y_top = bb_list[1] * 128
        y_bottom = bb_list[3] * 128

    else :
        x_left = bb_list[0]
        x_right = bb_list[2]
        y_top = bb_list[1]
        y_bottom = bb_list[3]

    width = x_right - x_left
    height = y_bottom - y_top

    ar = width/height

    return(ar)



def analyze_something(dataloader, cfg):
    labels_classes = ['total', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider']

    #for all the lists below, the list[0] corresponds to the total without distinction of class
    sum_size = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum_ar = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum_ars = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    nb = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    lst_size = [[], [], [], [], [], [], [], [], []]
    lst_ar = [[], [], [], [], [], [], [], [], []]
    lst_ars = [[], [], [], [], [], [], [], [], []]




    for batch in tqdm(dataloader):
        boxes = batch['boxes']
        [boxes] = boxes.tolist()

        for k in range(len(list(boxes))) :
            labels = batch['labels'][0]
            labels = labels.tolist()
            label = labels[k]

            size = size_box(list(boxes)[k])
            ar = aspect_ratio_box(list(boxes)[k], scale=False)
            ars = aspect_ratio_box(list(boxes)[k], scale=True)

            sum_size[label] += size
            sum_ar[label] += ar
            sum_ars[label] += ars

            nb[label] += 1

            sum_size[0] += size
            sum_ar[0] += ar
            sum_ars[0] += ars
            nb[0] += 1

            lst_size[0] += [size]
            lst_size[label] += [size]

            lst_ar[0] += [ar]
            lst_ar[label] += [ar]

            lst_ars[0] += [ars]
            lst_ars[label] += [ars]


        #exit()

    print(nb)

    for i in range(len(sum_size)) :
        if nb[i] != 0 :
            print("Average BB size for label", labels_classes[i], ":", sum_size[i]/nb[i])
            print("Average aspect ratio for label", labels_classes[i], ":", sum_ar[i]/nb[i])
            print("Average aspect ratio for label, SCALED", labels_classes[i], ":", sum_ars[i]/nb[i])


    with open("histo_data_size.csv", "w") as f1:
        wr = csv.writer(f1)
        wr.writerows(lst_size)

    with open("histo_data_ar.csv", "w") as f2:
        wr = csv.writer(f2)
        wr.writerows(lst_ar)

    with open("histo_data_ars.csv", "w") as f3:
        wr = csv.writer(f3)
        wr.writerows(lst_ars)



def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)


if __name__ == '__main__':
    main()
