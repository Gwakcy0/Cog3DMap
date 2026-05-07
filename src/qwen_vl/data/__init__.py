import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

SPAR = {
    "annotation_path": "data/train/spar_7m.jsonl",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K = {
    "annotation_path": "data/train/spar_234k.json",
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND = {
    "annotation_path": "data/train/llava_hound_255k.json",
    "data_path": "data/media",
    "tag": "2d"
}

LLAVA_HOUND_64K = {
    "annotation_path": "data/train/llava_hound_64k.json",
    "data_path": "data/media",
    "tag": "2d"
}

SCANNET_DET = {
    "annotation_path": "data/train/scannet_det_train_4frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCANREFER = {
    "annotation_path": "data/train/scanrefer_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP = {
    "annotation_path": "data/train/scan2cap_train_32frames.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP_POINT3R = {
    "annotation_path": "data/train/scan2cap_train_32frames_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCAN2CAP_POINT3R_DEBUG = {
    "annotation_path": "data/train/scan2cap_debug_32frames_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

SCANQA_POINT3R = {
    "annotation_path": "data/train/scanqa_train_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

SQA3D_POINT3R = {
    "annotation_path": "data/train/sqa3d_train_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

BEACON3D_POINT3R = {
    "annotation_path": "data/train/beacon3d_train_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

VSIBENCH = {
    "annotation_path": "data/train/vsibench_train.json",
    "data_path": "data/media",
    "tag": "3d"
}

VSIBENCH_POINT3R = {
    "annotation_path": "data/train/vsibench_train_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

VSTIBENCH = {
    "annotation_path": "data/train/vstibench_train.json",
    "data_path": "data/media",
    "tag": "3d"
}

VSTIBENCH_POINT3R = {
    "annotation_path": "data/train/vstibench_train_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_SUBSET_POINT3R = {
    "annotation_path": "data/train/spar_subset_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

VSIBENCH_BALANCED_POINT3R = {
    "annotation_path": "data/train/vsibench_train_balanced_point3r.json",
    "data_path": "data/media",
    "tag": "3d"
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "spar": SPAR,
    "llava_hound": LLAVA_HOUND,
    "scannet_det": SCANNET_DET,
    "scanrefer": SCANREFER,
    "scan2cap": SCAN2CAP,
    "scan2cap_point3r": SCAN2CAP_POINT3R,
    "scan2cap_point3r_debug": SCAN2CAP_POINT3R_DEBUG,
    "scanqa_point3r": SCANQA_POINT3R,
    "sqa3d_point3r": SQA3D_POINT3R,
    "beacon3d_point3r": BEACON3D_POINT3R,
    "spar_234k": SPAR_234K,
    "llava_hound_64k": LLAVA_HOUND_64K,
    "vsibench": VSIBENCH,
    "vsibench_point3r": VSIBENCH_POINT3R,
    "vstibench": VSTIBENCH,
    "vstibench_point3r": VSTIBENCH_POINT3R,
    "spar_subset_point3r": SPAR_SUBSET_POINT3R,
    "vsibench_balanced_point3r": VSIBENCH_BALANCED_POINT3R,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
