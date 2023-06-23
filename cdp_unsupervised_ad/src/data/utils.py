import os

from torch.utils.data import DataLoader

from data.cdp_dataset import get_split
from data.transforms import NormalizedTensorTransform

# Definitions
BAD_IDX_MOBILE = [158, 187, 198, 228, 240, 243, 252, 255, 258, 265, 267, 270, 274, 278, 281, 880, 104, 207, 222, 951, 1013, 1057, 1085, 1224, 1289, 1291, 267, 277, 945, 1165, 1217, 1237, 1337, 579, 584, 623, 624, 625, 637, 646, 659, 660, 662, 663, 665, 666, 691, 695, 700, 702, 703, 708, 711, 242, 244, 263, 267, 268, 288, 366, 472, 55, 79, 158, 195, 196, 199, 218, 228, 268, 280, 341, 917, 1084, 1241, 1312, 26, 269, 606, 1001, 1168, 1174, 1333, 147, 678, 952, 1338, 1349, 1356, 1422, 1438, 28, 142, 533, 534, 710, 967, 1024, 1252, 1336, 1361, 1433, 1434, 581, 583, 803, 829, 1181, 1195, 1230, 1237, 1251, 1253, 1266, 1285, 1297, 1348, 1372, 1396, 1397, 159, 414, 754, 759, 1157, 1371, 29, 229, 237, 238, 275, 323, 340, 882, 1122, 1238, 1264]


def load_cdp_data(args,
                  tp,
                  vp,
                  bs,
                  train_pre_transform=NormalizedTensorTransform(),
                  train_post_transform=None,
                  val_pre_transform=NormalizedTensorTransform(),
                  val_post_transform=None,
                  test_pre_transform=NormalizedTensorTransform(),
                  test_post_transform=None,
                  return_diff=False,
                  return_stack=False,
                  load=True,
                  bad_indexes=None
                  ):
    """Loads CDP data from the given directory, splitting according to the percentages and applying the transforms.
    Only loads the particular type of original selected. Returns the 3 data loaders and the number of fake codes."""
    t_dir = args["t_dir"]
    x_dirs = args["x_dirs"]
    f_dirs = args["f_dirs"]

    n_fakes = len(f_dirs)
    train_set, val_set, test_set = get_split(t_dir,
                                             x_dirs,
                                             f_dirs,
                                             train_percent=tp,
                                             val_percent=vp,
                                             train_pre_transform=train_pre_transform,
                                             train_post_transform=train_post_transform,
                                             val_pre_transform=val_pre_transform,
                                             val_post_transform=val_post_transform,
                                             test_pre_transform=test_pre_transform,
                                             test_post_transform=test_post_transform,
                                             return_diff=return_diff,
                                             return_stack=return_stack,
                                             load=load,
                                             bad_indexes=bad_indexes
                                             )
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True) if tp > 0 else None
    val_loader = DataLoader(val_set, batch_size=bs) if vp > 0 else None
    test_loader = DataLoader(test_set, batch_size=bs) if tp + vp < 1 else None

    return train_loader, val_loader, test_loader, n_fakes
