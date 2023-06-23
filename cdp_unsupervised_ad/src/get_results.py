"""Obtains images and tables to be added into a scientific paper."""

import os
import cv2
import numpy as np

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from skimage.metrics import structural_similarity

import json

sns.set_theme(style='white', font_scale=3)

PROJECT_DIR = "/Users/bp/Desktop/Projects/cdp_unsupervised_ad/"

# Templates
TEMPLATES_PATH = os.path.join(PROJECT_DIR, "datasets/mobile/orig_template")

# Mobile codes
MOBILE_PATH = os.path.join(PROJECT_DIR, "datasets/mobile/")

MOBILE_ORIG_PATH = os.path.join(MOBILE_PATH, "orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG")

ORIGINAL_IPHONE_PATHS = [
    (f"Original iPhone ({run})",
     os.path.join(MOBILE_ORIG_PATH, f"iPhone12Pro_run{run}_ss100_focal12_apperture1/rcod_hist"))
    for run in range(1, 7)]

ORIGINAL_SAMSUNG_PATHS = [
    (f"Original Samsung ({run})",
     os.path.join(MOBILE_ORIG_PATH, f"SamsungGN20U_run{run}_ss100_focal12_apperture1/rcod_hist"))
    for run in range(1, 7)]

MOBILE_SYN_PATH = os.path.join(MOBILE_PATH, "synthetic_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG/seed_0")

SYNTHETIC_IPHONE_PATHS = [
    (f"Synthetic iPhone ({run})",
     os.path.join(MOBILE_SYN_PATH, f"iPhone12Pro_run{run}_ss100_focal12_apperture1"))
    for run in range(1, 7)]

SYNTHETIC_SAMSUNG_PATHS = [
    (f"Synthetic Samsung ({run})",
     os.path.join(MOBILE_SYN_PATH, f"SamsungGN20U_run{run}_ss100_focal12_apperture1"))
    for run in range(1, 7)]

MOBILE_FAKE_PATH = os.path.join(MOBILE_PATH, "fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55")

FAKE_IPHONE_PATHS = [
    (f"Fake iPhone ({run})",
     os.path.join(MOBILE_FAKE_PATH, f"iPhone12Pro_run{run}_ss100_focal12_apperture1/rcod_hist"))
    for run in range(1, 7)]

FAKE_SAMSUNG_PATHS = [
    (f"Fake Samsung ({run})",
     os.path.join(MOBILE_FAKE_PATH, f"SamsungGN20U_run{run}_ss100_focal12_apperture1/rcod_hist"))
    for run in range(1, 7)]

# Scanner codes
SCANNER_PATH = os.path.join(PROJECT_DIR, "datasets/scanner")

ORIG_SCANNER_PATHS = [
    (f"Original scanner ({nr})",
     os.path.join(SCANNER_PATH, f"originals_{nr}"))
    for nr in [55, 76]
]

FAKE_SCANNER_PATHS = [
    (f"Fake scanner ({nr1}/{nr2})",
     os.path.join(SCANNER_PATH, f"fakes_{nr1}_{nr2}"))
    for nr1, nr2 in [(55, 55), (55, 76), (76, 55), (76, 76)]
]

SYN_SCANNER_PATHS = [
    (f"Synthetic scanner ({nr})",
     os.path.join(SCANNER_PATH, f"synthetic_{nr}/seed_0"))
    for nr in [55, 76]
]

ALL_FILE_NAMES = set(os.listdir(ORIGINAL_IPHONE_PATHS[0][1]))
for run in range(6):
    ALL_FILE_NAMES = ALL_FILE_NAMES.intersection(set(os.listdir(ORIGINAL_IPHONE_PATHS[run][1])))
    ALL_FILE_NAMES = ALL_FILE_NAMES.intersection(set(os.listdir(ORIGINAL_SAMSUNG_PATHS[run][1])))
    ALL_FILE_NAMES = ALL_FILE_NAMES.intersection(set(os.listdir(FAKE_IPHONE_PATHS[run][1])))
    ALL_FILE_NAMES = ALL_FILE_NAMES.intersection(set(os.listdir(FAKE_SAMSUNG_PATHS[run][1])))

for i in range(4):
    if i < 2:
        ALL_FILE_NAMES = ALL_FILE_NAMES.intersection((set(os.listdir(ORIG_SCANNER_PATHS[i][1]))))
        ALL_FILE_NAMES = ALL_FILE_NAMES.intersection((set(os.listdir(SYN_SCANNER_PATHS[i][1]))))
    ALL_FILE_NAMES = ALL_FILE_NAMES.intersection((set(os.listdir(FAKE_SCANNER_PATHS[i][1]))))

ALL_FILE_NAMES = sorted(list(ALL_FILE_NAMES))
print(f"Found {len(ALL_FILE_NAMES)} file names which are present in all folders.")

METRICS = ["MSE", "SSIM", "PCorr", "NPCorr"]
METRICS_TO_PARAM = {
    "MSE": r"$\alpha$",
    "SSIM": r"$\beta$",
    "PCorr": r"$\gamma$",
    "NPCorr": r"$\delta$"
}


# Metrics
def mse(x, y):
    return np.mean((x - y) ** 2)


def pcorr(x, y):
    return cv2.matchTemplate(x.flatten(), y.flatten(), cv2.TM_CCOEFF).ravel()[0]


def normalized_pcorr(x, y):
    return cv2.matchTemplate(x.flatten(), y.flatten(), cv2.TM_CCOEFF_NORMED).ravel()[0]


def ssim(x, y):
    return structural_similarity(x.squeeze(), y.squeeze())


def load_images(path):
    images = []
    for fn in ALL_FILE_NAMES:
        img = cv2.imread(os.path.join(path, fn), cv2.IMREAD_GRAYSCALE).astype(np.float32)  # / 255.
        img = cv2.resize(img, (228, 228))
        img = np.expand_dims(img, 2)
        images.append((fn, img))
    return images


def get_score(metric, target, originals, fakes):
    fn = None
    if metric == "MSE":
        fn = mse
    elif metric == "SSIM":
        fn = ssim
    elif metric == "PCorr":
        fn = pcorr
    else:
        fn = normalized_pcorr

    y_true, y_score = [], []
    for t, o in zip(target, originals):
        t, o = t[1], o[1]
        y_true.append(0 if metric == "MSE" else 1)
        y_score.append(fn(t, o))

    for t, f in zip(target, fakes):
        t, f = t[1], f[1]
        y_true.append(1 if metric == "MSE" else 0)
        y_score.append(fn(t, f))

    return roc_curve(y_true, y_score)


def get_all_scores(templates, o_iphone, o_samsung, o_scanner, f_iphone, f_samsung, f_scanner, s_iphone, s_samsung,
                   s_scanner):
    scores = {}

    # Comparing with templates
    for orig, fakes in tqdm([(o_iphone, f_iphone), (o_samsung, f_samsung), (o_scanner, f_scanner)]):
        for o_name, o in orig:
            for f_name, f in fakes:
                for metric in METRICS:
                    fpr, tpr, threshold = get_score(metric, templates[1], o, f)
                    scores[str(("template", o_name, f_name, metric, "fpr"))] = list(fpr)
                    scores[str(("template", o_name, f_name, metric, "tpr"))] = list(tpr)
                    scores[str(("template", o_name, f_name, metric, "threshold"))] = list(threshold.astype(np.float64))

    # Comparing with originals and synthetics
    for target, orig, fakes in tqdm(
            [(o_iphone, o_iphone, f_iphone),
             (s_iphone, o_iphone, f_iphone),
             (o_samsung, o_samsung, f_samsung),
             (s_samsung, o_samsung, f_samsung),
             (o_scanner, o_scanner, f_scanner),
             (s_scanner, o_scanner, f_scanner)
             ],
            desc="Computing scores"):

        for t_name, t in tqdm(target):
            for o_name, o in orig:
                for f_name, f in fakes:
                    for metric in METRICS:
                        fpr, tpr, threshold = get_score(metric, t, o, f)
                        scores[str((t_name, o_name, f_name, metric, "fpr"))] = list(fpr)
                        scores[str((t_name, o_name, f_name, metric, "tpr"))] = list(tpr)
                        scores[str((t_name, o_name, f_name, metric, "threshold"))] = list(threshold.astype(np.float64))
    return scores


def cdps_imgs(templates, scanner, iphone, samsung, idx=0):
    pass


def pearson_correlation_tables(target, originals, fakes, synthetics):
    def mean_npcorr(target, other):
        n = len(target)
        mean = 0
        for t, o in zip(target, other):
            mean += normalized_pcorr(t[1], o[1]) / n
        return round(mean, 2)

    table = []
    for _, t in tqdm(target, desc="Getting Pearson Correlations"):
        line = []
        for _, o in originals:
            line.append(mean_npcorr(t, o))
        for _, f in fakes:
            line.append(mean_npcorr(t, f))
        for _, s in synthetics:
            line.append(mean_npcorr(t, s))

        table.append(line)
    return table


def roc_auc_tables(scores, metric="MSE"):
    fakes = [f"Fake {phone} ({nr})" for phone in ["iPhone", "Samsung"] for nr in range(1, 7)]
    for auth in ["template", "Original", "Synthetic"]:
        for phone in ["iPhone", "Samsung"]:
            table = np.empty((6, 6))
            for i in range(6):
                for j in range(6):
                    aucs = []

                    key_auth = auth + f" {phone} ({i + 1})" if auth != "template" else auth
                    key_o = f"Original {phone} ({j + 1})"
                    key_m = metric
                    keys = [(str((key_auth, key_o, fake, key_m, "fpr")),
                             str((key_auth, key_o, fake, key_m, "tpr"))) for fake in fakes]

                    for fpr_k, tpr_k in keys:
                        if fpr_k in scores.keys():
                            aucs.append(auc(scores[fpr_k], scores[tpr_k]))

                    table[i][j] = round(np.mean(aucs), 2)

            if "iPhone" in phone:
                print(f"{auth} - {phone} ({metric})")
                for line in table:
                    print(line)
                    if auth == "template":
                        break
                print(f"Mean --> {np.mean(table):.2f}")
                print("\n")


def perror_plots(scores, plot_type=None):
    # https://matplotlib.org/stable/gallery/scales/custom_scale.html#sphx-glr-gallery-scales-custom-scale-py
    # Perror plots
    for auth in ["template", "Original", "Synthetic"]:
        os.makedirs(f"{auth}", exist_ok=True)
        for phone in ["iPhone", "Samsung"]:
            for metric in METRICS:
                # Making a Pe plot for the authentication method, the phone and the metric
                keys = []
                for key in scores.keys():
                    k_auth, k_o, k_f, k_m, _ = key.split(",")
                    if \
                            auth in k_auth and \
                                    (phone in k_auth or auth == "template") \
                                    and phone in k_o \
                                    and phone in k_f \
                                    and metric in k_m \
                                    and k_auth.split("'")[1] != k_o.split("'")[1]:
                        keys.append(key)

                fprs = [scores[k] for k in keys if "fpr" in k]
                tprs = [scores[k] for k in keys if "tpr" in k]
                ts = [scores[k] for k in keys if "threshold" in k]

                # Plotting options
                plt.figure(figsize=(10, 10))
                plt.grid(visible=True, which="both")
                plt.yscale("log")

                for fpr, tpr, t in zip(fprs, tprs, ts):
                    fpr, tpr, t = np.array(fpr), np.array(tpr), np.array(t)

                    if plot_type == "roc":
                        plt.plot(fpr, tpr)
                        plt.xlabel(r"$P_{fa}$")
                        plt.ylabel(r"$1 - P_{miss}$")
                    else:
                        p_error = (6 * fpr + (1 - tpr)) / 7
                        # plt.xscale("log")
                        plt.plot(t, p_error)
                        plt.xlabel(METRICS_TO_PARAM[metric])
                        plt.ylabel(r"$P_e$")

                # plt.legend(prop={"size": 18})
                plt.savefig(f"{auth}/{phone}_{metric}", bbox_inches='tight')
                plt.show()


def main():
    # Images
    templates = ("Templates", load_images(TEMPLATES_PATH))

    desc = "Loading printed CDPs "
    o_iphone = [(name, load_images(path)) for name, path in tqdm(ORIGINAL_IPHONE_PATHS, desc=f"{desc} (1/9)")]
    o_samsung = [(name, load_images(path)) for name, path in tqdm(ORIGINAL_SAMSUNG_PATHS, desc=f"{desc} (2/9)")]

    f_iphone = [(name, load_images(path)) for name, path in tqdm(FAKE_IPHONE_PATHS, desc=f"{desc} (3/9)")]
    f_samsung = [(name, load_images(path)) for name, path in tqdm(FAKE_SAMSUNG_PATHS, desc=f"{desc} (4/9)")]

    s_iphone = [(name, load_images(path)) for name, path in tqdm(SYNTHETIC_IPHONE_PATHS, desc=f"{desc} (5/9)")]
    s_samsung = [(name, load_images(path)) for name, path in tqdm(SYNTHETIC_SAMSUNG_PATHS, desc=f"{desc} (6/9)")]

    o_scanner = [(name, load_images(path)) for name, path in tqdm(ORIG_SCANNER_PATHS, desc=f"{desc} (7/9)")]
    f_scanner = [(name, load_images(path)) for name, path in tqdm(FAKE_SCANNER_PATHS, desc=f"{desc} (8/9)")]
    s_scanner = [(name, load_images(path)) for name, path in tqdm(SYN_SCANNER_PATHS, desc=f"{desc} (9/9)")]

    # Getting all scores
    if not os.path.isfile("scores.json"):
        scores = get_all_scores(templates,
                                o_iphone, o_samsung, o_scanner,
                                f_iphone, f_samsung, f_scanner,
                                s_iphone, s_samsung, s_scanner)

        with open("scores.json", "w") as file:
            json.dump(scores, file)

    # Loading the scores
    print("Loading scores...")
    file = open("scores.json", "r")
    scores = json.load(file)
    file.close()
    print("Scores loaded.")

    # Storing example of CDPs
    cdps_imgs(templates, o_scanner, o_iphone, o_samsung)
    cdps_imgs(templates, f_scanner, f_iphone, f_samsung)
    cdps_imgs(templates, s_scanner, s_iphone, s_samsung)

    # Person Correlation Tables (iPhone and Samsung)
    print("\n\nPerson correlation table (iPhone)")
    for line in pearson_correlation_tables(o_iphone, o_iphone, f_iphone, s_iphone):
        print(line)
    print("\n\nPerson correlation table (Samsung)")
    for line in pearson_correlation_tables(o_samsung, o_samsung, f_samsung, s_samsung):
        print(line)

    # ROC-AUC Tables
    roc_auc_tables(scores, "NPCorr")

    # Plots on the probability of error
    perror_plots(scores)

    # Plotting ROC curves
    perror_plots(scores, "roc")


if __name__ == '__main__':
    main()
