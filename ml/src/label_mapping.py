CLASS_LABELS = {
    "Healthy": 0,
    "Alternaria alternata": 1,
    "Alternaria solani": 2,
    "Botrytis cinerea": 3,
    "Fusarium oxysporum": 4
}


def get_label(class_name):
    if class_name not in CLASS_LABELS:
        raise ValueError("Unknown disease class")

    return CLASS_LABELS[class_name]


def get_class_name(label):
    for name, value in CLASS_LABELS.items():
        if value == label:
            return name

    raise ValueError("Unknown class label")


def get_all_classes():
    return CLASS_LABELS.copy()