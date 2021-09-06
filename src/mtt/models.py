import tensorflow as tf


class MTTModel:
    models = {
        "DenseNet121": {"model": tf.keras.applications.DenseNet121, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        "DenseNet169": {"model": tf.keras.applications.DenseNet169, "perf": 0},
        "DenseNet201": {"model": tf.keras.applications.DenseNet201, "perf": 0},
        "EfficientNetB0": {"model": tf.keras.applications.EfficientNetB0, "perf": 0},
        "EfficientNetB1": {"model": tf.keras.applications.EfficientNetB1, "perf": 0},
        "EfficientNetB2": {"model": tf.keras.applications.EfficientNetB2, "perf": 0},
        "EfficientNetB3": {"model": tf.keras.applications.EfficientNetB3, "perf": 0},
        "EfficientNetB4": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB5": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB6": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "EfficientNetB7": {"model": tf.keras.applications.EfficientNetB4, "perf": 0},
        "InceptionResNetV2": {"model": tf.keras.applications.InceptionResNetV2, "perf": 0},
        "InceptionV3": {"model": tf.keras.applications.InceptionV3, "perf": 0},
        "MobileNet": {"model": tf.keras.applications.MobileNet, "perf": 0},
        "MobileNetV3Large": {"model": tf.keras.applications.MobileNetV3Large, "perf": 0},
        "MobileNetV3Small": {"model": tf.keras.applications.MobileNetV3Small, "perf": 0},
        "NASNetMobile": {"model": tf.keras.applications.NASNetMobile, "perf": 0},
        "ResNet101": {"model": tf.keras.applications.ResNet101, "perf": 0},
        "ResNet101V2": {"model": tf.keras.applications.ResNet101V2, "perf": 0},
        "ResNet152": {"model": tf.keras.applications.ResNet152, "perf": 0},
        "ResNet152V2": {"model": tf.keras.applications.ResNet152V2, "perf": 0},
        "ResNet50": {"model": tf.keras.applications.ResNet50, "perf": 0},
        "ResNet50V2": {"model": tf.keras.applications.ResNet50V2, "perf": 0},
        "VGG16": {"model": tf.keras.applications.VGG16, "perf": 0},
        "VGG19": {"model": tf.keras.applications.VGG19, "perf": 0},
        "Xception": {"model": tf.keras.applications.Xception, "perf": 0}
    }

    # Chosen by the Training of the "LongList"
    top_models = {
        # "MobileNet": {"model": tf.keras.applications.MobileNet, "perf": 0},
        "MobileNetV2": {"model": tf.keras.applications.MobileNetV2, "perf": 0},
        # "DenseNet201": {"model": tf.keras.applications.DenseNet201, "perf": 0},
        # "ResNet50V2": {"model": tf.keras.applications.ResNet50V2, "perf": 0},
        # "Xception": {"model": tf.keras.applications.Xception, "perf": 0},
        # "VGG16": {"model": tf.keras.applications.VGG16, "perf": 0},
    }
