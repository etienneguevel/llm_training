import torchvision.transforms as v2


def image_transform_training(resize_shape: int, croping_shape):
    transform = v2.Compose([
        v2.Resize(size=resize_shape, interpolation=v2.InterpolationMode.BICUBIC),
        v2.RandomCrop(croping_shape),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform

