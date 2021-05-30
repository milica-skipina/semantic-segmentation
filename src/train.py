from data_loader.data_loader import Dataset

dataset = Dataset('../data/raw/leftImg8bit/train', '../data/raw/gtFine/train')
x, y = dataset.load_data()

