import fiftyone as fo
import fiftyone.zoo as foz
import os

# dataset = foz.load_zoo_dataset(
#     "open-images-v6",
#     split="validation",
#     max_samples=100,
#     seed=51,
#     shuffle=True,
# )

# session = fo.launch_app(dataset)

classes = ["Adhesive tape",
	   "Axe",
	   "Boot",
	   "Clothing",
	   "Computer keyboard",
	   "Computer mouse",
	   "Corded phone",
	   "Dagger",
	   "Door handle",
	   "Drill (Tool)"]
splits = ["train","validation"]

nb_train = 200
nb_val = nb_train*20//100

for split in splits:
	if split == "train":
		samples = nb_train
	else:
		samples = nb_val
	for classe in classes:
		dataset = foz.load_zoo_dataset(
		    "open-images-v6",
		    split=split,
		    label_types=["segmentations"],
		    classes = [classe],
		    max_samples=samples,
		    seed=51,
		    shuffle=True,
		)

		# session = fo.launch_app(dataset)

		# session.wait()
		data = os.path.expanduser('~/fiftyone/open-images-v6/')
		data = os.path.join(data, split, 'data')
		data_class = os.path.join(data, classe)
		os.mkdir(data_class)
		for img in next(os.walk(data))[-1]:
			if img.endswith('.jpg'):
				os.rename(os.path.join(data, img), os.path.join(data_class, img))
