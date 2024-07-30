import os 
import pandas as pd 
import numpy as np 
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Image, Value

def load_qata_covid19_dataset(data_dir: str) -> Dataset:
	assert os.path.exists(data_dir)
	splits = ["train", "validation"]
	dataset_dict = {}
	for split in tqdm(splits, desc="collating the dataset"):
		# define dataset paths
		base_image_dir = os.path.join(data_dir, split, "images")
		base_label_dir = os.path.join(data_dir, split, "labels")
		img_data_csv = os.path.join(base_image_dir, "metadata.csv")
		label_data_csv = os.path.join(base_label_dir, "metadata.csv")

		# read csv file 
		image_df = pd.read_csv(img_data_csv)
		mask_df = pd.read_csv(label_data_csv)

		# create a list of image filepaths and label filepaths 
		img_list = [os.path.join(base_image_dir, fn) for fn in image_df["file_name"]]
		label_list = [os.path.join(base_label_dir, fn) for fn in mask_df["file_name"]]
		img_desc_list = image_df["text"].values

		# create data from dictionary and assign column datatype
		features = Features({
		    "image": Image(decode=True, mode="L", id=None),
		    "label": Image(decode=True, mode="L", id=None),
		    "text":  Value(dtype="string", id=None),
		})
		data_dict = {"image": img_list, "label": label_list, "text": img_desc_list}
		dataset = Dataset.from_dict(data_dict).cast(features)

		dataset_dict[split] = dataset
	
	# Combine into a DatasetDict
	dataset_dict = DatasetDict(dataset_dict)
	return dataset_dict 
