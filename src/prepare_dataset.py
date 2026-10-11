from astropy.io import fits
import pandas as pd
import os

#Paths
PATH_FITS             = "D:/Downloads_3/GZ_2/fits/gz2_hart16.fits"
PATH_FILENAME_MAPPING = "D:/Downloads_3/GZ_2/imgs/gz2_filename_mapping.csv"
PATH_IMAGES           = "D:/Downloads_3/GZ_2/imgs/images"
PATH_OUTPUT           = "datasets/dataset.csv"

#Reading fits file and preparing fits dataframe
hdul      = fits.open(PATH_FITS)
fits_data = hdul[1].data
df_fits   = pd.DataFrame(fits_data)
hdul.close()
print("Num. of rows in FITS file:", len(df_fits))

print("\n")

#Reading csv containing the names needed for mapping and preparing the dataframe for mapping
df_mapping = pd.read_csv(PATH_FILENAME_MAPPING)
print("Num. of rows in MAPPING file:", len(df_mapping))

#Extracting the first letter of the gz2 class
def map_class(gz2_class):
    if pd.isna(gz2_class): #if the class is empty, return none
        return None
    gz2_class = str(gz2_class)
    if gz2_class.startswith("A"):
        return "A"
    elif gz2_class.startswith("E"):
        return "E"
    elif gz2_class.startswith("S"):
        return "S"
    else:
        return "I"

df_fits["class4"] = df_fits["gz2_class"].apply(map_class)

print("\n")

#Merging the fits and csv
df_merged = pd.merge(
    df_fits,
    df_mapping,
    left_on="dr7objid",
    right_on="objid",
    how="inner"
)

print("After merging:", len(df_merged))

print("\n")

#Making of the image paths
df_merged["image_path"] = df_merged["asset_id"].apply(
    lambda x: os.path.join(
        PATH_IMAGES,
        f"{x}.jpg"
    )
)

#Searching wether the pictures exist and saving the names of the ones that do
print("Loading image filenames from disk.")
before_filter = len(df_merged)
existing_images = {
    os.path.splitext(filename)[0]
    for filename in os.listdir(PATH_IMAGES)
    if filename.endswith(".jpg")
}
df_merged = df_merged[
    df_merged["asset_id"].astype(str).isin(existing_images)
]
after_filter = len(df_merged)
print("Num. of images found:", after_filter)
print("Missing images:", before_filter - after_filter)

#Saving only the relevant classes
df_dataset = df_merged[
    [
        "dr7objid",
        "asset_id",
        "sample_x",
        "gz2_class",
        "class4",
        "image_path"
    ]
]

#Statistics print
print("\n")
print("Distribution amongst classes:")
print(
    df_dataset["class4"].value_counts()
)
print("\nDistribution amongst classes in %:")
print(
    df_dataset["class4"].value_counts(normalize=True) * 100
)
print("\nData example:")
print(
    df_dataset.head()
)

print("\n")

#Saving of the new dataset
df_dataset.to_csv(
    PATH_OUTPUT,
    index=False
)
print("\nSaved at:")
print(PATH_OUTPUT)

print("\n")

#Creating the binary E/S dataset
print("\nCreating E/S dataset.")
df_es = df_dataset[
    df_dataset["class4"].isin(["E", "S"])
].copy()
#Saving the relevant columns
df_es = df_es[
    [
        "image_path",
        "class4"
    ]
]
#Changing the name of the class: class4 -> label
df_es = df_es.rename(
    columns={
        "class4": "label"
    }
)
PATH_OUTPUT_ES = "datasets/dataset_ES.csv"
df_es.to_csv(
    PATH_OUTPUT_ES,
    index=False
)
print("\nE/S dataset saved at:")
print(PATH_OUTPUT_ES)