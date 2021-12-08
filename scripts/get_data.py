#%%
from google_drive_downloader import GoogleDriveDownloader as gdd

#%%
gdd.download_file_from_google_drive(
    file_id="1_IKXrpMhcl5DA1PfHWo1cWpNzCUJmrRi", dest_path="/data/lst_files.tar", unzip=True
)
gdd.download_file_from_google_drive(
    file_id="1yLENoAWYa8BJV5Tomd0GCAYCwO0sESjc",
    dest_path="/data/formula_images_processed.tar",
    unzip=True,
)
