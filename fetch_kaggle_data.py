from mlglance.utils import (
    download_kaggle_dataset,
    submit_to_competition,
    download_kaggle_competition_data,
    download_kaggle_competition_data_file
)

download_kaggle_dataset(
    dataset_name="kmmohsin/amex-denoised-aggregated-features", savepath="data/"
)
download_kaggle_dataset(
    dataset_name="odins0n/amex-parquet", savepath="data/"
)

# submit_to_competition(
#     competition_name="amex-default-prediction",
#     submission_path="submission/submission.csv",
#     message="Submititng to competition",
# )

# download_kaggle_competition_data(
#     competition_name="amex-default-prediction", savepath="data/"
# )

download_kaggle_competition_data_file(
   competition_name="amex-default-prediction", savepath="data/amex-default-prediction/",filename="train_labels.csv"
)
