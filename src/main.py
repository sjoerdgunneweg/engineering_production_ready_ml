import click
from data.data_preprocessing import get_preprocessed_data
from features.feature_extractor import FeatureExtractor
from random_forest import RandomForestModel


@click.command()
@click.option("--preprocess", is_flag=True, help="Flag argument for running Preprocessing step.")
@click.option("--feat-eng", is_flag=True, help="Flag argument for running Feature Engineering step.")
@click.option("--training", is_flag=True, help="Flag argument for running Training step.")
def main(preprocess: bool, feat_eng: bool, training: bool):

    if preprocess:
        data = get_preprocessed_data() # TODO wrtie to parquet file
        print(data.head())

    if feat_eng:
        feature_extractor = FeatureExtractor() # TODO implement
        data = feature_extractor.get_features(data) # TODO read from parquet file 

    if training:
        data = data.toPandas()  # TODO read this out from parquet directly as pandas df, write the features to parquet before 
        random_forest_model = RandomForestModel() 
        random_forest_model.train_model(data)
if __name__ == "__main__":
    main()