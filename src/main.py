from data.data_preprocessing import get_preprocessed_data
from features.feature_extractor import FeatureExtractor
from random_forest import RandomForestModel

def main(): # TODO make this a multi step process
    data = get_preprocessed_data() 
    print(data.head())

    feature_extractor = FeatureExtractor() # TODO implement
    data = feature_extractor.get_features(data)

    data = data.toPandas()  # TODO read this out from parquet directly as pandas df, write the features to parquet before 
    random_forest_model = RandomForestModel() 
    random_forest_model.train_model(data)

if __name__ == "__main__":
    main()