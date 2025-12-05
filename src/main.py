from data.data_preprocessing import get_preprocessed_data
from features.feature_extractor import FeatureExtractor
from models.model_training import Model

def main():
    data = get_preprocessed_data() # TODO 
    print(data.head())  # TODO remove this line

    feature_extractor = FeatureExtractor()
    data = feature_extractor.get_features(data)


    # model = Model() # TODO
    # model.train_model(data) # TODO

if __name__ == "__main__":
    main()