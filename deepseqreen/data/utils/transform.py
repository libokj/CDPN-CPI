from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler


def scale_transform(features, scaler: type[StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler]):
    scaler = scaler()
    features = scaler.fit_transform(features)
    features = scaler.transform(features)
    return features
