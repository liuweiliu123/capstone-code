import argparse

from catboost import CatBoostClassifier

from common import default_csv_path, prepare_catboost_data, print_metrics, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train CatBoost for toddler autism trait classification.")
    parser.add_argument("--csv-path", default=str(default_csv_path()), help="Path to the toddler autism CSV dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    x, y, cat_features = prepare_catboost_data(args.csv_path)
    x_train, x_test, y_train, y_test = split_data(x, y)

    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=0,
    )
    model.fit(x_train, y_train, cat_features=cat_features)
    y_pred = model.predict(x_test).astype(int).ravel()

    print_metrics("CatBoost", y_test, y_pred, len(x_train), len(x_test))


if __name__ == "__main__":
    main()

