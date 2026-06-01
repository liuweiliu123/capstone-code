import argparse

from xgboost import XGBClassifier

from common import default_csv_path, prepare_one_hot_data, print_metrics, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost for toddler autism trait classification.")
    parser.add_argument("--csv-path", default=str(default_csv_path()), help="Path to the toddler autism CSV dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = prepare_one_hot_data(args.csv_path)
    x_train, x_test, y_train, y_test = split_data(x, y)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_metrics("XGBoost", y_test, y_pred, len(x_train), len(x_test))


if __name__ == "__main__":
    main()

