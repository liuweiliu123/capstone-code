import argparse

from sklearn.ensemble import RandomForestClassifier

from common import default_csv_path, prepare_one_hot_data, print_metrics, split_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest for toddler autism trait classification.")
    parser.add_argument("--csv-path", default=str(default_csv_path()), help="Path to the toddler autism CSV dataset.")
    return parser.parse_args()


def main():
    args = parse_args()
    x, y = prepare_one_hot_data(args.csv_path)
    x_train, x_test, y_train, y_test = split_data(x, y)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print_metrics("Random Forest", y_test, y_pred, len(x_train), len(x_test))


if __name__ == "__main__":
    main()

