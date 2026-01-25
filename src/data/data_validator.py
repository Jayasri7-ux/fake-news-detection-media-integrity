class DataValidator:
    def get_report(self, df):
        report = {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "missing_values": df.isnull().sum().sum(),
            "duplicate_rows": df.duplicated().sum(),
            "invalid_labels": df[~df["label"].isin([0, 1])].shape[0]
        }
        return report
