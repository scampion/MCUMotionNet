def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.

    Function takes the root directory path and recursively parses
    all events data.
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.

    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.

    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.

    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.

    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator
    from tensorflow.python.framework import tensor_util

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        value_proto = tfevent.summary.value[0]
        if value_proto.HasField('tensor'):
            value = tensor_util.MakeNdarray(value_proto.tensor)
        else:
            value = value_proto.simple_value
        return dict(
            wall_time=tfevent.wall_time,
            name=value_proto.tag,
            step=tfevent.step,
            value=float(value),
        )

    columns_order = ['wall_time', 'name', 'step', 'value']

    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)

    return all_df.reset_index(drop=True)


if __name__ == "__main__":
    dir_path = "./logs_leo/fit_500/small_rnn_lstm_filters_4_dense_8"
    exp_name = "validation"
    df = convert_tb_data(f"{dir_path}/{exp_name}")

    print(df.head())
    #save to csv
    df.to_csv(f"results/all.csv", index=False)