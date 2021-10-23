import pandas as pd
import os
from packaging import version
import Owls as ow
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


def get_ogl_versions(df):
    """read ogl_version."""
    return set(df.index.get_level_values("ogl_version"))


def idx_query(df, idx, val):
    """Shortcut to select specific index."""
    return df[df.index.get_level_values(idx) == val]


def idx_not_query(df, idx, val):
    """Shortcut to filter specific index."""
    return df[df.index.get_level_values(idx) != val]


def sel_ogl_version(df, ogl_version):
    """Shortcut to select a specific ogl version."""
    return idx_query(df, "ogl_version", ogl_version)


def process_meta_data(s):
    """Given a csr file as string the function reads embeded metatdata dict."""
    if (not ("#" in s)):
        return None
    else:
        try:
            meta_data_str = s.replace("#", "").replace("\n", "")
            return eval(meta_data_str)
        except:
            pass


def read_logs(folder):
    """Reads the logs file in the folder."""
    def clean_hash(s):
        return s.replace("hash: ", "").replace("\n", "")

    log_handles = []
    with open(folder + "/logs") as log_handle:
        log_content = log_handle.read()
        marker = "=" * 80 + "\n"
        log_content = log_content.split(marker)

    logs = {}
    for log in log_content:
        log_ = log.split("\n")
        log_header = log_[0]
        log_body = "\n".join(log_[1:])
        logs.update({clean_hash(log_header): log_body})

    keys = {}

    keys_solver = {
        "Solving for {}".format(f): [
            "init_residual",
            "final_residual",
            "iterations",
        ]
        for f in ["p", "U"]
    }

    keys_linear_solver = {
        "linear solve " + f: ["linear_solve_p"]
        for f in ["p"]
    }
    keys.update(keys_linear_solver)

    for log_hash, log_content in logs.items():
        if not log_content:
            continue
        try:
            logs[log_hash] = ow.read_log_str(log_content, keys)
        except Exception as e:
            print(e)
            pass

    return logs


def walk_ogl_data_folder(folder):
    root, ogl_versions, _ = next(os.walk(folder))
    for ogl_version in ogl_versions:
        _, ginkgo_versions, _ = next(os.walk(root + "/" + ogl_version))
        for ginkgo_version in ginkgo_versions:
            yield ogl_version, ginkgo_version


def read_ogl_data_folder(folder,
                         filt=None,
                         min_version="0.0.0",
                         import_logs=False):
    """Reads all csv files and logs from the given folder.

    returns a concatenated dataframe
    """
    # TODO refactor this
    dfs = []
    failures = {"empty": [], "dtype": []}
    metadata = {}
    logs = {}
    for ogl_version, ginkgo_version in walk_ogl_data_folder(folder):
        root = (folder + "/" + ogl_version + "/" + ginkgo_version)
        _, folders, reports = next(os.walk(root))

        if import_logs:

            log_folders = [
                f for f in folders if ("Logs" in f and not filt(root))
            ]
            if log_folders:
                logs[root] = read_logs(root + "/" + log_folders[0])

        for r in reports:
            fn = root + "/" + r
            if filt:
                if filt(fn):
                    continue

            with open(fn) as csv_handle:
                # read metadata
                try:
                    content = csv_handle.readlines()
                except:
                    continue
                if len(content) < 2:
                    continue
                metadata_ = process_meta_data(content[1])
                if not metadata_:
                    continue
                if not "OBR_REPORT_VERSION" in metadata_.keys():
                    continue
                obr_version = version.parse(metadata_["OBR_REPORT_VERSION"])
                if obr_version < version.parse(min_version):
                    continue
                metadata[fn] = metadata_

            df = pd.read_csv(fn, comment="#")
            # check if reading was a success
            if len(df) == 0:
                failures["empty"].append(fn)
                continue
            if (df["run_time"].dtype == "O"):
                failures["dtype"].append(fn)
                continue
            # append all version data
            df["ogl_version"] = ogl_version
            df["ginkgo_version"] = ginkgo_version
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True), failures, metadata, logs


def compute_speedup(df, ref, drop_indices=None):
    """Compute and return the speedup compared to a reference."""
    from copy import deepcopy
    df = deepcopy(df)
    if drop_indices:
        for idx in drop_indices:
            df.index = df.index.droplevel(idx)

    reference = idx_query(df, ref[0], ref[1])
    reference.index = reference.index.droplevel([ref[0]])

    def dropped_divide(df):
        from copy import deepcopy
        df = deepcopy(df)
        df.index = df.index.droplevel(ref[0])
        return df

    res = df.groupby(
        level=ref[0]).apply(lambda x: reference / dropped_divide(x))
    return res


def read_gingko_matrix(fn):
    import scipy.sparse as sparse
    rows = []
    cols = []
    vals = []
    with open(fn) as fh:
        lines = fh.readlines()
        for i, line in enumerate(lines):
            if (i == 1):
                n_rows, n_cols, nnz = line.split()
            if (i > 1):
                row, col, val = line.split()
                rows.append(int(row))
                cols.append(int(col))
                vals.append(float(val))
    return sparse.coo_matrix((vals, (rows, cols)),
                             shape=(int(n_rows) + 1, int(n_cols) + 1))


def short_hostname(x):
    if "uc2" in x:
        return "uc2"
    if "nla" in x:
        return "nla"
    if "guyot" in x:
        return "guyot"
    if "s" in x:
        return "devcloud"


def import_results(path,
                   case,
                   filt=None,
                   min_version="0.0.0",
                   import_logs=False,
                   short_hostnames=False,
                   dimension=3):
    """Import and postprocess obr benchmark results."""
    df, failures, metadata, logs = read_ogl_data_folder(
        path + case, filt, min_version, import_logs)

    # to use pandas multiindices data and non-data columns need to be separated
    data_columns = [
        "log_id",
        "run_time",
        "setup_time",
        "number_iterations_p",
        "number_iterations_U",
        "init_linear_solve_p",
        "linear_solve_p",
        "init_linear_solve_U",
        "linear_solve_U",
    ]

    df["solver_p"] = df["solver_p"].transform(
        lambda x: x.replace("GKO", "").replace("P", ""))

    if short_hostnames:
        df["node"] = df["node"].transform(lambda x: short_hostname(x))

    indices = [c for c in df.columns if c not in data_columns]

    df["linear_solve_p"] = 0

    # Refactor
    for directory, logs_ in logs.items():
        for log_hash, ff in logs_.items():
            try:
                latest_time = max(ff["Time"])
                df.loc[df["log_id"] == log_hash, "linear_solve_p"] = ff.at(
                    "Time", latest_time)["linear_solve_p"].mean()

            except Exception as e:
                print(e)
                pass

    # calculate some further metrics
    df["linear_solve_p_per_iter"] = (df["linear_solve_p"] /
                                     df["number_iterations_p"])
    df["cells"] = df["resolution"]**dimension
    df["linear_solve_p_per_cell_and_iter"] = (df["linear_solve_p_per_iter"] /
                                              df["cells"])

    # reorder indices a bit
    indices[0], indices[1] = indices[1], indices[0]
    df = df.fillna(0)
    df.set_index(indices, inplace=True)

    # compute mean of all values grouped by all indices
    # this will compute mean when all indices are identical
    mean = df.groupby(level=indices).mean()

    return {
        "raw": df,
        "mean": mean,
        "failures": failures,
        "metadata": metadata,
        "logs": logs,
        "indices": indices
    }


def min_over_index(df, col, idx):
    indices = list(df.index.names)
    df = df[df[col] > 0]
    return (df.loc[df[col].groupby(
        level=indices).idxmin()].stack().unstack(idx).min(
            axis=1, skipna=True).unstack())


def min_time_multi(df, col="run_time"):
    """Get the minimum run_time (non zero) for multiprocess data."""
    return min_over_index(min_over_index(df, col, "omp_threads"), col,
                          "mpi_ranks")


def min_time(df, col="run_time"):
    """Get the minimum run_time for multiprocess data."""
    indices = list(df.index.names)
    filt_idx = [
        "solver_p", 'executor_p', 'preconditioner_p', 'resolution', 'node',
        'ogl_version', 'ginkgo_version'
    ]
    indices = [idx for idx in indices if idx not in filt_idx]
    for idx in indices:
        df = min_over_index(df, col, idx)
    return df
