import glob
import pandas as pd
import json
import os
from packaging import version
import Owls as ow
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
    
def get_ogl_versions(df):
    return set(df.index.get_level_values("ogl_version"))

def sel_ogl_version(df, ogl_version):
    sel = df[df.index.get_level_values("ogl_version") == ogl_version]
    sel.index = sel.index.droplevel("ogl_version")
    return sel

def idx_query(df, idx, val):
    return df[df.index.get_level_values(idx) == val]

def idx_not_query(df, idx, val):
    return df[df.index.get_level_values(idx) != val]

def process_meta_data(s):
    if (not ("#" in s)):
        return None
    else:                   
        try:
            meta_data_str = s.replace("#","").replace("\n","")
            return eval(meta_data_str)
        except:
            pass
        
def read_logs(folder):
    def clean_hash(s):
        return s.replace("hash: ","").replace("\n","")
        
    log_handles = []
    with open(folder + "/logs" ) as log_handle:
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
        
    keys_solver =  {"Solving for {}".format(f): [
                                 "init_residual",
                                 "final_residual",
                                 "iterations",
                             ]  for f in ["p","U"]}
    
    #keys.update(keys_solver)
    
    keys_linear_solver = {"linear solve " + f: ["linear_solve_p"] for f in ["p"] }
    keys.update(keys_linear_solver)
    
    for log_hash, log_content in logs.items():
        if not log_content:
            continue
        try:
            logs[log_hash] = ow.read_log_str(log_content, keys)
        except Exception as e:
            print(e)
            pass
        
    #print(logs)
            
    return logs

def walk_ogl_data_folder(folder):
    root, ogl_versions, _ = next(os.walk(folder))
    for ogl_version in ogl_versions:
        _, ginkgo_versions, _ = next(os.walk(root + "/" + ogl_version))
        for ginkgo_version in ginkgo_versions:
                yield ogl_version, ginkgo_version

def read_ogl_data_folder(folder, filt=None, min_version="0.0.0", import_logs=False):
    """reads all csv files from the given folder
    
    returns a concatenated dataframe """
    from os import walk
    
    #TODO refactor this
    dfs = []
    failures = {"empty": [], "dtype":[]}
    metadata = {}
    logs = {}
    for ogl_version, ginkgo_version in walk_ogl_data_folder(folder):
        root = (folder 
                        + "/" + ogl_version 
                        + "/" + ginkgo_version)
        _, folders, reports = next(walk(root))
        
        
        if import_logs:

            log_folders = [f for f in folders if ("Logs" in f and not filt(root))]
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
                if len(content) <  2:
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

def compute_speedup(df):
    #df = deepcopy(df)
    #df.index = df.index.droplevel("no")
    reference = idx_query(df, "executor_p", "Serial")

    res = df.groupby(level="executor_p").apply(
        lambda x: reference/x
    )
    # FIXME for now we change the backend of mpi results to GKO to avoid dropping the data
    # the above introduced some extra index columns
    res.index = res.index.droplevel(0)
    #print(res.dropna())
    #res = res[res.index.get_level_values("backend") != "OF"] 
    return res.dropna()

def compute_speedup_vs_OF_same_solver(df):
    df = deepcopy(df)
    #df.index = df.index.droplevel("no")
    reference = idx_query(df, "executor_p", "Serial")

    res = df.groupby(level="executor_p").apply(
        lambda x: reference/x
    )
    # FIXME for now we change the backend of mpi results to GKO to avoid dropping the data
    # the above introduced some extra index columns
    res.index = res.index.droplevel(0)
    #print(res.dropna())
    #res = res[res.index.get_level_values("backend") != "OF"] 
    return res.dropna()

def compute_speedup_renumbered(df_unrenumbered, df_renumber):
    """ compute of preconditioned vs unpreconditioned () runs """
    res = df_renumber/df_unrenumbered
    return res.dropna()

def compute_speedup_precond(df, precond):
    """ compute of preconditioned vs unpreconditioned () runs """
    reference = idx_query(df, "preconditioner", "none")
    reference.index = reference.index.droplevel("preconditioner")
    res = df.groupby(level="preconditioner").apply(
        lambda x: reference/x
    ) 
    res.index = res.index.droplevel(0)
    return res.dropna()

def compute_speedup_solver(df, solver):
    """ compute of preconditioned vs unpreconditioned () runs """
    reference = idx_query(df, "solver_p", solver)
    reference.index = reference.index.droplevel("solver_p")
    res = df.groupby(level="solver_p").apply(
        lambda x: reference/x
    ) 
    res.index = res.index.droplevel(0)
    return res.dropna()

def iterate_versions(df):
    ogl_versions = get_ogl_versions(df)
    for ogl_version in ogl_versions:
        ogl_sel = idx_query(df, "ogl_version", ogl_version)
        gko_versions=set(ogl_sel.index.get_level_values("ginkgo_version"))
        for gko_version in gko_versions:
            gko_sel = idx_query(ogl_sel, "ginkgo_version", gko_version)
            yield gko_sel, ogl_version, gko_version

def iterate_df_l3(df, level):
    l1_key = level[0]
    l1_names = set(df.index.get_level_values(l1_key))

    for l1_name in (l1_names):
        sel_l1 = idx_query(df, l1_key, l1_name)

        l2_key = level[1]
        l2_names = set(sel_l1.index.get_level_values(l2_key))
        for l2_name in l2_names:
            sel_l2 = idx_query(sel_l1, l2_key, l2_name)

            l3_key = level[2]
            l3_names = set(sel_l2.index.get_level_values(l3_key))
            for l3_name in l3_names:
                sel_l3 = idx_query(sel_l2, l3_key, l3_name)
                yield sel_l3, l1_name, l2_name, l3_name
            
def iterate_df_l2(df, level):
    l1_key = level[0]
    l1_names = set(df.index.get_level_values(l1_key))

    for l1_name in (l1_names):
        sel_l1 = idx_query(df, l1_key, l1_name)

        l2_key = level[1]
        l2_names = set(sel_l1.index.get_level_values(l2_key))
        for l2_name in l2_names:
            sel_l2 = idx_query(sel_l1, l2_key, l2_name)
            yield sel_l2, l1_name, l2_name
            
def iterate_df(df, level):
    if len(level) == 2:
        return iterate_df_l2(df, level)
    else:
        return iterate_df_l3(df, level)


def plot_run_times(df, case, name, properties, keys,
                   y="run_time", y_label="run time", 
                   x_label="resolution", x="resolution",
                  log_y=True):


    for sel, ogl_version, gko_version in iterate_versions(df):
        l0_key = keys[0]
        l0_names = set(df.index.get_level_values(l0_key))
        
        
        fig, axes = plt.subplots(1, len(l0_names), 
                         figsize=(13*len(l0_names), 12),
                         sharex=True, sharey=True, 
                         gridspec_kw={'wspace':0},
                         subplot_kw={'frameon':False})
        
        if len(l0_names) == 1:
           axes = [axes]
        
        for i, l0_name in enumerate(l0_names):
            sel_l0 = idx_query(df, l0_key, l0_name)
            ax = axes[i]
            
            if log_y:
                ax.set_yscale('log')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label.replace("_"," "))
            ax.set_title(l0_name)
            #ax.set_xscale('log')
            # https://matplotlib.org/stable/gallery/color/named_colors.html

            legends=[]
            lines = []
            for sel_ in iterate_df(sel_l0, keys[1:]):
                sel = sel_[0]
                unstacks = sel_[1:]
                
                current = sel[y]
                for i, unstack in enumerate(unstacks):
                    current = current.unstack(keys[i+1])[unstack]
                
                remaining_indices = current.index.names
                # the selection should only contain a single index which is used
                # as x axis, thus all remaining indices are droped here. 
                # Otherwise after calling reset index everthing becomes a mess
                drop_indices = [drop_idx for drop_idx in remaining_indices if drop_idx != x]
                line = (current
                    .reset_index(level=drop_indices, drop=True)
                    .plot(x=x,
                        ax=ax,  
                        linestyle=properties["linestyle"](unstacks),
                        marker=properties["marker"](unstacks), 
                        color=properties["color"](unstacks),
                        lw=3,
                        ms=15,
                         )
                       )
                legends.append(properties["label"](unstacks))
                line.legend(legends)

                #lines.append(line)
                ax.grid(True, axis='y', which="both")
                ax.grid(True, axis="y", which="minor", alpha=0.5)
                ax.grid(True, axis='x', which="both")
                ax.grid(True, axis="x", which="minor", alpha=0.5)
                

        title = "{}/{}/{}/{}.png".format(case, ogl_version, gko_version, name)
        savefig("../OGL_DATA/" + title, bbox_inches='tight')

                    
    
def plot_into_axes(df, axes, ref, **kwargs):
    # use groupby to loop through the `Factory Zone`
    for (k,d), ax in zip(df.groupby(['solver_p']), axes):
        # plot the data into subplot
        (d[d.index.get_level_values(ref[0]) != ref[1]] # dont show reference executor
         .unstack("executor") # facet into executor
         #.unstack("solver_U")
         .reset_index()
         .plot(x="resolution", kind="bar", ax=ax, stacked=False, width=1.0))
        
        # set label to the `Factory Zone`
        ax.set_xlabel("resolution")
        ax.set_ylabel("speedup")
        if kwargs.get("ylim"):
            ax.set_ylim(kwargs["ylim"])
    
        # remove the extra legend in each subplot
        legend = ax.legend()
        #xticks = [t + 0.1 for t in ax.get_xticks()[::2]]
        #xticklabels = ax.get_xticklabels()
        #ax.set_xticks(xticks)
        #ax.set_xticklabels(xticklabels[::2], rotation=90)
        ax.set_title(k)
        handlers = ax.get_legend_handles_labels()
        ax.legend().remove()
        ax.grid(True, axis='y', which="both")
        ax.grid(True, axis="y", which="minor", alpha=0.35)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.minorticks_on()

    # reinstall the last legend
    ax.legend(*handlers)
    #return fig
    
    
def plot_ogl(df, supertitle=None, ref=("executor","reference"), **kwargs):
    """ """
    num_solver =  len(set(df.index.get_level_values("solver_p")))
    # setting up the subplots
    fig, axes = plt.subplots(1, num_solver, 
                         figsize=(7*num_solver,5),
                         sharex=True, sharey=True, 
                         gridspec_kw={'wspace':0},
                         subplot_kw={'frameon':False})
    if num_solver == 1:
        axes = [axes]
    #if supertitle:
    #    fig.suptitle(supertitle, fontsize=10)
    plot_into_axes(df, axes, ref, **kwargs)
    return fig

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
    return sparse.coo_matrix((vals, (rows, cols)), shape=(int(n_rows)+1, int(n_cols)+1 ))

def plot_and_save_all_versions(name, case, df, **kwargs):
    for selection, ogl_version, gko_version, eqn in iterate_versions(df):
        title = "{}/{}/{}/{}{}.png".format(case, ogl_version, gko_version, eqn, name)
        solver = plot_ogl(selection, title, **kwargs)
        savefig("../OGL_DATA/" + title, bbox_inches='tight')

def plot_and_save_speedup_all_versions(name, case, speedup, **kwargs):
    from matplotlib.pyplot import savefig
    for selection, ogl_version, gko_version, eqn in iterate_versions(speedup["run_time"]):
        title = "{}/{}/{}/{}{}.png".format(case, ogl_version, gko_version, eqn, name)
        solver = plot_ogl(selection, title, **kwargs)
        savefig("../OGL_DATA/" + title, bbox_inches='tight')

def plot_sparsity_pattern(mtx):
    import matplotlib.pylab as plt
    fig, axes = plt.subplots(1, 1, 
                         figsize=(12,12),
                         sharex=True, sharey=True, 
                         gridspec_kw={'wspace':0},
                         subplot_kw={'frameon':False})
    plt.spy(mtx, markersize=0.5, figure=fig)
    return fig

def short_hostname(x):
        if "uc2" in x:
            return "uc2"
        if "nla" in x:
            return "nla"
        if "guyot" in x:
            return "guyot"
        if "s" in x:
            return "devcloud"
        
def import_results(path,  case, filt=None, min_version="0.0.0", import_logs=False, short_hostnames=False, dimension=3):
    
    df, failures, metadata, logs = read_ogl_data_folder(path+case, filt, min_version, import_logs)
    
    # to use pandas multiindices data and non-data columns need to be separated
    data_columns = ["log_id", "run_time", "setup_time", 
                    "number_iterations_p", "number_iterations_U", 
                    "init_linear_solve_p", "linear_solve_p",
                    "init_linear_solve_U", "linear_solve_U",]
    
    df["solver_p"] = df["solver_p"].transform(lambda x: x.replace("GKO","").replace("P",""))
    
    if short_hostnames:
        df["node"] = df["node"].transform(lambda x:  short_hostname(x) )
    
    indices = [c for c in df.columns if c not in data_columns]
    
    df["linear_solve_p"] = 0
    
    # Refactor
    for directory, logs_ in logs.items():
        for log_hash, ff in logs_.items():
            try:
                latest_time = max(ff["Time"])
                df.loc[ df["log_id"] == log_hash, "linear_solve_p" ] = ff.at("Time", latest_time )["linear_solve_p"].mean()
            
            except Exception as e:
                print(e)
                pass
    
    
    
    # calculate some further metrics
    #df["total_run_time"] = df["run_time"] 
    #df["run_time"] = df["run_time"] - df["setup_time"]
    
    #df["total_linear_solve_p"] = df["linear_solve_p"] 
    #df["linear_solve_p"] = df["linear_solve_p"] - df["init_linear_solve_p"]
    #df["run_time_per_iteration_p"] = df["run_time"]/df["number_iterations_p"]
    
    #df["iterations_per_cell"] = df["number_iterations_p"]/(df["resolution"]**dimension)
    #df["run_time_per_cell"] = df["run_time"]/(df["resolution"]**dimension)
    #df["run_time_per_cell_and_iter"] = df["run_time"]/((df["resolution"]**dimension)* df["number_iterations_p"])
    #df["linear_solve_per_cell_and_iter"] = df["linear_solve_U"]/((df["resolution"]**dimension)* df["number_iterations_p"])


    
    
    # reorder indices a bit
    indices[0], indices[1] = indices[1], indices[0]
    df = df.fillna(0)
    df.set_index(indices, inplace=True)
    



    # compute mean of all values grouped by all indices
    # this will compute mean when all indices are identical
    mean = df.groupby(level=indices).mean()
    
    # TODO check constraints or find workarounds
    # ie if job did not finish the resolutions might not match
    # hence for executor and solver the resolutions should have same length
    
    #executor = set(df.index.get_level_values("executor"))
    #for e in executor:
    #    backends = set(df.loc[e].index.get_level_values("backend"))
    #    for b in backends:
    #       solver = set(df.loc[e].loc[b].index.get_level_values("solver_p"))
    #        for s in solver:
    #            print(e,b,s, mean.loc[e].loc[b].loc[s].index.get_level_values("resolution"))

        
    #print({e: mean.loc[e] for e in executor})
    
    # drop device for now to allow division across different devices
    #mean.index = mean.index.droplevel("device")
    
    #indices_wo_device = list(filter(lambda x: x != "processes" and x != "device", indices))
    return  {"raw":df,
             "mean": mean,
             "failures": failures,
             "metadata": metadata,
             "logs": logs,
             "indices": indices
            }

def min_over_index(df, col, idx):
    indices = list(df.index.names)
    df = df[df[col]>0]
    return (df.loc[
        df[col].groupby(
            level=indices)
        .idxmin()]
        .stack()
        .unstack(idx).min(axis=1, skipna=True)
        .unstack()                
        )

def min_time_multi(df, col="run_time"):
    """ get the minimum run_time (non zero) for multiprocess data"""
    return min_over_index(min_over_index(df, col, "omp_threads"), col, "mpi_ranks")

def min_time(df, col="run_time"):
    """ get the minimum run_time for multiprocess data"""
    indices = list(df.index.names)
    filt_idx = ["solver_p", 'executor_p',  'preconditioner_p',  'resolution',  'node',  'ogl_version',  'ginkgo_version']
    indices = [idx for idx in indices if idx not in filt_idx ]
    for idx in indices:
        df = min_over_index(df, col, idx)
    return df
