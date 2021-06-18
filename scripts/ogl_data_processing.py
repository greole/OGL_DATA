import glob
import pandas as pd

def read_ogl_data_folder(folder):
    """reads all csv files from the given folder
    
    returns a concatenated dataframe """
    from os import walk
    _, ogl_versions, _ = next(walk(folder))
    
    import pandas as pd
    #TODO refactor this
    dfs = []
    for ogl_version in ogl_versions:
        _, ginkgo_versions, _ = next(walk(folder + "/" + ogl_version))
        for ginkgo_version in ginkgo_versions:
            _, devices, _ = next(walk(folder + "/" + ogl_version + "/" + ginkgo_version))
            for device in devices:
                root = (folder 
                        + "/" + ogl_version 
                        + "/" + ginkgo_version 
                        + "/" + device)
                _, _, reports = next(walk(root))
                reports = [r for r in reports if ".csv" in r]
                for r in reports:
                    fn = root + "/" + r
                    df = pd.read_csv(fn, comment="#")
                    # append all version data
                    df["ogl_version"] = ogl_version
                    df["ginkgo_version"] = ginkgo_version
                    df["device"] = device
                    dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def read_ogl_data_folders(folders):
    """reads all csv files within the given list of folders
    
    returns a concatenated dataframe """
    dfs = []
    for f in folders:
        files = glob.glob(f +"/*.csv")
        for file in files:
            print("reading file ", f)
            df = pd.read_csv(file, comment="#")
            df["file"] = file
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_speedup(df_):
    # first create new column speedup and set it to unity
    df_["speedup"] = 1.
    
    # compute speedup
    for e in ["cuda", "omp"]:
        # create a series of speedup results
        s = (df_
             .loc["reference"]
             .reset_index()["run_time"]
             / df_
             .loc[e]
             .reset_index()["run_time"])
        
        # assign speedup to executor selection
        df_.loc[(e), "speedup"] = s.values
    
    return df_

def plot_ogl(df, supertitle=None):
    import matplotlib.pyplot as plt
    # setting up the subplots
    fig, axes = plt.subplots(1, 2, 
                         figsize=(6,4),
                         sharex=True, sharey=True, 
                         gridspec_kw={'wspace':0},
                         subplot_kw={'frameon':False})
    if supertitle:
        fig.suptitle(supertitle, fontsize=20)

    # use groupby to loop through the `Factory Zone`
    for (k,d), ax in zip(df.groupby(['solver_p']), axes):
        # plot the data into subplot
        (d[d.index.get_level_values('executor') != 'reference'] # dont show reference executor
         .unstack("executor") # facet into executor
         .reset_index()
         .plot(x="resolution", y="speedup", kind="bar", ax=ax, stacked=True, width=1.0))
        
        # set label to the `Factory Zone`
        ax.set_xlabel("resolution")
        ax.set_ylabel("speedup")

    
        # remove the extra legend in each subplot
        legend = ax.legend()
        xticks = [t + 0.1 for t in ax.get_xticks()[::2]]
        xticklabels = ax.get_xticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels[::2], rotation=90)
        ax.set_title(k)
        handlers = ax.get_legend_handles_labels()
        ax.legend().remove()
        ax.grid(True, axis='y')

    # reinstall the last legend
    ax.legend(*handlers)
    return fig