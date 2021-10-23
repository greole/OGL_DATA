#!/usr/bin/env python3

import matplotlib.pylab as plt


def iterate_versions(df):
    ogl_versions = get_ogl_versions(df)
    for ogl_version in ogl_versions:
        ogl_sel = idx_query(df, "ogl_version", ogl_version)
        gko_versions = set(ogl_sel.index.get_level_values("ginkgo_version"))
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


def plot_run_times(df,
                   case,
                   name,
                   properties,
                   keys,
                   y="run_time",
                   y_label="run time",
                   x_label="resolution",
                   x="resolution",
                   log_y=True):

    for sel, ogl_version, gko_version in iterate_versions(df):
        l0_key = keys[0]
        l0_names = set(df.index.get_level_values(l0_key))

        fig, axes = plt.subplots(1,
                                 len(l0_names),
                                 figsize=(13 * len(l0_names), 12),
                                 sharex=True,
                                 sharey=True,
                                 gridspec_kw={'wspace': 0},
                                 subplot_kw={'frameon': False})

        if len(l0_names) == 1:
            axes = [axes]

        for i, l0_name in enumerate(l0_names):
            sel_l0 = idx_query(df, l0_key, l0_name)
            ax = axes[i]

            if log_y:
                ax.set_yscale('log')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label.replace("_", " "))
            ax.set_title(l0_name)
            #ax.set_xscale('log')
            # https://matplotlib.org/stable/gallery/color/named_colors.html

            legends = []
            lines = []
            for sel_ in iterate_df(sel_l0, keys[1:]):
                sel = sel_[0]
                unstacks = sel_[1:]

                current = sel[y]
                for i, unstack in enumerate(unstacks):
                    current = current.unstack(keys[i + 1])[unstack]

                remaining_indices = current.index.names
                # the selection should only contain a single index which is used
                # as x axis, thus all remaining indices are droped here.
                # Otherwise after calling reset index everthing becomes a mess
                drop_indices = [
                    drop_idx for drop_idx in remaining_indices if drop_idx != x
                ]
                line = (current.reset_index(
                    level=drop_indices, drop=True).plot(
                        x=x,
                        ax=ax,
                        linestyle=properties["linestyle"](unstacks),
                        marker=properties["marker"](unstacks),
                        color=properties["color"](unstacks),
                        lw=3,
                        ms=15,
                    ))
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
    for (k, d), ax in zip(df.groupby(['solver_p']), axes):
        # plot the data into subplot
        (d[d.index.get_level_values(ref[0]) !=
           ref[1]]  # dont show reference executor
         .unstack("executor")  # facet into executor
         #.unstack("solver_U")
         .reset_index().plot(x="resolution",
                             kind="bar",
                             ax=ax,
                             stacked=False,
                             width=1.0))

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


def plot_ogl(df, supertitle=None, ref=("executor", "reference"), **kwargs):
    """ """
    num_solver = len(set(df.index.get_level_values("solver_p")))
    # setting up the subplots
    fig, axes = plt.subplots(1,
                             num_solver,
                             figsize=(7 * num_solver, 5),
                             sharex=True,
                             sharey=True,
                             gridspec_kw={'wspace': 0},
                             subplot_kw={'frameon': False})
    if num_solver == 1:
        axes = [axes]
    #if supertitle:
    #    fig.suptitle(supertitle, fontsize=10)
    plot_into_axes(df, axes, ref, **kwargs)
    return fig


def plot_sparsity_pattern(mtx):
    fig, axes = plt.subplots(1,
                             1,
                             figsize=(12, 12),
                             sharex=True,
                             sharey=True,
                             gridspec_kw={'wspace': 0},
                             subplot_kw={'frameon': False})
    plt.spy(mtx, markersize=0.5, figure=fig)
    return fig


def plot_and_save_all_versions(name, case, df, **kwargs):
    for selection, ogl_version, gko_version, eqn in iterate_versions(df):
        title = "{}/{}/{}/{}{}.png".format(case, ogl_version, gko_version, eqn,
                                           name)
        solver = plot_ogl(selection, title, **kwargs)
        savefig("../OGL_DATA/" + title, bbox_inches='tight')


def plot_and_save_speedup_all_versions(name, case, speedup, **kwargs):
    from matplotlib.pyplot import savefig
    for selection, ogl_version, gko_version, eqn in iterate_versions(
            speedup["run_time"]):
        title = "{}/{}/{}/{}{}.png".format(case, ogl_version, gko_version, eqn,
                                           name)
        solver = plot_ogl(selection, title, **kwargs)
        savefig("../OGL_DATA/" + title, bbox_inches='tight')
