import numpy
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_roc_curve(FPRs: numpy.array, TPRs: numpy.array):
    plt.figure()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(FPRs, TPRs)
    plt.grid()
    plt.show()


def plot_bayes_error(effPriorLogOdds: numpy.array, DCFs: numpy.array, minDCFs: numpy.array):
    plt.figure()
    plt.plot(effPriorLogOdds, DCFs, label='DCF', color='r')
    plt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='b')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()


def plot_scatter(DTR: numpy.array, LTR: numpy.array):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    with PdfPages('target/scatters.pdf') as pdf:
        for i in range(DTR.shape[0]):
            for j in range(i, DTR.shape[0]):
                if i == j:
                    continue
                fig = plt.figure()
                plt.xlabel(f"Feature: {i}")
                plt.ylabel(f"Feature: {j}")
                plt.scatter(f0[i, :], f0[j, :], label='Spoofed')
                plt.scatter(f1[i, :], f1[j, :], label='Authentic')

                plt.legend()
                pdf.savefig(fig)
                plt.close(fig)


def plot_hists(DTR: numpy.array, LTR: numpy.array):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    with PdfPages('target/hists.pdf') as pdf:
        for i in range(DTR.shape[0]):
            fig = plt.figure()
            plt.xlabel(f"Feature: {i}")
            plt.ylabel("Density")
            plt.hist(f0[i, :], density=True, bins=20, label='Spoofed', alpha=0.4)
            plt.hist(f1[i, :], density=True, bins=20, label='Authentic', alpha=0.4)

            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)


def __pair_plot_hist(DTR: numpy.array, LTR: numpy.array, axis, i, j):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    axis[i, j].hist(f0[0, :], density=True, bins=20, label='Spoofed', alpha=0.4)
    axis[i, j].hist(f1[0, :], density=True, bins=20, label='Spoofed', alpha=0.4)


def __pair_plot_scatter(DTR: numpy.array, LTR: numpy.array, axis, i, j):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]

    axis[i, j].scatter(f0[0, :], f0[1, :], label='Spoofed')
    axis[i, j].scatter(f1[0, :], f1[1, :], label='Authentic')


def pair_plot(DTR: numpy.array, LTR: numpy.array):
    feature_count = DTR.shape[0]

    fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count, squeeze=False)
    fig.set_size_inches(feature_count * 4, feature_count * 4)

    # Iterate through features to plot pairwise.
    for i in range(0, feature_count):
        for j in range(0, feature_count):
            if i == j:
                __pair_plot_hist(DTR[i:i + 1], LTR, axis, i, j)
            else:
                new_data = numpy.vstack([DTR[i:i+1], DTR[j:j+1]])
                __pair_plot_scatter(new_data, LTR, axis, i, j)

    plt.show()


def plot_correlation_matrix(M: numpy.array, title: str):
    plt.imshow(M)
    plt.title(title)
    plt.colorbar()
    plt.show()


def plot_simple_plot(X: numpy.array, Y: numpy.array, x_label: str, y_label: str, color: str, title: str, x_scale: str):
    fig = plt.figure()
    plot_simple_plot_no_show(fig, X, Y, x_label, y_label, color, title, title, x_scale)
    plt.show()


def plot_simple_plot_no_show(fig, X: numpy.array, Y: numpy.array, x_label: str, y_label: str,
                             color: str, label: str, title: str, x_scale: str):
    plt.plot(X, Y, color=color, marker='.', linewidth=1, label=label)
    plt.xscale(x_scale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="upper right")

    # Major ticks every 20, minor ticks every 5
    if x_scale != "log":
        ax = fig.add_subplot(1, 1, 1)
        major_ticks = numpy.arange(0, 11, 1)
        minor_ticks = numpy.arange(0, 11, 0.5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        major_ticks = numpy.arange(0, 1.1, 0.1)
        minor_ticks = numpy.arange(0, 1.1, 0.05)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
    else:
        plt.grid(which='minor', alpha=0.2)
        plt.grid(which='major', alpha=0.5)
    return fig


def plot_gmm_g_grid_search(
        minDCFs: numpy.array,
        max_g_c1_vec: numpy.array,
        sigma_type: str,
        PCA_dimensions: int):
    x = numpy.arange(len(max_g_c1_vec))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = -1.5

    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in minDCFs.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=f"Non-target g={attribute}",
                       color=plt.get_cmap("tab20")(attribute // 2))
        ax.bar_label(rects, padding=12, rotation='vertical')
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xlabel('Target class "g" components')
    ax.set_title(f'{sigma_type}\nPCA = {PCA_dimensions}')
    ax.set_xticks(x + width, max_g_c1_vec)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1)

    plt.show()


def show_plot():
    plt.show()


def new_figure():
    return plt.figure()
