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

    fig, axis = plt.subplots(nrows=feature_count, ncols=feature_count)
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
