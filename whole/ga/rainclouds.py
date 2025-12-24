from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats

####################################
# warming up.. plotting a function #
####################################
# x = np.arange(-10, 10, step=1)
# x = np.linspace(-10, 10, num=20)
# y = x**2 + 1
# data = {
#     "x": x,
#     "y": y
# }
# df = pd.DataFrame(data)

# _, ax = plt.subplots()
# ax.scatter(df["x"], df["y"])
# plt.show()

##############################################
#classic central limit theorem visualisation #
##############################################
# experiments = 100
# samples = []
# for e in range(experiments):
#     sample = np.random.random(100)*20 #100 observations between 0 and 20
#     mean = sample.mean()
#     samples.append(mean)

# _, ax = plt.subplots()
# ax.hist(samples, bins=25)
# plt.show()

##############################
# RAINCLOUD PLOTS ☔️✨ ######
##############################

# def rainclouds(groups: list[list], assume_normal=True, save=False, filepath=None):
#     fig, ax = plt.subplots()

#     for i, s in enumerate(groups):

#         if assume_normal:
#             mean = s.mean()
#             std = s.std()
#             x = np.linspace(mean - 6*std, mean + 6*std, len(s))
#             y = stats.norm.pdf(x, mean, std)
#         else:
#             # add density function (both fill up and delimiting lines?)
#             kde = stats.gaussian_kde(s)
#             x = np.linspace(min(s)-1, max(s)+1, num=len(s))
#             y = kde(x)
        
#         y = y / y.max() * 0.4 

#         ax.fill_between(x, i, y + i, alpha=0.4, color="lightgrey") # space between base and top of cloud
#         ax.plot(x, y + i, color="darkgrey")

#         # add boxplot
#         ax.boxplot(s,
#                 vert=False,
#                 positions=[i],
#                 widths=0.08,
#                 patch_artist=True,
#                 showfliers=True) # show outliers!!!

#         # add raindrops (scatter)
#         jitter = np.random.normal(loc=0, scale=0.05, size=len(s))
#         ax.scatter(x=s,
#                 y=i - 0.2 + jitter,
#                 alpha=0.4,
#                 s=10)

#     if save == "True" and filepath is not None:
#         fig.savefig(filepath, dpi=300, bbox_inches="tight")
    
#     plt.show()

# a = np.random.normal(loc=3, scale=2, size=30)
# b = np.random.normal(loc=5, scale=1, size=30)
# samples = [a, b]

# rainclouds(samples, assume_normal=True)

summary = {
    "fashion": {"AE": None, "noAE": None}, # gonna be list of two lists 
    "mnist": {"AE": None, "noAE": None},
    "kmnist": {"AE": None, "noAE": None}
}

def rainclouds(data: dict[str, dict[str, np.array]], assume_normal=True):

    _, ax = plt.subplots(nrows=1, ncols=3, figsize =(10, len(data)*0.6))

    for col, (dataset, results) in enumerate(data.items()):
        for i, (condition, dv) in enumerate(results.items()):

            if assume_normal:
                mean = dv.mean()
                std = dv.std()
                x = np.linspace(mean - 6*std, mean + 6*std, len(dv))
                y = stats.norm.pdf(x, mean, std)
            else:
                # add density function (both fill up and delimiting lines?)
                kde = stats.gaussian_kde(dv)
                x = np.linspace(min(dv)-1, max(dv)+1, num=len(dv))
                y = kde(x)
            
            y = y / y.max() * 0.4 

            ax[col].fill_between(x, i, y + i, alpha=0.4, color="lightgrey") # space between base and top of cloud
            ax[col].plot(x, y + i, color="darkgrey")

            # add boxplot
            ax[col].boxplot(dv,
                    vert=False,
                    positions=[i],
                    widths=0.08,
                    patch_artist=True,
                    showfliers=True) # show outliers!!!

            # add raindrops (scatter)
            jitter = np.random.normal(loc=0, scale=0.05, size=len(dv))
            ax[col].scatter(x=dv,
                    y=i - 0.2 + jitter,
                    alpha=0.4,
                    s=10)

    plt.tight_layout()
    plt.show()