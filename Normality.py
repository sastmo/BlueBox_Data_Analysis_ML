# https://chat.openai.com/share/5d129f95-6214-4260-97a5-73ab8782fd31
# https://www.statology.org/normality-test-python/

# Method 1: Create a Histogram
import math
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# make this example reproducible
np.random.seed(1)

# generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

# create histogram to visualize values in dataset
plt.hist(lognorm_dataset, edgecolor='black', bins=20)
plt.show()

# Method 2: Create a Q-Q plot

import math
import numpy as np
from scipy.stats import lognorm
import statsmodels.api as sm
import matplotlib.pyplot as plt

# make this example reproducible
np.random.seed(1)

# generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

# create Q-Q plot with 45-degree line added to plot
fig = sm.qqplot(lognorm_dataset, line='45')

plt.show()

# Method 3: Perform a Shapiro-Wilk Test
import math
import numpy as np
from scipy.stats import shapiro
from scipy.stats import lognorm

# make this example reproducible
np.random.seed(1)

# generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

# perform Shapiro-Wilk test for normality
result_sh = shapiro(lognorm_dataset)
print(result_sh)

# Method 4: Perform a Kolmogorov-Smirnov Test
import math
import numpy as np
from scipy.stats import kstest
from scipy.stats import lognorm

# make this example reproducible
np.random.seed(1)

# generate dataset that contains 1000 log-normal distributed values
lognorm_dataset = lognorm.rvs(s=.5, scale=math.exp(1), size=1000)

# perform Kolmogorov-Smirnov test for normality
result_kt = kstest(lognorm_dataset, 'norm')
print(result_kt)

