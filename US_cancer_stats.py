import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

FILE_PATH = "./DATA/US_cancer_stats.xlsx"

DATA = pd.read_excel(FILE_PATH)

# Plot the trend lines for total cancer Dx and the CUP percentage over time
# Extract the data from the source
years = DATA['year']
total = DATA['total']
cup = DATA['CUP']
percentage = DATA['percentage']

# Create Interpolation functions to smooth the trend lines
years_smooth = np.linspace(years.min(), years.max(), 365)

f_total = interp1d(years, total, kind='cubic')
total_smooth = f_total(years_smooth)

f_percentage = interp1d(years, percentage, kind='cubic')
percentage_smooth = f_percentage(years_smooth)

# Set graphical style of the plot
sns.set_style("dark")

# Create the main figure
fig, ax1 = plt.subplots(figsize=(17, 6))

# Plot total cancer Dx trend line
sns.lineplot(x=years_smooth, y=total_smooth, ax=ax1, color='blue', label='Total cancer Dx', linewidth=2)
ax1.set_xlabel('Year')
ax1.set_ylabel('Absolute number of cancer Dx', color='blue')
ax1.yaxis.labelpad = 10

# Second y-axis for percentage of CUP
ax2 = ax1.twinx()
sns.lineplot(x=years_smooth, y=percentage_smooth, ax=ax2, color='red', label='CUP percentage', linewidth=2)
ax2.set_ylabel('Percentage of CUP Dx in overall cancer Dx', color='red')
ax2.yaxis.labelpad = 10

# Third y-axis for the bar plot (absolute number of CUP)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis to the right
ax3.bar(years, cup, color='gray', alpha=0.5, label='Total CUP Dx')
ax3.set_ylabel('Absolute number of CUP Dx', color='gray')
ax2.yaxis.labelpad = 10
ax3.set_ylim(0, 100000)

# Adjust position of the plot in the middle of the figure
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)


plt.title('Trends of Total Cancer Diagnoses and the Percentage of CUP Over Time in USA')
plt.savefig('./US_cancer_stats_plt.png')
plt.show()
