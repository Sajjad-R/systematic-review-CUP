import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import os

FILE_PATH = "./DATA/final_study_characteristics_table.xlsx"

DATA = pd.read_excel(FILE_PATH)

# Set the figure theme
sns.set_style("white")
font = {'fontname': 'Arial', 'fontsize': 11, 'fontweight': "bold"}

# Create the main figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(17, 12))

# Adjust the spacing between subplots
fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.30)

# Plot A (distribution of studies across the eras) #####################################################################
# Extract data
eras = DATA["Era"]
count_per_era = eras.value_counts()
print(count_per_era)
count_per_era.name = ""  # so that the label "count" does not appear in the plot

# Define a custom autopct function to round the percentages instead of trimming to one decimal point


def autopct_format(pct, all_values):
    total = sum(all_values)
    value = int(round(pct * total / 100))
    return f'{value}%'


# Convert the value counts to percentages
percentages = count_per_era / count_per_era.sum() * 100

# make plot A - Pie chart
colors = ['#523675', '#D15472', '#FFA600']
# count_per_era.plot.pie(ax=axs[0, 0], colors=colors, autopct='%1.1f%%', textprops=font, pctdistance=1.15, labels=None,
#                        startangle=35, counterclock=False)  #one decimal point version
count_per_era.plot.pie(ax=axs[0, 0], colors=colors, autopct=lambda pct: autopct_format(pct, percentages), textprops=font, pctdistance=1.15, labels=None,
                       startangle=35, counterclock=False)
axs[0, 0].axis('equal')

axs[0, 0].set_title("A.", loc="left", pad=5, **font)
axs[0, 0].legend(["2D imaging", "3D imaging", "Emerging technologies"], prop={'family': 'arial', 'size': 11, 'style': 'normal'}, loc='lower left',
                 bbox_to_anchor=(-0.08, -0.1), borderpad=1, frameon=True)

# Plot B (number of studies per modality) ##############################################################################
# Extract data
modality_paper = {"PET": {"Total": 0, "18F-FDG PET": 0, "18F-FDG PET/CT": 0, "18F-FDG PET/MRI": 0, "Non-FDG PET": 0},
                  "MRI": {"Breast MRI": 0, "Other MRI": 0}, "CT": 0, "Others": 0}

for n in range(len(DATA)):
    record = DATA.iloc[n]
    record_modality_str = record["Modality"]

    if "*" in record_modality_str:
        record_modality_list = record_modality_str.split(" * ")
    else:
        record_modality_list = [record_modality_str]

    add = {"PET": {"Total": 0, "18F-FDG PET": 0, "18F-FDG PET/CT": 0, "18F-FDG PET/MRI": 0, "Non-FDG PET": 0}, "MRI":
        {"Breast MRI": 0, "Other MRI": 0}, "CT": 0, "Others": 0}

    for m in record_modality_list:
        if m == "CT" or m == "CT Radiomics":
            add["CT"] = 1
        elif m == "MRI" or m == "MRI Radiomics" or m == "WB DWI MRI" or m == "MP MRI" or m == "DWI MRI":
            add["MRI"]["Other MRI"] = 1
        elif m == "Breast MRI":
            add["MRI"]["Breast MRI"] = 1
        elif m == '68-Ga DOTANOC PET/CT' or m == '68-Ga DOTATOC PET/CT' or m == '68-Ga DOTATATE PET/CT' or \
                m == '68-Ga FAPI PET/CT' or m == "18F-DOPA PET/CT":
            add["PET"]["Non-FDG PET"] = 1
            add["PET"]["Total"] = 1
        elif m == "18F-FDG PET/CT" or m == '18F-FDG PET/CT Radiomics' or m == "18F-FDG PET or PET/CT":
            add["PET"]["18F-FDG PET/CT"] = 1
            add["PET"]["Total"] = 1
        elif m == '18F-FDG PET/MRI':
            add["PET"]["18F-FDG PET/MRI"] = 1
            add["PET"]["Total"] = 1
        elif m == "18F-FDG PET":
            add["PET"]["18F-FDG PET"] = 1
            add["PET"]["Total"] = 1
        else:
            add["Others"] = 1

    for c in modality_paper.keys():
        if isinstance(modality_paper[c], dict):
            for sc in modality_paper[c].keys():
                modality_paper[c][sc] += add[c][sc]
        elif isinstance(modality_paper[c], int):
            modality_paper[c] += add[c]
        else:
            raise ValueError

modality_paper["MRI"]["Total"] = modality_paper["MRI"]["Breast MRI"] + modality_paper["MRI"]["Other MRI"]

# Correction for study number 53 (Multiple = 18F-FDG PET + CT + MRI)
modality_paper["PET"]["Total"] += 1
modality_paper["PET"]["18F-FDG PET"] += 1
modality_paper["CT"] += 1
modality_paper["MRI"]["Other MRI"] += 1
modality_paper["MRI"]["Total"] += 1
modality_paper["Others"] -= 1

print("Modality_paper: ", modality_paper)

# Data preprocessing for stacked bar plot
processed_data = modality_paper
processed_data['PET']["18F-FDG PET only"] = processed_data['PET']["Total"] - (processed_data['PET']["18F-FDG PET/CT"] +
                                        processed_data['PET']["18F-FDG PET/MRI"] + processed_data['PET']["Non-FDG PET"])
del processed_data['PET']['Total']
del processed_data['PET']['18F-FDG PET']
del processed_data['MRI']['Total']

print('Plotted_Modality_paper: ', processed_data)

# ################

categories = list(processed_data.keys())
has_subcategories = {cat: isinstance(processed_data[cat], dict) for cat in categories}

# Collect all subcategories
subcategories = set()
for cat, subcat in has_subcategories.items():
    if subcat:
        subcategories.update(processed_data[cat].keys())
subcategories = sorted(subcategories)

# Initialize matrices to hold values for subcategories and single values
values_matrix = np.zeros((len(subcategories), len(categories)))
single_values = np.zeros(len(categories))

# Fill the matrices with data
for i, category in enumerate(categories):
    if has_subcategories[category]:
        for j, subcategory in enumerate(subcategories):
            values_matrix[j, i] = processed_data[category].get(subcategory, 0)
    else:
        single_values[i] = processed_data[category]

bar_positions = np.arange(len(categories))

# Define colors for each stack in the bars
# print(subcategories)  # To assign the correct color in the next line
colors = ['#7d3244', '#a7435b', '#d15472', '#523675', '#de879c', '#806FAF']

# Define legends for each stacked bar based on the colors above
# PET bar
pet_gray_patch = mpatches.Patch(color='#7d3244', label='18F-FDG PET only')
pet_red_patch = mpatches.Patch(color='#a7435b', label='18F-FDG PET/CT')
pet_yellow_patch = mpatches.Patch(color='#D15472', label='18F-FDG PET/MRI')
pet_green_patch = mpatches.Patch(color='#de879c', label='Non-FDG PET')

# MRI bar
mri_gray_patch = mpatches.Patch(color='#523675', label='Breast MRI')
mri_red_patch = mpatches.Patch(color='#806FAF', label='Other MRI')

# make plot B - Stacked bar plot
bottom_values = np.zeros(len(categories))
for i, subcategory in enumerate(subcategories):
    bars = axs[0, 1].bar(bar_positions, values_matrix[i], bottom=bottom_values, color=colors[i], label=subcategory)
    bottom_values += values_matrix[i]

# Plot the single value bars
axs[0, 1].bar(bar_positions, single_values, bottom=bottom_values, color=['#41688E', '#205F64'], label='Single Value')

# Add legend for stacked bars
legend_pet = axs[0, 1].legend(handles=[pet_gray_patch, pet_red_patch, pet_yellow_patch, pet_green_patch, mri_gray_patch, mri_red_patch],
                              loc='upper right', bbox_to_anchor=(1, 1), prop={'family': 'arial', 'size': 11,
                                                                              'style': 'normal'})
# legend_mri = axs[0, 1].legend(handles=[mri_gray_patch, mri_red_patch], title="MRI", loc='upper right',
#                               bbox_to_anchor=(1, 0.7), prop={'family': 'arial', 'size': 11, 'style': 'normal'})

# Add both legends to the plot
axs[0, 1].add_artist(legend_pet)
# axs[0, 1].add_artist(legend_mri)

# Add and modify labels
axs[0, 1].set_title("B.", loc="left", pad=5, **font)
axs[0, 1].set_xlabel('Modality', fontdict=font)
axs[0, 1].set_ylabel('Number of studies', fontdict=font)
axs[0, 1].set_xticks(bar_positions)
axs[0, 1].set_xticklabels(categories, fontdict=font)
axs[0, 1].set_ylim(0, 140)
axs[0, 1].tick_params(axis='y', labelsize=11, labelcolor='black', labelrotation=45)
axs[0, 1].xaxis.labelpad = 10
axs[0, 1].yaxis.labelpad = 10

# ################

# Plot C (study population distribution) ###############################################################################
# Extract data
populations = DATA["Study Population"]

print("Study populations (Median): ", populations.median())
IQR = populations.quantile(0.75) - populations.quantile(0.25)
print("Study populations (IQR): ", IQR)
print("Study populations (Min): ", populations.min())
print("Study populations (Max): ", populations.max())

# make plot C - Histogram
sns.histplot(data=populations, kde=True, binwidth=9, fill=True, color="#D15472", ax=axs[1, 0])

# Add and modify labels
axs[1, 0].set_xlabel('Sample size', fontdict=font)
axs[1, 0].set_ylabel('Number of studies', fontdict=font)
axs[1, 0].xaxis.labelpad = 10
axs[1, 0].yaxis.labelpad = 10
axs[1, 0].tick_params(axis='y', labelsize=11, labelcolor='black', labelrotation=45)
axs[1, 0].tick_params(axis='x', labelsize=11, labelcolor='black', labelrotation=0)
axs[1, 0].set_title("C.", loc="left", pad=5, **font)

# Plot D (prospective and retrospective studies) #######################################################################
# Extract data
isprospective_df = DATA["Study type (retrospective vs prospective)"]
num_prospective = 0
num_retrospective = 0

for n in isprospective_df:
    if pd.isnull(n):
        num_retrospective += 1
    else:
        num_prospective += 1

# Temp:
num_prospective = 51
num_retrospective = 89
# make plot D - Donut chart
labels = ['Prospective', 'Retrospective']
sizes = [num_prospective, num_retrospective]
colors = ['#a69ac7', '#806FAF']

axs[1, 1].pie(sizes, colors=colors, autopct='%1.1f%%', startangle=120, wedgeprops={'width': 0.5}, textprops=font,
              pctdistance=0.75)
axs[1, 1].axis('equal')

axs[1, 1].set_title("D.", loc="left", pad=5, **font)
axs[1, 1].legend(labels, prop={'family': 'arial', 'size': 11, 'style': 'normal'}, loc='lower right',
                 bbox_to_anchor=(1.1, 0), borderpad=1, frameon=True)
total_num_publications = len(DATA)
axs[1, 1].text(0, 0, f'Total publications\n\n{total_num_publications}', horizontalalignment='center', verticalalignment='center', fontdict=font)

# Save the figure


def save_figure(f_format: str or list, dpi: int or list, path: str, f_name: str):
    """f_format without the dot, path to the folder you want to save the figures without the slash at the end,
    f_name without the extension"""
    def sub_function():
        if isinstance(dpi, list):
            for d in dpi:
                plt.savefig(fname=f"{path}/{f_name}_{d}.{f}", dpi=d)
        else:
            d = dpi
            plt.savefig(fname=f"{path}/{f_name}_{d}.{f}", dpi=d)

    if isinstance(f_format, list):
        for f in f_format:
            sub_function()
    else:
        f = f_format
        sub_function()


# make a new directory to save the figures
destination_folder = './study_characteristics_graphs'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


save_figure(f_format=["png", "svg", "pdf", "eps"], dpi=[300, 600, 900, 1200],
            path=destination_folder, f_name="Figure_2")

plt.show()
