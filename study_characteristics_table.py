import pandas as pd

source_doc = pd.read_excel("./DATA/study_characteristics_file/source_doc.xlsx")
processed_doc = pd.read_excel("./DATA/study_characteristics_file/processed_doc.xlsx")

included_papers = processed_doc[(processed_doc["Not included"].isna()) | (processed_doc["Not included"] == 2)]

extracted_info = []
for n in range(len(included_papers)):
        info = {"rayyan_id": included_papers.iloc[n]["rayyan_id"], "processed_doc_id": included_papers.iloc[n]["ID"],
                "era": included_papers.iloc[n]["Era"], "year": included_papers.iloc[n]["Year"],
                "quality_tool": included_papers.iloc[n]["Quality Assessment Tool"]}
        extracted_info.append(info)


def author_list_trimmer(author_list: str):
    a = author_list.find(" and")
    return f"{author_list[:a]} et al."


for n in extracted_info:
    rayyan_id = n["rayyan_id"]
    paper_row = source_doc[source_doc["key"] == rayyan_id]
    n["title"] = paper_row["title"].values[0]
    n["journal"] = paper_row["journal"].values[0]
    n["authors_complete"] = paper_row["authors"].values[0]
    n["author"] = author_list_trimmer(paper_row["authors"].values[0])

output_df = pd.DataFrame(extracted_info)

output_df.to_excel("Study Characteristics.xlsx", index=False)
