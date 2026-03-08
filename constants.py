BASE_PATH = "datasets/Itunes-Amazon"
MARKERS_ATTRIBUTES = ["Song_Name", "Artist_Name", "Album_Name"]

# LSH parameters
num_tables = 15
num_planes = 6

PROMPT = "You are an expert in data integration and entity resolution. " \
        "Your task is to determine whether the following two records refer to the exact same real-world entity.\n" \
        "Record A: {Record A}\n" \
        "Record B: {Record B}\n" \
        "Question: Do these records match? Please answer strictly with 'Yes' or 'No'."