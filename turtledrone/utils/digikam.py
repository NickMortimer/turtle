import sqlite3
import pandas as pd
from pathlib import Path

# Update this to your actual digikam4.db path
db_path = Path("/media/mor582/ROV_DISK/surveys/turtles/digikam4.db")  # e.g. "/home/user/Pictures/digikam/digikam4.db"
csv_output = Path("/media/mor582/ROV_DISK/surveys/turtles/digikam.csv")

# Connect to SQLite
conn = sqlite3.connect(db_path)

# SQL query: image paths + tags
query = """
SELECT 
    Albums.relativePath || '/' || Images.name AS image_path,
    GROUP_CONCAT(Tags.name, ', ') AS labels
FROM 
    ImageTags
JOIN Images ON ImageTags.imageid = Images.id
JOIN Tags ON ImageTags.tagid = Tags.id
JOIN Albums ON Images.album = Albums.id
GROUP BY Images.id
ORDER BY image_path;
"""

# Load into a DataFrame
df = pd.read_sql_query(query, conn)
df_filtered = df[df["image_path"].str.contains("thumbs", case=False, na=False)]
# Save to CSV
df_filtered.to_csv(csv_output, index=False)

print(f"Exported {len(df_filtered)} rows to {csv_output}")