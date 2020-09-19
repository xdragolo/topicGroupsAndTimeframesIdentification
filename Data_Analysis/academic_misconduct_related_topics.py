import json
import pandas as pd

from closest_topics import heatmap

with open(r'C:\Users\annad\Documents\IGA\academia_exchage\Data\academic_misconduct_related.json','r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

n_most_frequent = ['personal-misconduct', 'research-misconduct', 'sexual-misconduct', 'abuse', 'acknowledgement', 'cheating',
            'discrimination', 'disreputable-publishers','plagiarism', 'self-plagiarism']

heatmap(df,n_most_frequent,'academic_misconduct_heatmap.png')

