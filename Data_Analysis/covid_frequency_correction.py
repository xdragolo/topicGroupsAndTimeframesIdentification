import json
import pandas as pd

with open(r'C:\Users\annad\Documents\IGA\academia_exchage\Data\academia3.json') as f:
    data = json.load(f)
df = pd.DataFrame(data)

def containes_covid_reference(post_text):
    covid_reference = ['covid-19','coronavirus', 'corona-virus', 'covid', 'covid 19', 'virus', 'corona', 'covid19']
    for r in covid_reference:
        if r in post_text.lower():
            return True
    return False


covid_selected = df.loc[df.apply(lambda x :  containes_covid_reference(x['post_text']),axis=1)]
#
covid_selected.to_json('./covid_corrected.json', orient = 'records')
print(covid_selected['title'].count())
print(covid_selected.apply(lambda x :  'covid-19' in x['tags'],axis=1))



