import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyodide.http import open_url 

data = pd.read_csv(open_url("https://raw.githubusercontent.com/Itsa11/destination-city/main/Destiny.csv"))

data['Rating']=data['Rating'].apply(str)

selected_features = ['City']

combined_features = data['City']
  
vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)
  
similarity = cosine_similarity(feature_vectors)
  
name = input()
  
list_of_all_Places = data['City'].tolist()
  
def levenshtein_distance(s, t):
      m, n = len(s), len(t)
      d = [[0] * (n + 1) for _ in range(m + 1)]
      for i in range(1, m + 1):
          d[i][0] = i
      for j in range(1, n + 1):
          d[0][j] = j
      for j in range(1, n + 1):
          for i in range(1, m + 1):
              if s[i - 1] == t[j - 1]:
                  cost = 0
              else:
                  cost = 1
              d[i][j] = min(d[i - 1][j] + 1,  
                            d[i][j - 1] + 1,  
                            d[i - 1][j - 1] + cost)  
      return d[m][n]
  
def get_close_matches(query, choices, n=3):
      distances = [(levenshtein_distance(query, word), word) for word in choices]
      matches = [word for distance, word in sorted(distances) if distance < len(query)]
      return matches[:n]
  
find_close_match = get_close_matches(name, list_of_all_Places,n=3)
  
close_match = find_close_match[0]
 
index_of_the_Des=data.index[data['City']==close_match]
index_of_the_Des=index_of_the_Des[0]

similarity_score = list(enumerate(similarity[index_of_the_Des]))

Sugg = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
  
print('  ','You would also Like : \n')
i = 1
destiny = input()
for D in Sugg:
    index = D[0]
    Places_from_index = data[data.index==index]['Place'].values[0]
    City_from_index = data[data.index==index]['City'].values[0]
    Ratings_from_index = data[data.index==index]['Rating'].values[0]
    if (i< 14493 and Ratings_from_index == destiny and City_from_index== name):
      print( '   ' ,i, ' . ' , Places_from_index , ' - ' , City_from_index,' - ',Ratings_from_index)
      i+=1 
