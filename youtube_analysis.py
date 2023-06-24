#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


df=pd.read_csv(r"C:\prac\UScomments.csv",error_bad_lines=False)


# In[9]:


df


# In[8]:


df.head()


# In[10]:


df.isnull().sum()


# In[11]:


df.dropna(inplace=True)


# In[14]:


from textblob import TextBlob


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


polarity=[]
for comment in df['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[18]:


df['polarity']=polarity


# In[19]:


df


# In[21]:


filter1 = df['polarity']==1


# In[22]:


positive_comments=df[filter1]


# In[23]:


filter2=df['polarity']==-1


# In[24]:


negative_comments=df[filter2]


# In[25]:


from wordcloud import WordCloud , STOPWORDS


# In[ ]:





# In[26]:


wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(' '.join(positive_comments['comment_text']))


# In[28]:


plt.imshow(wordcloud)
plt.axis('off')


# In[29]:


wordcloud1=WordCloud(stopwords=set(STOPWORDS)).generate(' '.join(negative_comments['comment_text']))


# In[30]:


plt.imshow(wordcloud1)
plt.axis('off')


# In[31]:


import emoji


# In[32]:


all_emojis_list = []

for comment in df['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)


# In[34]:


all_emojis_list


# In[33]:


from collections import Counter


# In[35]:


Counter(all_emojis_list).most_common(10)


# In[37]:


emojis = [Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]


# In[38]:


freqs = [Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]


# In[ ]:





# In[ ]:





# In[36]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[39]:


trace = go.Bar(x=emojis , y=freqs)


# In[40]:


iplot([trace])


# In[ ]:





# In[ ]:





# In[66]:


import os


# In[67]:


files=os.listdir(r'C:\prac\additional_data')


# In[68]:


files_csv=[file for file in files if '.csv' in file]


# In[69]:


files_csv


# In[70]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:





# In[73]:


full_df=pd.DataFrame()
path=r'C:\prac\additional_data'
for file in files_csv:
    curr_df=pd.read_csv(path+'/'+file,encoding='iso-8859-1',error_bad_lines=False)
    full_df=pd.concat([full_df,curr_df],ignore_index=True)


# In[74]:


full_df.shape


# In[75]:


full_df[full_df.duplicated()].shape


# In[76]:


full_df = full_df.drop_duplicates()


# In[77]:


full_df.shape


# In[79]:


full_df.to_csv(r'C:\prac/youtube_sample.csv' , index=False)


# In[81]:


full_df.to_json(r'C:\prac/youtube_sample.json')


# In[86]:


json_df = pd.read_json(r"C:\prac\additional_data\US_category_id.json")
json_df


# In[87]:


cat_dict = {}

for item in json_df['items'].values:
    ## cat_dict[key] = value (Syntax to insert key:value in dictionary)
    cat_dict[int(item['id'])] = item['snippet']['title']


# In[89]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[90]:


plt.figure(figsize=(12,8))
sns.boxplot(x='category_name' , y='likes' , data=full_df)
plt.xticks(rotation='vertical')


# In[91]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[92]:


plt.figure(figsize=(8,6))
sns.boxplot(x='category_name' , y='like_rate' , data=full_df)
plt.xticks(rotation='vertical')
plt.show()


# In[93]:


sns.regplot(x='views' , y='likes' , data = full_df)


# In[94]:


full_df[['views', 'likes', 'dislikes']].corr()


# In[95]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr() , annot=True)


# In[96]:


full_df.head(6)


# In[97]:


full_df['channel_title'].value_counts()


# In[98]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[99]:


cdf = cdf.rename(columns={0:'total_videos'})
cdf


# In[100]:


import plotly.express as px


# In[101]:


px.bar(data_frame=cdf[0:20] , x='channel_title' , y='total_videos')


# In[ ]:





# In[ ]:





# In[102]:


import string


# In[103]:


string.punctuation


# In[104]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[105]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[106]:


sample = full_df[0:10000]


# In[107]:


sample['count_punc'] = sample['title'].apply(punc_count)


# In[108]:


sample['count_punc']


# In[109]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='views' , data=sample)
plt.show()


# In[110]:


plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc' , y='likes' , data=sample)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




