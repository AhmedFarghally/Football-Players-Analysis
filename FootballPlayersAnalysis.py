#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate Database_Soccer
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This dataset contains two csv files, the first is "player" and the second is "player_attributes". The "player" csv files contains personal information about each player such as name, birth date, etc. The second file "player_attributes" which contains Players attributes sourced from EA Sports' FIFA video game series from 2007 to 2016, including the weekly updates. So, I will try to join the two files and make my analysis.
# 
# 
# ### Question(s) for Analysis
# As I mentioned above, I will try to join the "player" csv file with the "player attributes" csv file, to form just one dataset, and then I inspect it and figure out some questions, and then use pandas, numpy and matplotlib or seaborn or plotely express to not onyl find the answers to these question, but also make a suitable visualization of my answers. Some of the questions such as:
# Who were the top eleven players?
# Who were the top rated "Goalkeepers"?
# Who were the "Defenders" with the highest rating ?
# Who were the "Midfielders" with the highest rating?
# Who were the "Forwards" with the highest rating ?
# Which players had the highest penality kicking rating ?
# Who was the tallest player (in Centimeters) ?
# How does the Overall rating correlate with other players attributes ?
# Who was the fastest player ?
# What was the most prefered foot between players (Right foot or Left foot ) ?, and 
# At which year did EA Sports' FIFA video game company make the most updates on players_attributes ? 
# 
# 

# At first we load the required libraries to be used in our project.

# In[1]:


# Downloading plotly express library for data visualization
#! pip install plotly_express==0.4.0 
# Upgrade pandas to use dataframe.explode() function. 
#! pip install --upgrade pandas==0.25.0


# In[1]:


# Loading required libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# In this section of the report, we will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. 

# Let's read our "player" and "player_attributes" csv files and save them in "df_player" and "df_player_attributes" dataframes.

# In[2]:


# Loading our data 
df_player = pd.read_csv('Desktop/Player.csv')
df_player_attributes = pd.read_csv('Desktop/Player_Attributes.csv')


# In[3]:


df_player.head()
df_player_attributes.head(2)


# 
# ### Data Cleaning 

# Let's check if our datasets contain "null" values or not?, and if the datasets contain "null" values we will process these "null" values to get a clean datasets

# In[4]:


# check null values for (df_player)
df_player.isnull().sum()


# In[5]:


# check null values for (df_player_attributes)
df_player_attributes.isnull().sum()


# In[6]:


# Removing rows with null values from (df_player_attributes) dataset
df_player_attributes.dropna(inplace = True)


# In[7]:


# Check, if the rows with null values were deleted or not?! 
df_player_attributes.isna().sum()


# So, after checking the dataset for the null values, we found that the (df_player) has no any null values, but (df_player_attributes) has many null values in many columns, and as we can't get the exact values of these features, we decided to remove the rows with the null values, to make our data clean.

#                          ---------------------------------------------

# Let's check if our datasets contain "duplicated" values or not?, and if the datasets contain "duplicated" we will process these "duplicated" values to get a clean datasets.

# In[8]:


# Check duplicates for (df_player)
df_player.duplicated().sum()


# In[9]:


# Check duplicates for (df_player)
df_player_attributes.duplicated().sum()


# So we are lucky, and we hav not any duplicates in our datasets

#                               - - - - - - - - - - - - - - - - -

# Now let's check the data types of our datasets columns, and fix these types if needed.

# In[10]:


# Check datatypes for "df_player"
df_player.dtypes


# In[11]:


# For df_player dataset
# Let's convert "height" from string to float, and
# convert birthday from object to date.
df_player['height'] = df_player['height'].astype('int64')
df_player['birthday'] = pd.to_datetime(df_player['birthday'])


# In[12]:


# Check if they were converted or not
df_player.info()


# In[13]:


# Check datatypes for "df_player_attributes"
df_player_attributes.info()


# In[14]:


# convert "date" column from string datatype to date datatype
df_player_attributes['date'] = pd.to_datetime(df_player_attributes['date'])


# Before EDA, We must join df_player dataset with the df_player_attributes together to form just one dataset that contains all the needed information, let's do that!..

# In[15]:


# Joining df_player with df_player_attributes using pd.merge method
df = pd.merge(df_player, df_player_attributes, on=['player_fifa_api_id', 'player_api_id'])


# In[16]:


# Deleting unnecessary columns
df.drop(columns=['id_x', 'id_y', 'player_api_id', 'player_fifa_api_id'], inplace = True)


# Now we gonna make a new columns that we can use in our analysis later on.

# In[17]:


# Extract the years from the date, and put these years into a new column "year"
df['year'] = df['date'].dt.year


# In[18]:


df.head(1)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Now, we will move to the analysis phase, we will try to ask some questions and find answers to them, and by that we can analysis our data and we can make a strong conclusion about our findings.
#               ___________________________________________________________
# 
# 
# 
# 

# Now we can create a function that used some dataframe inputs, analyis the data, and then plot our result

# In[19]:


def myTopPlot(df, grpVar, yVar, stat = 'max', cols = None, top = 10, position = None):
    ''' This function used to take the dataframe, make the suitable analysis 
        then plot the result
        
        Inputs: df (dataframe)
                grpVar (specifc column in the dataframe say 'player_name')
                yVar (specifc column in the dataframe say 'overall_rating')
                stat (agg type,for example say 'max' to get the maximum overall rating)
                cols (columns selected from the dataframe)
                top (number of players)
                position (player position , for example : Goalkeeper)
        Outputs:
                Bar_chart of the results, and         
        '''
    
    if cols:
        tmp = df[cols].copy()
        tmp[yVar] = tmp[ [x for x in cols if x!= grpVar] ].mean(axis=1).round()
    else:
        tmp = df.copy()

    data = tmp.groupby([grpVar])[yVar].agg(stat).round().nlargest(top).reset_index()
    
    
    title = f"Top {top} {position}"
    
    # Data visualization using plotly express
    bar_chart = px.bar(data, x = grpVar, y= yVar, title = title, 
                       color = grpVar,text = yVar).update_layout(
                       xaxis_title = f"{grpVar.replace('_',' ')}".title(), 
                       yaxis_title=f"{yVar.replace('_',' ')}".title() 
                       )
    bar_chart.show()   
    return data                


#                       ************************************************

# ### Research Question 1: Who were the top eleven players?
# Note: The players will not be duplicated over the years (Only unique players will be displayed), for example if Lionel Messi has the highest rating in more than year we will take his highest rating only one time, and this method will be used for the rest of the questions.

# In[20]:


myTopPlot(df, 'player_name', 'overall_rating')


# ---> We notice that the Argentinian player "Lionel Messi" had the highest overall rating among the players during the ten years from 2007 to 2016.

#                 ***********************************************************

# ### Research Question 2 : Who were the top rated "Goalkeepers"?
# To answer this question, we should collect the Goalkeepers attributes from the dataset and then put them into another dataset, then analysis this new dataset to get the top 5 "goal keepers".
# 
# Note: We will take only the highest rating of the players over the ten years (from 2007 to 2016).

# In[21]:


# Consider the following goalkeeper attributes :
# gk_diving ,gk_handling ,gk_kicking, gk_positioning, and gk_reflexes are the most important features for a good goalkeeper, 
# and let's get the top 10 goalkeepers ratings through them.
GK_attributes = ['player_name','gk_diving','gk_handling','gk_kicking','reactions',
                 'gk_positioning', 'gk_reflexes']


# In[22]:


myTopPlot(df, 'player_name', 'overall_rating', cols = GK_attributes, position='Goalkeepers')


# ---> We notice that the Italian goalkeeper "Gianluigi Buffon" had the highest average rating among the rest of goalkeepers during the ten years from 2007 to 2106. 

#                   **************************************************

# ### Research Question 3 : Who were the "Defenders" with the highest rating ?
# 
# To answer this question, we should collect the "Defenders" attributes from the dataset and then put them into another dataset, then analysis this new dataset to get the top ten "Defenders".

# In[23]:


# Let's select Defending attributes and save it in a new dataframe
Df_attributes = ['player_name', 'jumping','stamina','strength','aggression',
                    'interceptions','marking','standing_tackle','sliding_tackle','positioning', 'reactions']


# In[24]:


myTopPlot(df, 'player_name', 'overall_rating', cols = Df_attributes, position='Defenders')


# ---> here, we notice that the English defender "John Terry" and the Spanish defender ""Carles Puyol" reached to the highest defending rating during the ten years from 2007 to 2016.

#                         **************************************************

# ### Research Question 4 : Who were the "Midfielders" with the highest rating?
# 
# To answer this question, we should collect the "Midfielders" attributes from the dataset and then put them into another dataset, then analysis this new dataset to get the top 10 "Midfielders".

# In[25]:


# Let's save the most important features of the midfielder in a list
Mid_attributes = ['player_name','short_passing', 'long_passing', 'ball_control', 
                    'agility', 'balance', 'stamina','crossing', 'vision','interceptions']


# In[26]:


# Display results
myTopPlot(df, 'player_name', 'overall_rating', cols=Mid_attributes, position= 'Midfielders')


# ---> So, the Spanish midfielder "Xavi Hernandez" had the highest rating during the ten years from 2007 to 2016

#                     *****************************************************

# ### Research Question 5 : Who were the "Forwards" with the highest rating ?
# 
# To answer this question, we should collect the "Forwards" attributes from the dataset and then put them into another dataset, then analysis this new dataset to get the top ten "Forwards".

# In[27]:


# Let's save the most important features of the Forward in a list
Fw_attributes = [
                 'player_name', 'ball_control', 'finishing', 'heading_accuracy', 
                 'balance', 'agility', 'stamina'
                ]


# In[28]:


myTopPlot(df, 'player_name', 'overall_rating', cols=Fw_attributes, position= 'Forwards')


# ### Research Question 6 : Which players had the highest penality kicking rating ?
# 

# In[29]:


Top_penality_kickers = df.groupby('player_name')['penalties'].max().nlargest(6)


# In[30]:


# Top5_penality_kickers dataframe
Top_five_penality_kickers = pd.DataFrame({'Player_name':Top_penality_kickers.index,
                                        'Highest_rating':Top_penality_kickers.values})


# In[31]:


Top_five_penality_kickers


# In[32]:


# Data visualization using matlibplot and seaborn libraries
# define Seaborn color palette to use
palette_color = sns.color_palette('Set2')
  
# plotting data on chart
plt.pie(Top_five_penality_kickers.Highest_rating, labels = Top_five_penality_kickers.Player_name,
        explode=[0.3, 0.1, 0.1, 0.1, 0.1,0] ,colors = palette_color,autopct ='%.1f%%', shadow = True)
plt.title("Top rated penality_kicker players")        
  
# displaying chart
plt.show()


# ---> So, the English  Centre-Forward "Rickie Lambert" was in the lead during the ten years from 2007 to 2016.

#                   ********************************************************

# ### Research Question 7 : Who was the tallest player (in Centimeters) ?
# 

# In[33]:


# Let's find the tallest player
Most_tallest_player = df.groupby('player_name')['height'].max().nlargest(1)


# In[34]:


# Tallest player datafram
Tallest_player = pd.DataFrame({'Player_name':Most_tallest_player.index,
                                  "Player_height":Most_tallest_player.values})


# In[35]:


Tallest_player


# We notice that the Belgian Goalkeeper "Kristof van Hout" was the tallest player.

# ### Research Question 8 : What was the most prefered foot between players (Right foot or Left foot ) ?
# 

# In[36]:


Right_lef_foot = df['preferred_foot'].value_counts()


# In[37]:


# Data visualization using matlibplot and seaborn libraries
# define Seaborn color palette to use
palette_color = sns.color_palette('RdYlGn')
  
# plotting data on chart
plt.pie(Right_lef_foot.values, labels = Right_lef_foot.index,explode=[0.1, 0.0] ,colors = palette_color, 
        autopct ='%.1f%%', shadow = True)
plt.title("Percentage of right foot players to left foot players")  
# displaying chart
plt.show()


# From the pie chart above we can notice that the right_foot player were dominant in numbers by 75.5%

#                 ******************************************************

#  ### Research Question 9 : Which players had the most Freekick accuracy ?

# In[38]:


# Let's select players with the highest freekick accuracy
Top_freekick_players = df.groupby('player_name')['free_kick_accuracy'].max().nlargest(6).round(1)


# In[39]:


# Now we can create our dataframe containing the players with the highest freekick accuracy
df_top_freekick_players = pd.DataFrame({'Player_name': Top_freekick_players.index,
                                        'Freekick_accuracy': Top_freekick_players.values})


# In[40]:


df_top_freekick_players


# In[41]:


# Data visualization using matlibplot and seaborn libraries
# define Seaborn color palette to use
palette_color = sns.color_palette('deep')
  
# plotting data on chart
plt.pie(df_top_freekick_players.Freekick_accuracy, labels = df_top_freekick_players.Player_name,
        explode=[0.3, 0.1, 0.1, 0.1, 0,0] ,colors = palette_color,autopct ='%.1f%%', shadow = True)
plt.title("Top rated free_kick players")        
#  displaying chart
plt.show()


# So, the Brazilian midfielder "Juninho Pernambucano" was the player with highest freekick accuracy.

#                     ******************************************************

#  ### Research Question 10 : Who were the fastest players ?
#  

# In[42]:


# Creating fast players dataframe
Fast_players_attributes = ['player_name', 'sprint_speed']
Fast_players = df[Fast_players_attributes] 
Fast_players['Speed_rating'] = (Fast_players.sprint_speed)


# In[43]:


# Select the fastest players
Fastest_palyers = df.groupby('player_name')['sprint_speed'].max().nlargest(10)


# In[44]:


# Creating our dataframe
Top_fastest_players = pd.DataFrame({'Player_name':Fastest_palyers.index,
                                     'speed_rating':Fastest_palyers.values})


# In[45]:


Top_fastest_players


# ---> here, we notice that the player "Mathis Bolly" from Côte d'Ivoire (Ivory Coast) and "David Odonkor" from Germany were the fastest players.

#                    ******************************************************

# ### Research Question 11 : At which year did EA Sports' FIFA video game company make the most updates on players_attributes ? 
# 

# In[46]:


Year = df['year'].value_counts()


# In[47]:


# Using plotely express for visualizing our answer
Pie_chart = px.pie(values = Year.values, labels= Year.index, hover_name = Year.index,
            title='Number of player_attributes update for each year', names= Year.index)
Pie_chart.update_traces(textposition='inside', textinfo='percent+label')
Pie_chart.update_traces(pull=[0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
Pie_chart.show()


# ---> We notice that "2013" had the most number of player_attributes updates by EAsports Fifa gaming company.

#               *****************************************************************

# ### Research Question 12 : How does the Overall rating correlate with other players attributes ?
# 

# To answer this question, let's get the correlation of the overall rating with other attributes

# In[48]:


df_corr_matrix = df.corr(method='pearson')


# In[49]:


df_corr_matrix[['overall_rating']]


# Now we can see that the "Overall rating" has a strong correlation with "Potential" attribute, "Potential" attribute means the ability of the player to develop, so it's logical that the "Overall rating" has this strong relationship with "Potential" attribute. To explain this idea farther more, let's take a player from our dataset, say "Kaka", and see how the "Potential" has a very strong effect on player progress, which here means increasing player "Overall rating"..

# In[50]:


# Filter our data to contain information only about "Lionel Messi"
Kaka = df[df['player_name'] == 'Kaka']


# In[51]:


# reset index to start from 0 to 1, 2, 3, etc.....
Kaka.reset_index(inplace = True)
del(Kaka['index'])


# In[52]:


# Beacouse the updates dates are ordered randomly so our data is out of order,
# so let's order our data by date
Kaka= Kaka.sort_values(['date'])


# In[53]:


# Now our data is in order by date
Kaka.head()


# In[54]:


# Now let's Figure out the relationship between "Overall rating" and "Potential"
Kaka.plot.scatter(x='potential', y='overall_rating')
plt.title('Relationship between Player Potential and Player Overall rating')
plt.xlabel('Player Potential')
plt.ylabel("Player Overall rating")
plt.show()


# From the above figure, we can notice that  "Over_all rating" is almost directly proportional to  "Potential attribute". This means that as "Potential attribute" increases, "Over_all rating" increases and as "Potential attribute" decreases, "Over_all rating" decreases.

#        ****-----------------------------------------------------------------------****

# <a id='conclusions'></a>
# ## Conclusions
# 
# At first, We have loaded libraries required.
# Then, we performed data cleaning (removed duplicates, removed rows with Nan values, modified data types). 
# Finally, Pedrformed our data analysis by answering some questions, and found answers:
#     Who were the top ten players? (Lionel Messi leaded the rating) 
#     Who were the top rated "Goalkeepers"? (Gianluigi Buffon leaded the rating)
#     Who were the "Defenders" with the highest rating ? (Carles Puyol leaded the rating)
#     Who were the "Midfielders" with the highest rating? (Xavi Hernandiz leaded the rating)
#     Who were the "Forwards" with the highest rating ? (Cristiano Ronaldo leaded the rating)
#     Which players had the highest penality kicking rating ? (Rickie Lambert leaded the rating)
#     Who was the tallest player (in Centimeters) ? (Kristof van Hout)
#     Who was the fastest player ?
#     What was the most prefered foot between players (Right foot or Left foot ) ? (Right foot) 
#     At which year did EA Sports' FIFA video game company make the most updates on         players_attributes ?  (2013)
#     How does the Overall rating correlate with other players attributes ? (potential attribute).
# 
# 
# Limitations:
# ____________
# The limitations of the data analysis are:
# 
# Issues with the dataset:
# 
# we (almost always) find issues with the sample of data that we are working with (missing observations, data that appears to be inconsistent
# Other limitations involve:
# 
# Issues with the methods of analysis:
# 
# Typically, a type of analysis that we would like to perform is not possible due to data limitations.
# There is no statistical inference performed in your analysis (to test the significance of the results that you found). While this is not required, it is still a limitation of your analysis.
# 
# I think that there are more things that I can investigate with data, I can ask many and many questions, and link the information to get answers and make a very useful and strong conclusion about the data, but also I hope that I did well, and I will try and try to develop myself.

# In[ ]:




