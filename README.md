This project will try to find the similarities in playstyle between all players that participated in VCT 2022 Champions. We find that there are 6 groups of different playstyles for all professional players.

The paper for this project can be found [here](https://github.com/prakhosha/Valorant-VCT-2022-Player-s-Playstyle-Analysis/blob/main/Valorant%20VCT%202022%20Player's%20Playstyle%20Analysis.pdf)


# Introduction

This notebook contains an analysis on similirarity between all professional valorant players that played in VCT 2022 Champions.

The data comes from vlr.gg.

The data consists of:

1. player_name: Player name
2. player_url: Link to players profile on vlr.gg
3. player_RND: How many rounds the player played
4. player_rating: rating
5. player_ACS: Average combat score
6. player_KAST: Kill, assist, survive, trade %
7. player_KD_ratio: Kill-death ratio
8. player_ADR: Average damage per round
9. player_KPR: Average kill per round
10. player_APR: Average assist per round
11. player_FKPR: First kill per round
12. player_FDPR: First death per round
13. player_HS: Headshot %
14. player_CL_percentage: Clutch success %
15. player_CL: How many clutches the player won / how many clutches the player played
16. three agent picks for every player of all players that participated in VCT 2022 Champions.


```python
!pip install pingouin
print('')
!pip install adjustText

import pandas as pd
import numpy as np
import scipy.stats as stat
import pingouin as pg
import seaborn as sb
import matplotlib.pyplot as plt
from adjustText import adjust_text
```


```python
data_path = '/kaggle/input/valorant-vlr-vct-2022-per-round-data/vlr_vct2022_data.csv'
df = pd.read_csv(data_path)

df['player_CL_percent'] = df['player_CL_percent'].fillna('0%')
df['player_CL_percent'] = df['player_CL_percent'].str.rstrip('%').astype('float') / 100.0
df['player_HS'] = df['player_HS'].str.rstrip('%').astype('float') / 100.0
df['player_KAST'] = df['player_KAST'].str.rstrip('%').astype('float') / 100.0

print("Preview of the data:")
df.head()
```

## Correlation for All Stats to Rating

Before we dwell further, this analysis is supposed to analyze a player's playstyle without considering their ability in the game. So to accommodate this, I compute the correlation of all parameters in the game to the rating as rating is an indicator on how good the player is in playing the game. I then remove those parameters that have more than 0.5 correlation.


```python
vlr_stat = ['player_ACS', 'player_KAST', 'player_KD_ratio', 'player_ADR',
            'player_KPR', 'player_APR', 'player_FKPR', 'player_FDPR',
            'player_HS', 'player_CL_percent']

corr = df[vlr_stat].corrwith(df['player_rating'])
sb.set(rc={'figure.figsize':(13, 5)})
plot = sb.heatmap(pd.DataFrame(corr), annot=True)
plt.title("Pearson correlation of player rating")
plt.xlabel('Pearson correlation')

fig = plot.get_figure()
fig.savefig("/kaggle/working/pearson_corr1.png") 
```

Although player headshot percentage does not indicate any correlation to player rating, I still remove this from analysis because it is an indication of how good the player is at aiming.

Also, while the correlation shows that player’s clutch percentage does not correlate with player rating that much, I believe that putting it in is an unfair judgement because this parameter still contains information about how good someone is at winning the round.

Thus I suggest creating a new parameter called clutch situation frequency or CSF for short. This new parameter can be obtained by dividing the number of clutch situations they have played by the number of rounds they have played.




```python
df['player_CL'] = df['player_CL'].str.split('/').str[1].astype('float')
df['player_CSF'] = df['player_CL']/df['player_RND']
```


```python
vlr_stat = ['player_ACS', 'player_KAST', 'player_KD_ratio', 'player_ADR',
            'player_KPR', 'player_APR', 'player_FKPR', 'player_FDPR',
            'player_HS', 'player_CSF']

corr = df[vlr_stat].corrwith(df['player_rating'])
sb.set(rc={'figure.figsize':(13, 5)})
plot = sb.heatmap(pd.DataFrame(corr), annot=True)
plt.title("Pearson correlation of player rating")
plt.xlabel('Pearson correlation')

fig = plot.get_figure()
fig.savefig("/kaggle/working/pearson_corr2.png") 
```

As you can see, our new CSF parameter does not have any correlation with player rating.

Based on this result, we will use APR, FKPR, FDPR, and CSF to analyze a player's playstyle.

## APR, FKPR, FDPR, and CSF Distribution for All Professional Players

Also before we go further, I remove the agent pool because the data in vlr.gg can only show 3 agents that the player played. And I don't know if those agents are the most used agents by the player or not. So, it's best to remove the agent pool from the analysis.


```python
df.fillna('', inplace=True)
df['top_3_agents'] = df['player_top_3_agents_1_image'] + df['player_top_3_agents_2_image'] + df['player_top_3_agents_3_image']

# ----- Uncomment all below to include agents into analysis ----- #
#agents = ['astra', 'breach', 'brimstone', 'chamber', 'cypher',
#          'harbor', 'jett', 'kayo', 'killjoy', 'neon', 'omen',
#          'phoenix', 'raze', 'reyna', 'sage', 'skye', 'sova', 'viper', 'yoru']

#for agent in agents:
#    df[agent] = pd.np.where(df['top_3_agents'].str.contains(agent), 1, 0)
    
df.drop(columns=['player_url', 'player_RND', 'player_ADR', 'player_KPR', 'player_ACS',
                 'player_rating', 'player_KAST', 'player_KD_ratio', 
                 'player_HS', 'player_CL_percent', 'player_CL',
                 'player_top_3_agents_1_image',
                 'player_top_3_agents_2_image',
                 'player_top_3_agents_3_image',
                 'top_3_agents'], inplace=True)

df.head()
```

### APR


```python
sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df, x="player_APR", bins=10, stat="density", element="step", kde=True)
print(pg.normality(df['player_APR']))

mean_confi = stat.t.interval(alpha=0.95, df=len(df['player_APR'])-1, loc=np.mean(df['player_APR']), scale=stat.sem(df['player_APR']))
print('Mean with 95% confidence interval: {}'.format(mean_confi))

fig = plot.get_figure()
fig.savefig("/kaggle/working/APR_dist.png") 
```

Assist per round (APR) for all players through VCT 2022 seems distributed normally.

### FKPR


```python
sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df, x="player_FKPR", bins=20, stat="density", element="step", kde=True)
print(pg.normality(df['player_FKPR']))

mean_confi = stat.t.interval(alpha=0.95, df=len(df['player_FKPR'])-1, loc=np.mean(df['player_FKPR']), scale=stat.sem(df['player_FKPR']))
print('Mean with 95% confidence interval: {}'.format(mean_confi))

fig = plot.get_figure()
fig.savefig("/kaggle/working/FKPR_dist.png") 
```

First kill per round (FKPR) for all players through VCT 2022 Champions is not normally distributed. And it makes sense because FKPR a little bit relies on how skilled the player is so there are abnormalities on the data.

### FDPR


```python
sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df, x="player_FDPR", bins=15, stat="density", element="step", kde=True)
print(pg.normality(df['player_FDPR']))

mean_confi = stat.t.interval(alpha=0.95, df=len(df['player_FDPR'])-1, loc=np.mean(df['player_FDPR']), scale=stat.sem(df['player_FDPR']))
print('Mean with 95% confidence interval: {}'.format(mean_confi))

fig = plot.get_figure()
fig.savefig("/kaggle/working/FDPR_dist.png") 
```

The interesting thing is that based on FKPR I expect the distribution of FDPR would be left-skewed (because abnormal players that successfully entry should live longer than most players so result in small FDPR), but in fact it is right-skewed.

In my opinion, this is because in defense situations, most players tend to sit deep holding an angle while some of them hold a dangerous position resulting in right-skewed data.

This just shows that FPDR is one way to measure how aggressive a player is.

### CSF


```python
sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df, x="player_CSF", bins=20, stat="density", element="step", kde=True)
print(pg.normality(df['player_CSF']))

mean_confi = stat.t.interval(alpha=0.95, df=len(df['player_CSF'])-1, loc=np.mean(df['player_CSF']), scale=stat.sem(df['player_CSF']))
print('Mean with 95% confidence interval: {}'.format(mean_confi))

fig = plot.get_figure()
fig.savefig("/kaggle/working/CSF_dist.png") 
```

Clutch situation frequency also suggests an abnormal distribution.

# Distance for All Players


```python
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
```

I use a standard scaler to standardize any abnormal distributed data.


```python
df_visualize = df.drop(columns="player_name")

scaler = StandardScaler()
df_visualize = scaler.fit_transform(df_visualize)
df_visualize = pd.DataFrame(df_visualize)
```


```python
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS

D = pairwise_distances(df_visualize)
D.shape

sb.set(rc={'figure.figsize':(8, 8)})
plot = sb.heatmap(D, annot=False, xticklabels=False, yticklabels=False)
plt.title("Heatmap of Distances (in other word 'similarity') for All Players in VCT 2022 Champions")
plt.xlabel('Players, (Rank from Left to Right)')
plt.ylabel('Players, (Rank from Top to Bottom)')

fig = plot.get_figure()
fig.savefig("/kaggle/working/heatmap.png") 
```

The euclidean distances indicate there seems to be no particular pattern between high-rating players and low-rating players.


```python
model = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=5, random_state=1)
df_visualize = model.fit_transform(df_visualize)
df_visualize = pd.DataFrame(df_visualize)
df_visualize['player_name'] = df['player_name']
```


```python
sb.set(rc={'figure.figsize':(18, 15)})
sb.set_style("darkgrid", {'axes.grid' : False})

plot = sb.scatterplot(data=df_visualize, x=0, y=1, hue='player_name', legend=False)

texts = [plt.text(df_visualize[0][row],
                  df_visualize[1][row],
                  df_visualize['player_name'][row],
                  fontweight='semibold') for row, player in enumerate(df_visualize['player_name'])
        ]
adjust_text(texts)

plt.title("Player Similarity in VCT 2022 Champions based on APR, FKPR, FDPR, and how many clutch situation they have played (Clutch Situation Frequency/CSF)",
          fontdict={'fontsize': 15, 'fontweight':'semibold'})

plot.set(xlabel=None)
plot.tick_params(bottom=False)
plot.axes.xaxis.set_visible(False)
plot.axes.yaxis.set_visible(False)

fig = plot.get_figure()
fig.savefig("/kaggle/working/similarity.png") 
```

Visually we can see there are about 6 different groups of playstyle. Here is the analysis:

Note: Direct utilities does not mean utilities in general. Direct utilities means abilities that result in assist. As for abilities that do not result in assist such as those to take space, to hold space, or dummy abilities do not count in direct utilities.

1. Usually play as the deepest member in a site or the last one to entry a site and also tend to use direct utilities to help other members.

2. Play safely or passively and is the most likely group to use direct utilities (utilities that result in assist) to help other members. (except for Smoggy, the reason he is here is because he played really well as Jett and as a result he did not experience many first deaths, and he also played really well as Kay/o).

3. Play safely or passively and tend to use direct utilities to help other members.

4. Tend to play aggressively or to hold a dangerous position. This is the least likely group to use direct utilities other than the space taker.

5. Tend to play aggressively to be able to use direct utilities to help other members or tend to hold a dangerous position.

6. Designated as space taker and first contact players (hold a dangerous position). The most aggressive group of players.


### Further Analysis


```python
group1 = ['stellar', 'mindfreak', 'Benkai', 'Melser', 'CHICHOO', 'blaZek1ng',
          'sScary ', 'Derrek', 'SUYGETSU', 'tehbotoL', 'stax', 'nzr']

group2 = ['Smoggy', 'BcJ', 'Marved', 'bang', 'Mazino', 'Shyy', 'Boaster',
          'dephh', 'crashies', 'Mistic', 'pANcada', 'Sacy', 'Enzo', 'Mako', 'Shao']

group3 = ['delz1k', 'AYRIN', 'crow', 'Klaus', 'adverso', 'Zest', 'Nivera', 'Khalil', 
          'dimasick', 'SugarZ3ro', 'Crws', 'd4v41', 'soulcas', 'FNS']

group4 = ['foxz', 'Tacolilla', 'fl1pzjder', 'Famouz', 'Dep', 'Less', 'Haodong', 'nobody']

group5 = ['Mazin', 'saadhak', 'Asuna', 'ANGE1', 'Suhsiboys', 'Quick', 'Rb']

group6 = ['Surf', 'Life', 'f0rsakeN', 'BerserX', 'Jinggg', 'Victor', 'keznit', 'Alfajer', 
          'TENNN', 'Derke', 'aspas', 'zekken', 'BuZz', 'Scream', 'Will', 'ardiis', 'Zyppan', 
          'kiNgg', 'Jamppi', 'Cryocells', 'yay', 'dgzin', 'Zmjjkk', 'NagZ', 'Laz']

df['group'] = pd.np.where(df['player_name'].isin(group1), 1, 
                          pd.np.where(df['player_name'].isin(group2), 2, 
                                      pd.np.where(df['player_name'].isin(group3), 3, 
                                                  pd.np.where(df['player_name'].isin(group4), 4, 
                                                             pd.np.where(df['player_name'].isin(group5), 5, 6)))))

df.head()
```

### APR for All Groups


```python
df_viz = df[df['group'].isin([1, 2, 3, 4, 5, 6])]
palette = {1:"tab:blue",
           2:"tab:green", 
           3:"tab:olive",
           4:"tab:pink",
           5:"tab:orange",
           6:"tab:red"}

sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df_viz, x="player_APR", bins=10, stat="density", 
                   element="step", kde=True, hue='group', palette=palette)

fig = plot.get_figure()
fig.savefig("/kaggle/working/APR_dist_grouped.png") 
```

### FKPR for All Groups


```python
df_viz = df[df['group'].isin([1, 2, 3, 4, 5, 6])]
palette = {1:"tab:blue",
           2:"tab:green", 
           3:"tab:olive",
           4:"tab:pink",
           5:"tab:orange",
           6:"tab:red"}

sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df_viz, x="player_FKPR", bins=10, stat="density", 
                   element="step", kde=True, hue='group', palette=palette)

fig = plot.get_figure()
fig.savefig("/kaggle/working/FKPR_dist_grouped.png") 
```

### FDPR for All Groups


```python
df_viz = df[df['group'].isin([1, 2, 3, 4, 5, 6])]
palette = {1:"tab:blue",
           2:"tab:green", 
           3:"tab:olive",
           4:"tab:pink",
           5:"tab:orange",
           6:"tab:red"}

sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df_viz, x="player_FDPR", bins=10, stat="density", 
                   element="step", kde=True, hue='group', palette=palette)

fig = plot.get_figure()
fig.savefig("/kaggle/working/FDPR_dist_grouped.png") 
```

### CSF for All Groups


```python
df_viz = df[df['group'].isin([1, 2, 3, 4, 5, 6])]
palette = {1:"tab:blue",
           2:"tab:green", 
           3:"tab:olive",
           4:"tab:pink",
           5:"tab:orange",
           6:"tab:red"}

sb.set(rc={'figure.figsize':(10, 8)})
plot = sb.histplot(data=df_viz, x="player_CSF", bins=10, stat="density", 
                   element="step", kde=True, hue='group', palette=palette)

fig = plot.get_figure()
fig.savefig("/kaggle/working/CSF_dist_grouped.png") 
```

# Suggestion


1. Add agents pool for every player into analysis. This will incorporate agents pool as indication of different playstyle.

2. Add a new stat that shows when the player died as a respect to their team (first to die, second to die, third to die, etc). This will show who is the one that trades the space taker or who is the one that plays passively in a site execution.

## Thankyou for reading :)


```python

```
