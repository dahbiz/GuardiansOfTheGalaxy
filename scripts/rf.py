import pandas as pd
import numpy as np
import PyQt5
labels = pd.read_csv("Galaxy_data/training_solutions_rev1.csv")
labels=labels.drop(labels[labels['Class1.3'] > 0.5].index)
labels = labels[['GalaxyID', 'Class1.1', 'Class1.2', 'Class6.1']]
labels['Result'] = 'i'
labels.loc[(labels['Class1.1'] > 0.7) & (labels['Class6.1'] < 0.1), 'Result'] = 'e'
labels.loc[(labels['Class1.2'] > 0.7) & (labels['Class6.1'] < 0.1), 'Result'] = 's'
labels.loc[labels['Class6.1'] > 0.63, 'Result'] = 'o'
elip_df = labels[labels['Result'] == 'e']
spiral_df = labels[labels['Result'] == 's']
odd_df = labels[labels['Result'] == 'o']

# Sample 5000 values from each category
category1_sampled = elip_df.sample(n=1000, random_state=42)
category2_sampled = spiral_df.sample(n=1000, random_state=42)
category3_sampled = odd_df.sample(n=1000, random_state=42)

# Concatenate the sampled DataFrames back together
sampled_df = pd.concat([category1_sampled, category2_sampled, category3_sampled])

galaxy =  sampled_df.sort_values(by='GalaxyID')


pca = np.load("pca_09.npy", allow_pickle=True)
df = pd.DataFrame(pca)

df.columns = df.iloc[0]
total_pixels = len(df.columns) 
header = [f'Comp_{i}' for i in range(1, total_pixels + 1)]
df.columns = header
new_column = galaxy['Result']
new_column = new_column.values.astype(str)
df.insert(0, 'Shape', new_column)

X=df.drop(columns=['Shape'])
y=df['Shape']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline=Pipeline([('rf',RandomForestClassifier())])

param_grid={
    'rf__n_estimators':[50,100,150],
    'rf__max_depth':[None,10,20],
    'rf__min_samples_split':[2,5,10],
    'rf__min_samples_leaf':[1,2,4],
    'rf__max_features':['auto','sqrt','log2']
}

from sklearn.model_selection import GridSearchCV
grid_search= GridSearchCV(estimator=pipeline,
                         param_grid=param_grid,
                         cv=5)

grid_search.fit(X_train,y_train)
best_params = grid_search.best_params_
final_model = pipeline.set_params(**best_params)
final_model.fit(X_train,y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

# classification ifo
print("Classification Report:")
print(classification_report(y_test,y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))

