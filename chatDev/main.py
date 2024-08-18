import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import itertools
from sklearn.datasets import load_iris

class MLExperimentationTool:
    def __init__(self, data=None):
        if data is None:
            # Load the Iris dataset as an example
            iris = load_iris()
            self.data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                                     columns=iris['feature_names'] + ['target'])
        else:
            self.data = data
        
        self.X = self.data.drop('target', axis=1)
        self.y = self.data['target']
        self.feature_sets = []
        self.results = []

    def generate_feature_sets(self, max_features=None):
        all_features = list(self.X.columns)
        if max_features is None:
            max_features = len(all_features)
        
        for i in range(1, max_features + 1):
            self.feature_sets.extend(itertools.combinations(all_features, i))

    def run_experiments(self, test_size=0.2, random_state=42):
        for feature_set in self.feature_sets:
            X_subset = self.X[list(feature_set)]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, self.y, test_size=test_size, random_state=random_state
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(random_state=random_state)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.results.append({
                'features': feature_set,
                'accuracy': accuracy
            })

    def get_best_result(self):
        return max(self.results, key=lambda x: x['accuracy'])

    def print_results(self):
        for result in sorted(self.results, key=lambda x: x['accuracy'], reverse=True):
            print(f"Features: {result['features']}, Accuracy: {result['accuracy']:.4f}")

# Usage example with Iris dataset
print("Experimenting with Iris dataset:")
tool = MLExperimentationTool()  # This will automatically load the Iris dataset
tool.generate_feature_sets()  # Generate all possible feature combinations
tool.run_experiments()
tool.print_results()
print("\nBest result:", tool.get_best_result())

# If you want to use your own dataset, you can do:
# custom_data = pd.read_csv('your_data.csv')
# tool = MLExperimentationTool(custom_data)
# Then proceed with generate_feature_sets(), run_experiments(), etc.

print("\nNote: The Iris dataset used in this example can be found at:")
print("https://archive.ics.uci.edu/ml/datasets/iris")
print("It's also available through scikit-learn's datasets module, as demonstrated in the code.")