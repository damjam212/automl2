import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from Preprocess import PaddingEstimator, add_pad, extract_features_with_window, process_labels_with_window, WindowFeatureExtractor, WindowLabelProcessor

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

class CustomPipeline(Pipeline):
    def fit(self, X, y=None):
        # Wypisanie kształtów przed transformacją
        print(f'Original X shape: {X[0].shape}')
        print(f'Original y shape: {y[0].shape}')
        
        # Wywołanie oryginalnej metody fit
        super().fit(X, y)
        
        # Wypisanie kształtów po transformacji
        for step_name, step_transformer in self.steps:
            if hasattr(step_transformer, 'transform'):
                X, y = step_transformer.transform(X, y)  # Zastosowanie transformacji
                print(f'Post-transformation X shape: {X[0].shape}')
                print(f'Post-transformation y shape: {y[0].shape}')
        print(f'X type: {type(X)}')
        print(f'y type: {type(y)}')
        return self

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class AutoMlMultiLabelClassifier:
    def __init__(self, model=None):
        """

        Args:
            model (sklearn.base.BaseEstimator, optional): Model klasyfikacyjny. 
                Domyślnie RandomForestClassifier.
        """
        self.model = model if model else RandomForestClassifier()
        self.is_fitted = False

    def fit(self, X, y):
        """
        Trenuje model na podanych danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Etykiety (labels).
        """
        try:
                        
            # Definicja modelu
            model = OneVsRestClassifier(XGBClassifier(n_jobs=8, max_depth=20, n_estimators=1000,
                                                       eval_metric='auc', objective='binary:hinge',
                                                       tree_method='hist'))
            
            param_grid = {
                # 'estimator__max_depth': [5, 10, 15, 20],
                'estimator__n_estimators': [500, 1000, 1500],
                # 'estimator__learning_rate': [0.01, 0.1, 0.2],
            }
                        
            # Tworzenie pipeline
            one_rep = 50
            divison = 1
            pipeline = Pipeline([
                ('feature_extraction', WindowFeatureExtractor(window_size=one_rep, step_size=one_rep // divison)),
                ('label_processing', WindowLabelProcessor(window_size=one_rep, step=one_rep // divison)),
                ('classifier', model)
            ])
            
            # Przekształcanie danych
            feature_extractor = pipeline.named_steps['feature_extraction']
            label_processor = pipeline.named_steps['label_processing']
            classifier = pipeline.named_steps['classifier']
            X = feature_extractor.transform(X)
            y = label_processor.transform(y)
            y = y[:, [0, 1, 2, 3, 5, 7]]
            
            # Podział na dane treningowe i testowe
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Inicjalizacja GridSearchCV
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            
            # Dopasowanie modelu
            grid_search.fit(X_train, y_train)
            
            # Najlepsze parametry
            print(f'Best Parameters: {grid_search.best_params_}')
            print(f'Best Cross-Validation Score: {grid_search.best_score_}')
            
            # Predykcja na zbiorze testowym
            y_pred = grid_search.best_estimator_.predict(X_test)
            
            # Ocena modelu
            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')
            
            recall = recall_score(y_test, y_pred, average='weighted')
            print(f'Recall: {recall}')

            # self.model.fit(X, y)
            # self.is_fitted = True
            
            # pipeline = CustomPipeline([
            #     ('padding', PaddingEstimator()),  # Pierwszy krok: PaddingEstimator
            #     ('classifier', MultiOutputClassifier(xgb.XGBClassifier(n_estimators=100)))
            # ], verbose = True)  # Umożliwia wyświetlanie komunikatów w trakcie działania pipeline

            # # Teraz możemy użyć pipeline do dopasowania modelu
            # pipeline.fit(X, y)
            
            # X, y = add_pad(X,y)
            # print(f'PAD-transformation X shape: {X.shape}')
            # print(f'PAD-transformation y shape: {y.shape}')       
            
#             X = extract_features_with_window(X)
#             print(f'Post-transformation X shape: {X.shape}')
#             y = process_labels_with_window(y)
#             # print(f'Post-transformation y shape: {y.shape}')            



#             # # Używamy PCA do zmniejszenia wymiarowości
#             # pca = PCA(n_components=300)  # Wybieramy 100 głównych składowych
#             # X = pca.fit_transform(X)

#             y=y[:, [0, 1, 2, 3, 5, 7]]
#             print(f'Post-transformation y shape: {y.shape}')            

#             # Podział na zbiór treningowy i testowy
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
#             # Tworzymy klasyfikator XGBoost
#             model = XGBClassifier(
#             n_estimators=10,
#             objective='binary:logistic',
#             tree_method='hist',
#             multi_strategy='multi_output_tree',
#             random_state=42
#                                  )  # Zwiększenie głębokości drzew           
#             # multi_output_model = MultiOutputClassifier(model)
#             # Trenowanie modelu
            #     model = OneVsRestClassifier(XGBClassifier(n_jobs=8, max_depth=20, n_estimators=1000,
            #                                                   eval_metric= 'auc',  
            #     objective='binary:hinge',
            #     tree_method='hist',))
            #     one_rep = 50
            #     divison = 1
            #     pipeline = Pipeline([
            #     ('feature_extraction', WindowFeatureExtractor(window_size=one_rep, step_size=one_rep // divison)),
            #     ('label_processing', WindowLabelProcessor(window_size=one_rep, step=one_rep // divison)),
            #     ('classifier', model)
            # ])  

            #     feature_extractor = pipeline.named_steps['feature_extraction']
            #     label_processor = pipeline.named_steps['label_processing']
            #     classifier = pipeline.named_steps['classifier']
            #     X = feature_extractor.transform(X)
            #     y = label_processor.transform(y)                
            #     y=y[:,[0, 1, 2, 3, 5, 7]]

            #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
            #     classifier.fit(X_train, y_train)
            #     print("predykcja")

            #     y_pred = classifier.predict(X_test)
            #     accuracy = accuracy_score(y_test, y_pred)
            #     print(f'Accuracy: {accuracy}')
                
            #     recall = recall_score(y_test, y_pred, average='weighted')
            #     print(f'Recall: {recall}')
            
            #             model.fit(X_train, y_train)            
#             # Predykcja na zbiorze testowym
# # Uzyskanie prawdopodobieństw dla każdej klasy
# # Uzyskanie prawdopodobieństw dla każdej klasy
#             y_pred = model.predict(X_test)
#             y_prob = model.predict_proba(X_test)
#             print(y_prob.shape)
#             # print(y_prob)
#             # # Dostosowanie progu decyzyjnego
#             # threshold = 0.3  # Zmniejszamy próg decyzyjny
#             # print('len(y_prob): ',len(y_prob))
#             # print('len(X_test): ',len(X_test))
#             # y_pred = (np.array(y_prob) > threshold).astype(int)  # Przypisujemy 1, jeśli prawdopodobieństwo > threshold, w przeciwnym razie 0


        

#             class_distribution = np.sum(y, axis=0)

#             print("Rozkład klas:")
#             for i, count in enumerate(class_distribution):
#                 print(f"Klasa {i}: {count} wystąpień")
            
#             print("Przykładowe trafione predykcje (100% poprawności, nie będące samymi zerami):")
#             for i in range(len(y_test)):  # Iterujemy przez wszystkie przykłady
#                 if np.array_equal(y_test[i], y_pred[i]) and not np.all(y_test[i] == 0):  # Sprawdzamy trafienie i brak samych zer
#                     print(f"Przykład {i+1}:")
#                     print(f"  Prawdziwe etykiety: {y_test[i]}")
#                     print(f"  Predykcja: {y_pred[i]}")
#                     print("-" * 30)
            # classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))    
            # X_flat = X.reshape(X.shape[0], -1)  # Przekształcenie na (samples, timesteps * features)
            # y_flat = y.reshape(y.shape[0], -1)  # Przekształcenie na (samples, timesteps * features)
            # # Dopasowanie modelu
            # classifier.fit(X_flat, y_flat)
            # # print(X[0].shape)
            # # print(y[0].shape)
            # # print(X[1].shape)
            # # print(y[1].shape) 
            # print("Model został wytrenowany.")
        except Exception as e:
            print(f"Błąd podczas trenowania modelu: {e}")

    def predict(self, X):
        """
        Przewiduje etykiety dla podanych danych.

        Args:
            X (np.ndarray): Dane wejściowe (features).

        Returns:
            np.ndarray: Przewidywane etykiety.
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed predict.")
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            print(f"Błąd podczas przewidywania: {e}")
            return None

    def score(self, X, y):
        """
        Oblicza dokładność modelu na podanych danych testowych.

        Args:
            X (np.ndarray): Dane wejściowe (features).
            y (np.ndarray): Rzeczywiste etykiety (labels).

        Returns:
            float: Dokładność modelu.
        """
        if not self.is_fitted:
            raise ValueError("Model nie został jeszcze wytrenowany. Użyj metody fit przed score.")
        try:
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy
        except Exception as e:
            print(f"Błąd podczas obliczania dokładności: {e}")
            return None

# Przykład użycia
if __name__ == "__main__":
    # Generowanie przykładowych danych
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Użycie klasy
    cls = AutoSklearnClassifier()
    cls.fit(X_train, y_train)
    predictions = cls.predict(X_test)
    accuracy = cls.score(X_test, y_test)

    print("Przewidywania:", predictions[:10])
    print("Dokładność:", accuracy)
