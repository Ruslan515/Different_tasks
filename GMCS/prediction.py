#!/usr/bin/env python

from catboost import Pool, CatBoostClassifier

def predict(X):
    file_model = "model_catboost.dump"
    model = CatBoostClassifier()
    model.load_model(file_model)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, y_proba

def convert_data(body):
    pass

def main_p(df_json):
    body = df_json["body"]
    X = convert_data(body)
    count, count_prob = predict(X)
    ans = {
        'id_quest': df_json['body']['id_quest'],
        'body': {
            "ans": count,
            "prob": count_prob,
        }
    }
    return ans
    
