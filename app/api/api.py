from flask import Blueprint, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import  PolynomialFeatures



api_bp = Blueprint('api_bp', __name__)
api_bp_classify = Blueprint('api_bp_classify', __name__)
api_bp_cluster = Blueprint('api_bp_cluster', __name__)
api_bp_association = Blueprint('api_bp_association', __name__)
api_bp_fruit = Blueprint('api_bp_fruit', __name__)
api_bp_ai_real = Blueprint('api_bp_ai_real', __name__)
api = Blueprint('api', __name__)
home_api = Blueprint('home_api', __name__)


models = {
    'linear_model': joblib.load('app/models/regression/linear_model.joblib'),
    'multiple_model': joblib.load('app/models/regression/multiple_model.joblib'),
    'svr_model': joblib.load('app/models/regression/svr_model.joblib'),
    'rf_model': joblib.load('app/models/regression/rf_model.joblib'),
    'regressor': joblib.load('app/models/regression/regressor.joblib'),
    'poly_model': joblib.load('app/models/regression/poly_model.joblib')
}

    

poly_scaler = joblib.load('app/models/regression/preprocessing/poly_scaler.joblib')
scaler_all = joblib.load('app/models/regression/preprocessing/scaler_all.joblib')
svm_scaler = joblib.load('app/models/regression/preprocessing/svm_scaler.joblib')
scaler_k_best = joblib.load('app/models/regression/preprocessing/scaler_k_best.joblib')

scaler_xgb=joblib.load('app/models/classification/preprocessing/scaler_xgb.joblib')
scaler_k=joblib.load('app/models/classification/preprocessing/scaler_k.joblib')

models_c={
    'rf_ros': joblib.load('app/models/classification/rf_ros.joblib'),
    'rf_kros': joblib.load('app/models/classification/rf_kros.joblib'),
    'dt_ros': joblib.load('app/models/classification/dt_ros.joblib'),
    'dt_kros': joblib.load('app/models/classification/dt_kros.joblib'),
    'lr': joblib.load('app/models/classification/lr.joblib'),
    'svm_1': joblib.load('app/models/classification/svm_1.joblib'),
    'knn_1': joblib.load('app/models/classification/knn_1.joblib'),
    'xgb': joblib.load('app/models/classification/xgb.joblib'),
    'lr_k': joblib.load('app/models/classification/lr_k.joblib'),
    'svm_k': joblib.load('app/models/classification/svm_k.joblib'),
    'knn_k': joblib.load('app/models/classification/knn_k.joblib')
}


model_cluster={
    'pca': joblib.load('app/models/clustering/pca.joblib'),
    'kmeans_pca': joblib.load('app/models/clustering/kmeans_pca.joblib'),
    'agg_clustering': joblib.load('app/models/clustering/agg_clustering.joblib'),
    'db': joblib.load('app/models/clustering/db.joblib'),
   
   
}


std_scaler=joblib.load('app/models/clustering/preprocessing/std_scaler.joblib')



@home_api.route('/')
def home():
    return render_template('home.html')

@api_bp.route('/regression')
def regression_home():
    return render_template('regression.html')


@api_bp_classify.route('/classification')
def classification_home():
    return render_template('classification.html')


@api_bp_cluster.route('/cluster')
def cluster_home():
    return render_template('cluster.html')



@api_bp_association.route('/association')
def association_home():
    return render_template('association.html')



@api_bp_fruit.route('/fruit')
def fruit_home():
    return render_template('fruit.html')
@api_bp_ai_real.route('/ai_real')
def ai_real_home():
    return render_template('ai_real.html')


@api_bp.route('/predict_regression', methods=['POST'])
def predict_regression():
    try:
        
        data = request.get_json()  
        model_name = data.get('model')  
        features = np.array(data['features']).reshape(1, -1)  
        
        if model_name=='linear_model':
            features_scaled = scaler_all.transform(features)
            prediction = models[model_name].predict(features_scaled)
        elif model_name=='multiple_model':
            X_k_best = scaler_k_best.transform(features)
            X_train_scaled_k_best = scaler_k_best.fit_transform(X_k_best)
            prediction = models[model_name].predict(X_train_scaled_k_best)
        elif model_name=='regressor':
            prediction = models[model_name].predict(features)
        elif model_name=='rf_model':
            prediction = models[model_name].predict(features)
        elif model_name=='svr_model':
            X_k_best = scaler_k_best.transform(features)
            X_train_scaled_k_best = svm_scaler.fit_transform(X_k_best)
            prediction =models[model_name].predict(X_train_scaled_k_best)
        elif model_name == 'poly_model':
    
            X_k_best = scaler_k_best.transform(features)  
            # X_poly_k_best = poly_scaler.fit_transform(X_k_best)  
            # X_poly_k_best_scaled = poly_scaler.fit_transform(X_poly_k_best)
            # prediction = models[model_name].predict(X_poly_k_best_scaled) 
            # X_k_best = scaler_k_best.transform(features)
            # X_train_scaled_k_best = svm_scaler.fit_transform(X_k_best)
            # prediction =models[model_name].predict(X_train_scaled_k_best)

            poly = PolynomialFeatures(degree=2) 
            X_train_poly = poly.fit_transform(X_k_best)

            # X_train_poly_scaled = poly_scaler.fit_transform(X_train_poly)

           
            prediction = models[model_name].predict(X_train_poly)

        

        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400





@api_bp_classify.route('/predict_classify', methods=['POST'])
def predict_classification():
    try:
        data = request.get_json()  
        model_name = data.get('model')  
        features = np.array(data['features']) 

        indices = [1,3, 5, 8, 10, 11, 19, 24, 25, 29, 30, 31]  

        
    
        input_array1 = np.array(features).reshape(1, -1)  
        
        
        
            

        if model_name == 'xgb': 
            input_scaled = scaler_xgb.transform(features.reshape(1, -1))
            prediction = models_c[model_name].predict(input_scaled)

        else:
            input_scaled = scaler_k.transform(input_array1)
            prediction = models_c[model_name].predict(input_scaled)
            
        

        
        class_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
        predicted_class = class_mapping.get(prediction[0], "Unknown")
        print("predicted", predicted_class)

        return jsonify({'prediction': predicted_class})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    


@api_bp_cluster.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    try:
        data = request.get_json()  
        model_name = data.get('model')  
        features = np.array(data['features']).reshape(1, -1)   
        print(features)

        user_input_std = std_scaler.transform(features)  

        user_input_pca = model_cluster['pca'].transform(user_input_std) 
        x_input, y_input = user_input_pca[0]

        if model_name=='kmeans_pca':

            
            
            prediction = model_cluster['kmeans_pca'].predict(user_input_pca)[0]
            prediction = prediction.item()
        elif model_name=='agg_clustering':
            # agg_cluster = model_cluster[model_name].fit_predict(np.vstack([df_pca, user_input_pca]))[-1]
            # prediction = model_cluster[model_name].fit_predict(user_input_pca)[-1]
            # prediction = prediction.item()
            # Predict cluster for the user input using the pre-fitted model
            distances = pairwise_distances_argmin_min(user_input_pca, model_cluster[model_name].fit(user_input_pca).cluster_centers_)
            prediction = distances[0] 
        else:
            prediction = model_cluster[model_name].fit_predict(user_input_pca)[-1]  # Check DBSCAN cluster
            prediction = prediction.item()

        cluster_names = {
        0: "Moderate Income, Average Spenders(0)",
        1: "High Income, Low Spenders(1)",
        2: "Low Income, High Spenders(2)",
        3: "High Income, High Spenders(3)"
    }

        
        cluster_label = cluster_names.get(prediction, "Noise Cluster")

        # print(f"Predicted Cluster: {cluster_label}")
        return jsonify({'prediction': cluster_label})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

from collections import Counter
from mlxtend.frequent_patterns import association_rules
import pandas as pd

apriori= joblib.load('app/models/association/apriori_model.joblib')
fpgrowth= joblib.load('app/models/association/frequent_itemsets_fpgrowth.joblib')
te = joblib.load('app/models/association/transaction_encoder.joblib')
import os
model_dir='app/models/association'

@api_bp_association.route('/predict_association', methods=['POST'])
def predict_association():
    try:
        data = request.get_json(force=True)
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        if 'model' not in data:
            return jsonify({"error": "No model specified"}), 400
            
        if 'items' not in data:
            return jsonify({"error": "No items provided"}), 400
        
        items = []
        for item in data['items']:
            if isinstance(item, str):
                item = item.strip('"[]\'')
            items.append(str(item))
        
        model_type = data['model'] 
        
        try:
            if model_type == 'apriori':
                rules = joblib.load(os.path.join(model_dir, "association_rules.joblib"))
                model_name = "Apriori"
            elif model_type == 'fp_growth':
                rules = joblib.load(os.path.join(model_dir, "fpgrowth_rules.joblib"))
                model_name = "FP-Growth"
            else:
                return jsonify({"error": f"Unknown model type: {model_type}"}), 400
                
            print(f"Successfully loaded {model_name} model from {model_dir}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return jsonify({"error": f"Failed to load {model_type} model: {str(e)}"}), 500
        
        user_input = set(items)
        
        if not user_input:
            return jsonify({"error": "Empty items list"}), 400
        
        print(f"Processing {model_name} recommendation request for items: {user_input}")
        if rules.empty:
            recommended_items, confidence_scores = [], []
        else:
            # Filter rules where ANY of the input items appear in antecedents
            mask = rules['antecedents'].apply(lambda x: len(user_input.intersection(x)) > 0)
            filtered = rules[mask]
            
            # Further filter by confidence threshold
            min_confidence = 0.1
            filtered = filtered[filtered['confidence'] >= min_confidence]
            
            # If no rules found, return empty lists
            if filtered.empty:
                recommended_items, confidence_scores = [], []
            else:
                # Sort by confidence and lift
                filtered = filtered.sort_values(by=["confidence", "lift"], ascending=[False, False])
                
                # Collect consequents with their confidence scores
                item_scores = {}
                for _, row in filtered.iterrows():
                    for item in row['consequents']:
                        # Skip items already in the input set
                        if item in user_input:
                            continue
                        
                        # Update with higher confidence if exists, otherwise add new
                        if item not in item_scores or row['confidence'] > item_scores[item]:
                            item_scores[item] = row['confidence']
                
                # Sort items by confidence score and limit to max_items
                max_items = 10
                sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:max_items]
                
                # Separate items and scores
                recommended_items = [item for item, score in sorted_items]
                confidence_scores = [round(score * 100, 1) for _, score in sorted_items]
        
    
        return jsonify({
            "status": "success",
            "recommended_items": recommended_items,
            "confidence_scores": confidence_scores
        })
        
    except Exception as e:
        print(f"Exception in predict_association: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
    
import subprocess
# TENSORFLOW_ENV = "C:/Users/Admin/AppData/Local/Programs/Python/Python310/python.exe"  # Update with your TensorFlow venv path

from werkzeug.utils import secure_filename
import subprocess
import os
import tempfile
import re
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
@api_bp_fruit.route('/predict_fruit', methods=['POST'])
def predict_fruit():
    try:
            
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name

        # Choose model (default: model1)
        model_name = request.form.get("model", "model1")  

        # Call Python script using subprocess
        result = subprocess.run(
            ["C:/Users/Admin/ML_flask/tf_env/Scripts/python.exe", "C:/Users/Admin/ML_flask/app/api/predict_fruit.py",temp_filepath, model_name],
            capture_output=True,
            text=True
        )

        # Remove temp file after processing
        os.remove(temp_filepath)

        if result.returncode != 0:
            return jsonify({"error": "Prediction failed", "details": result.stderr}), 500

        match = re.search(r'\b(Apple|Banana)\b', result.stdout, re.IGNORECASE)
        prediction = match.group(0) if match else "Unknown"

        

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)})

@api_bp_ai_real.route('/predict_ai_real', methods=['POST'])
def predict_ai_real():
    try:
            
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Secure filename
        filename = secure_filename(file.filename)

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name

        # Choose model (default: model1)
        model_name = request.form.get("model", "model2")  

        # Call Python script using subprocess
        result = subprocess.run(
            ["C:/Users/Admin/ML_flask/tf_env/Scripts/python.exe", "C:/Users/Admin/ML_flask/app/api/predict_ai_real.py",temp_filepath, model_name],
            capture_output=True,
            text=True
        )

        # Remove temp file after processing
        os.remove(temp_filepath)

        if result.returncode != 0:
            return jsonify({"error": "Prediction failed", "details": result.stderr}), 500

        match = re.search(r'\b(Real|Fake)\b', result.stdout, re.IGNORECASE)
        prediction = match.group(0) if match else "Unknown"

        

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)})

