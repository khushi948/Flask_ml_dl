from flask import Flask
from app.api.api import api_bp,api_bp_classify,api_bp_cluster ,api_bp_association ,home_api,api_bp_fruit,api_bp_ai_real
import os
import logging

def create_app():
    app = Flask(__name__, template_folder= os.path.join(os.getcwd(), 'app', 'templates')) 
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),  # Log to a file
        ]
    )
    # app.logger = logging.getLogger("flask_app")

    app.register_blueprint(api_bp, url_prefix='/api_bp')
    app.register_blueprint(api_bp_classify, url_prefix='/api_bp_classify')
    app.register_blueprint(api_bp_cluster, url_prefix='/api_bp_cluster')
    app.register_blueprint(api_bp_association, url_prefix='/api_bp_association')
    app.register_blueprint(api_bp_fruit, url_prefix='/api_bp_fruit')
    app.register_blueprint(api_bp_ai_real, url_prefix='/api_bp_ai_real')
    app.register_blueprint(home_api, url_prefix='/')

    return app