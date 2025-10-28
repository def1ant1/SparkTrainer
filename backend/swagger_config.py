"""
Swagger/OpenAPI configuration for Flask application
"""

from flasgger import Swagger
from openapi_spec import generate_openapi_spec


def init_swagger(app):
    """Initialize Swagger UI for the Flask application"""

    # Get the OpenAPI spec
    spec = generate_openapi_spec()

    # Configure Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/docs"
    }

    swagger_template = {
        "swagger": "2.0",
        "info": spec["info"],
        "host": "localhost:5001",
        "basePath": "/",
        "schemes": ["http", "https"],
        "securityDefinitions": {
            "Bearer": {
                "type": "apiKey",
                "name": "Authorization",
                "in": "header",
                "description": "JWT Authorization header using the Bearer scheme. Example: 'Authorization: Bearer {token}'"
            }
        },
    }

    # Initialize Swagger
    swagger = Swagger(app, config=swagger_config, template=swagger_template)

    # Also serve the OpenAPI 3.0 spec
    @app.route('/openapi.json')
    def openapi_spec():
        from flask import jsonify
        return jsonify(spec)

    @app.route('/openapi.yaml')
    def openapi_yaml():
        from flask import Response
        import yaml
        yaml_spec = yaml.dump(spec, default_flow_style=False, sort_keys=False)
        return Response(yaml_spec, mimetype='text/yaml')

    return swagger
