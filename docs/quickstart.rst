Quick Start Guide
=================

This guide will help you get SparkTrainer up and running in minutes.

Prerequisites
-------------

* Docker & Docker Compose
* NVIDIA GPU with CUDA 11.8+ (for training)
* 16GB+ RAM recommended
* 50GB+ disk space

Installation
------------

1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/def1ant1/SparkTrainer.git
   cd SparkTrainer

2. Start infrastructure services
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker-compose up -d postgres redis mlflow

3. Initialize database
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd backend
   python init_db.py --sample-data

4. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

5. Start backend & workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Terminal 1 - Backend:

.. code-block:: bash

   python backend/app.py

Terminal 2 - Celery Worker:

.. code-block:: bash

   celery -A backend.celery_app.celery worker --loglevel=info --concurrency=2

6. Start frontend
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd frontend
   npm install
   npm run dev

Access the Application
----------------------

* **Frontend**: http://localhost:3000
* **Backend API**: http://localhost:5000/api
* **MLflow**: http://localhost:5001
* **Flower**: http://localhost:5555

Next Steps
----------

* Read the :doc:`user_guide/index` for detailed usage instructions
* Explore the :doc:`api_reference/index` for API documentation
* Check out :doc:`examples/index` for code examples
* See the :doc:`developer_guide/index` if you want to contribute

Configuration
-------------

Create a ``.env`` file in the root directory:

.. code-block:: bash

   # Database
   DATABASE_URL=postgresql://sparktrainer:password@localhost:5432/sparktrainer

   # Redis
   REDIS_URL=redis://localhost:6379/0

   # MLflow
   MLFLOW_TRACKING_URI=http://localhost:5001

   # Flask
   FLASK_ENV=development
   SECRET_KEY=your-secret-key

See the full configuration guide in the :doc:`user_guide/configuration` section.
