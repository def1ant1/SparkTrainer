SparkTrainer Documentation
==========================

Welcome to SparkTrainer's documentation! SparkTrainer is a production-ready multimodal AI training platform with comprehensive MLOps capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   user_guide/index
   api_reference/index
   developer_guide/index
   recipes/index
   examples/index

Overview
--------

SparkTrainer is an enterprise-grade machine learning training platform that combines powerful distributed training, automated data ingestion, experiment tracking, and production deployment tools into a unified system.

Key Features
------------

* **Job & Tracking Foundation**: Celery + Redis for async job processing with PostgreSQL backend
* **Data Ingestion**: Video-first workflow with frame extraction, captioning, and scene detection
* **Trainer Unification**: Recipe-based system with LoRA/QLoRA support
* **MLOps Features**: Model registry, inference adapters, A/B testing, GPU scheduling
* **UX Dashboards**: Project organization, leaderboard, and live metrics

Quick Links
-----------

* `GitHub Repository <https://github.com/def1ant1/SparkTrainer>`_
* `Issue Tracker <https://github.com/def1ant1/SparkTrainer/issues>`_
* `Discussions <https://github.com/def1ant1/SparkTrainer/discussions>`_

Installation
------------

.. code-block:: bash

   git clone https://github.com/def1ant1/SparkTrainer.git
   cd SparkTrainer
   pip install -r requirements.txt
   docker-compose up -d postgres redis mlflow

See the :doc:`quickstart` guide for detailed installation instructions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
