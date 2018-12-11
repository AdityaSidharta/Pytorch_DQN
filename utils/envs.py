import os

project_path = os.getenv('PROJECT_PATH')
logger_path = os.path.join(project_path, 'logger')

logger_file = os.path.join(logger_path, 'logger.log')