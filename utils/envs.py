import os

project_path = os.getenv("PROJECT_PATH")
logger_path = os.path.join(project_path, "logger")
output_path = os.path.join(project_path, "output")

logger_file = os.path.join(logger_path, "logger.log")
