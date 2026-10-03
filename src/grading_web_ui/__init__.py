import logging.config
import os
import re
import sys
from pathlib import Path
import yaml


def setup_logging() -> None:
  env_path = os.environ.get("LOGGING_CONFIG")
  app_dir = Path(os.environ.get("GRADING_APP_DIR", Path.cwd()))
  package_dir = Path(__file__).resolve().parent
  repo_root = package_dir.parent.parent

  candidates = [
    Path(env_path) if env_path else None,
    # Native deployments install the package into a virtual environment, where
    # package_dir is under site-packages rather than the checkout. The service
    # intentionally sets GRADING_APP_DIR to the checkout containing this file.
    app_dir / "logging.yaml",
    Path.cwd() / "logging.yaml",
    repo_root / "logging.yaml",
    package_dir / "logging.yaml",
  ]

  config_path = next(
    (path for path in candidates if path and path.is_file()),
    None
  )

  if config_path:
    with config_path.open('r') as f:
      config_text = f.read()

    # Process environment variables in the format ${VAR:-default}
    def replace_env_vars(match) -> str:
      var_name = match.group(1)
      default_value = match.group(2)
      return os.environ.get(var_name, default_value)

    config_text = re.sub(r'\$\{([^}:]+):-([^}]+)\}', replace_env_vars,
                         config_text)
    config = yaml.safe_load(config_text)
    try:
      for handler in config.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
          Path(filename).parent.mkdir(parents=True, exist_ok=True)
      logging.config.dictConfig(config)
    except OSError as error:
      # Keep a directly launched development server usable when its user does
      # not have permission to create the production log directory.
      logging.basicConfig(level=logging.INFO, force=True)
      print(f"Could not initialize file logging: {error}; using console logging",
            file=sys.stderr)
  else:
    # Fallback to basic configuration if logging.yaml is not found
    logging.basicConfig(level=logging.INFO)


# Call this once when your application starts
setup_logging()
