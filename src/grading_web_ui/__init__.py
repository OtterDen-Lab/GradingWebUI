import logging.config
import os
import re
from pathlib import Path
import yaml


def setup_logging() -> None:
  env_path = os.environ.get("LOGGING_CONFIG")
  package_dir = Path(__file__).resolve().parent
  repo_root = package_dir.parent.parent

  candidates = [
    Path(env_path) if env_path else None,
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
    logging.config.dictConfig(config)
  else:
    # Fallback to basic configuration if logging.yaml is not found
    logging.basicConfig(level=logging.INFO)


# Call this once when your application starts
setup_logging()
