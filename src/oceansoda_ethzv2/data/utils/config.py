from typing import Optional

import dotenv
import yaml


def read_dataset_config(
    yaml_file: str,
) -> dict:
    """ """
    from jinja2 import UndefinedError

    # read yaml as raw string
    with open(yaml_file, "r") as file:
        text = file.read()

    # parse jinja2 placeholders
    fname_dotenv = get_dotenv_fname()
    env_vars = get_dotenv_variables(fname_dotenv)

    try:
        parsed_text = parse_jinja2_placeholders(text, **env_vars)
    except UndefinedError as e:
        raise UndefinedError(
            f"Variable in '{yaml_file}' is undefined in '{fname_dotenv}': {e.message}"
        ) from e

    # parse yaml
    config = yaml.safe_load(parsed_text)

    return config


def get_dotenv_variables(fpath: Optional[str] = None) -> dict:
    if fpath is None:
        fpath = dotenv.find_dotenv()

    if fpath:
        env_vars = dotenv.dotenv_values(fpath)
        return env_vars
    else:
        raise FileNotFoundError(
            f"Environment file '{fpath}' not found in the current directory or its parents."
        )


def get_dotenv_fname(name: str = ".env", relative: bool = True) -> Optional[str]:
    import pathlib

    fpath = dotenv.find_dotenv(name)
    fpath = pathlib.Path(fpath)
    fpath = fpath.relative_to(pathlib.Path.cwd(), walk_up=True) if fpath else None

    if fpath and fpath.exists():
        return str(fpath)
    else:
        raise FileNotFoundError(
            f"Environment file '{name}' not found in the current directory or its parents."
        )


def parse_jinja2_placeholders(raw_file, **kwargs):
    """
    Parse Jinja2 placeholders in a raw string and return a dictionary of the placeholders.

    Parameters
    ----------
    raw_file : str
        The raw string containing Jinja2 placeholders.

    Returns
    -------
    dict
        A dictionary with the parsed placeholders.
    """
    from jinja2 import Environment, StrictUndefined

    env = Environment(undefined=StrictUndefined)

    template = env.from_string(raw_file)
    kwargs = (
        get_dotenv_variables() | kwargs
    )  # Merge dotenv variables with provided kwargs

    parsed = template.render(**kwargs)

    return parsed
