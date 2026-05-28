from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    auto_reload=True,
)
env.cache = None
env.bytecode_cache = None

templates = Jinja2Templates(env=env)
