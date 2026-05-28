from jinja2 import Environment, FileSystemLoader, select_autoescape
from fastapi.responses import HTMLResponse
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))
env.cache = None


def render(name: str, context: dict) -> HTMLResponse:
    template = env.get_template(str(name))
    return HTMLResponse(template.render(context))

# Provide the same API the routers expect
class Templates:
    env = env

    def get_template(self, name: str):
        return self.env.get_template(str(name))

    def TemplateResponse(self, name: str, context: dict):
        return render(name, context)


templates = Templates()
