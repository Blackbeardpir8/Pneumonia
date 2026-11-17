
from django.templatetags.static import static
from django.urls import reverse
from jinja2 import Environment
from django.conf import settings

def media(path):
    if not path:
        return ""

    return f"{settings.MEDIA_URL}{str(path).lstrip('/')}"


def environment(**options):
    env = Environment(**options)

    env.globals.update({
        "static": static,
        "url": reverse,
        "media": media,
    })

    return env