from django import template

register = template.Library()

@register.filter
def dict_get(dictionary, key):
    """
    Get a value from a dictionary using a key.
    Usage: {{ my_dict|dict_get:"my_key" }}
    """
    if not dictionary:
        return None
    return dictionary.get(key)
