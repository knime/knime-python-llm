import yaml

def render_structured(**kwargs) -> str:
    yaml_str = yaml.dump(kwargs, sort_keys=False)
    return f"```yaml\n{yaml_str}```"