
import yaml
def yaml_load(file_name, item_name):
    with open(file_name) as file:
        res = yaml.full_load(file)
        return res[item_name]

