import random

flat = {
    "name": "flat",
    "params": ["amp1", "amp2", "mod16", "mod66", "mod11", "mod23", "mod34", "mod45", "ratio1", "ratio2", "ratio3", "ratio4", "ratio5", "ratio6", "detune1", "detune2", "detune3", "detune4", "detune5", "detune6", "feedback", "lfofreq", "lfodepth"],
    "values_range": {
        "mod": [float(e)/1000 for e in range(0, 5000, 100)],
        "ratio": [float(e)/1000 for e in range(0, 200, 2)],
        "feedback": [float(e)/100 for e in range(100, 202, 2)],
        "lfofreq": [float(e)/100 for e in range(0, 100, 2)],
    },
    "constants": {
        "amp1": 1,
        "amp2": 1,
        "lfodepth": 0,
        "detune1": 0, 
        "detune2": 0, 
        "detune3": 0, 
        "detune4": 0, 
        "detune5": 0,
        "detune6": 0,
    },
    "PATCHES_IN_COMPOSITION": 4800,
    "PATCHES_IN_TOTAL": 9000,
    "SAMPLE_LENGTH": 0.5,
}

def sample_params(context):
    return {
        param: sampling_strategy(context, param) if get_value_range(context, param) else context["constants"][param] for param in context["params"]
    }

def sampling_strategy(context, param):
    value_range = get_value_range(context, param)
    param_value = random.choice(value_range)
    return param_value

def get_value_range(context, param):
    for k, v in context["values_range"].items():
        if k in param:
            return v
