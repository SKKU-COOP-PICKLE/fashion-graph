# Polyvore

The polyvore data can be downloaded running the `get_polyvore.sh` in `data/polyvore/`. Then run `./process_polyvore.sh`.

# Custom

## Preparation

Similar to polyvore, your custom fashion data must be below format (json).
Also, path of data should be `train.json`, `valid.json`, and `test.json`.
All these data should be in `custom/jsons/`.
```
[
    { // fashion set
        "items": [ // fashion items
            {
                "id": "..."
            }, 
            {
                "id": "...",
            }, 
            ...
        ]
    },
    ...
]
```

The images should be in `data/custom/images/`, and the name of image should be `{id}.jpg`. If you want to load images another way, check `extract_features.py`

## Create dataset

Run `process_custom.sh`
