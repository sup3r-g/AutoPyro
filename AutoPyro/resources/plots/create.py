TYPES = {
    "linear": ["a", "b"],
    "sigmoid": ["Q", "A", "B", "K", "M"],
    "polynomial": ["a", "b", "c", "d"],  # fix
    "exponent": ["a", "b", "c"],
    "power": ["a", "b", "c"],
}

chart_dict = {
    "name": "HI-Tmax",
    "title": "HI-Tmax",  # "label"
    "settings": {
        "fig_dims": [100, 100],
        "xlim": [400, 500],
        "ylim": [0, 1000],
        "major_ticks": 10,
        "minor_ticks": 5,
        "log": False,
        "legend": True,
        "grid": True
    },
    "annotations": [
        {"label": "Zone 1", "pos": {"x": 200, "y": 500}, "angle": 45},
        {"label": "Zone 2", "pos": {"x": 200, "y": 500}, "angle": 45},
    ],
    "data": {
        "lines": [  # Lines approximated by A line type with params
            {
                "label": "I",
                "color": "green",
                "marker": "+",
                "type": "sigmoid",
                "params": [],
            },
            {
                "label": "I",
                "color": "red",
                "marker": "*",
                "type": "sigmoid",
                "params": [],
            },
        ],
        "points": [  # Independent points not approximated by any line type
            {
                "label": "Один",
                "color": "green",
                "marker": "+",
                "pos": {"x": 100, "y": 100},
            },
            {
                "label": "Два",
                "color": "green",
                "marker": "+",
                "pos": {"x": 200, "y": 500},
            },
        ],
    },
}
