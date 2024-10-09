import ipywidgets as widgets

def mode_select(start=0):
    opts=['SED: 2.0-5.2', 'K: 1.95-2.45','L: 2.9-4.15','M: 4.5-5.2', 'H2O: 2.0-3.7','CH4: 3.1-3.5']
    scalesmode=widgets.Dropdown(
        options=opts,
        value=opts[start],
        description='Mode:',
        disabled=False,
    )
    return scalesmode


def gs_select_slide():
    guidestar = widgets.FloatSlider(
    value=5.0,
    min=0,
    max=15.0,
    step=0.1,
    description='H mag:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.2f',
    )
    return guidestar

def gs_select_freeform():
    guidestar = widgets.FloatText(
        value=5.0,
        description='H mag:',
        disabled=False
    )
    return guidestar


def gs_select(start=0):
    opts=[5, 12, 13, 14]
    guidestar=widgets.Dropdown(
        options=opts,
        value=opts[start],
        description='GS Mag:',
        disabled=False,
    )
    return guidestar