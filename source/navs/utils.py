
def p_heading(h: float, h_max=360):
    return h + h*(abs(h/h_max))