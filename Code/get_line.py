import inspect

def get_line_number():
    """
    Get the line number of the current cell in the Jupyter Notebook.
    """
    # Get the frame of the current function
    frame = inspect.currentframe()
    
    # Get the frame of the caller (the Jupyter Notebook cell)
    while frame:
        if frame.f_code.co_filename.endswith('input'):
            return frame.f_lineno
        frame = frame.f_back
        
    # If the cell is not found, return -1
    return -1

print(f"This code is on line {get_line_number()} of the Jupyter Notebook.")