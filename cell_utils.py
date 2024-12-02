import nbformat
from nbconvert import PythonExporter
import io
import contextlib
import traceback
from textwrap import wrap

import re
import ast

import matplotlib

# Set to store registered local filenames
REGISTERED_FILES = set()

# def read_normalized_notebook(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         nb = nbformat.read(f, as_version=4)
#     nbformat.validate(nb)  # Optional: validate the notebook first
#     nb = nbformat.normalize(nb)  # Normalize the notebook

def register_local_file(filename):
    """
    Registers a filename to replace external paths with local paths during cell execution.
    :param filename: The name of the file to register.
    """
    REGISTERED_FILES.add(filename)


def has_string_in_cell(notebook_path, cell_index, search_string, case_sensitive=False):
    """
    Checks if a specific string or sentence exists within the specified cell of a Jupyter notebook.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param cell_index: Index of the cell to check.
    :param search_string: The string to search for in the cell.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: Boolean indicating if the string is found in the specified cell.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Ensure the cell index is within the range of existing cells
    if cell_index >= len(nb.cells):
        raise IndexError("The provided cell index is out of range for this notebook.")
    
    cell = nb.cells[cell_index]
    cell_content = cell.source
    
    # Adjust string matching based on case sensitivity
    if not case_sensitive:
        cell_content = cell_content.lower()
        search_string = search_string.lower()
    
    return search_string in cell_content

def _replace_paths_with_local(code):
    """
    Replaces paths containing registered filenames with local paths (./filename).
    :param code: The original code from the notebook cell.
    :return: Modified code with paths replaced.
    """
    for filename in REGISTERED_FILES:
        # Regex to find occurrences of paths ending with the registered filename
        pattern = rf"[\"'].*?{re.escape(filename)}[\"']"
        code = re.sub(pattern, f'"./{filename}"', code)
    return code


def _safe_exec(code, namespace=None):
    """
    A wrapper around exec to replace paths and execute Python code safely.
    :param code: Python code to execute.
    :param namespace: The namespace to execute the code in (optional).
    """
    if namespace is None:
        namespace = {}
    
    # Replace paths with local paths before execution
    code = _replace_paths_with_local(code)

    # Execute the modified code
    try:
        exec(code, namespace)
    except Exception as e:
        print(f"Error executing the code: {traceback.format_exc()}")

def has_string_in_code_cells(notebook_path, start_index, end_index, search_string, case_sensitive=False):
    """
    Checks if a specific string or sentence exists within code cells in a specified index range of a Jupyter notebook.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to check (inclusive).
    :param end_index: Ending index of the cell range to check (exclusive).
    :param search_string: The string to search for in the cells.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: Boolean indicating if the string is found within any of the specified code cells.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Adjust string matching based on case sensitivity
    if not case_sensitive:
        search_string = search_string.lower()
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        if cell.cell_type == 'code':  # Only check code cells
            cell_content = cell.source
            if not case_sensitive:
                cell_content = cell_content.lower()

            if search_string in cell_content:
                return True
    
    return False

def print_text_and_output_cells(notebook_path, start_index, end_index, line_width=80):
    """
    Prints the content of text (markdown) and output cells in a specified index range of a Jupyter notebook,
    wrapping text lines to a specified width for better readability.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to print from (inclusive).
    :param end_index: Ending index of the cell range to print to (exclusive).
    :param line_width: Maximum width of the text lines, default is 80 characters.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        
        if cell.cell_type == 'markdown':
            print(f"Markdown Cell {cell_index}:")
            wrapped_text = "\n".join(wrap(cell.source, width=line_width))
            print(wrapped_text)
            print("-" * 40)  # Separator for readability
        
        elif cell.cell_type == 'code':
            print(f"Code Cell {cell_index}:")
            for output in cell.outputs:
                if output.output_type == 'stream':
                    wrapped_text = "\n".join(wrap(output.text, width=line_width))
                    print(wrapped_text)
                elif output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        wrapped_text = "\n".join(wrap(output.data['text/plain'], width=line_width))
                        print(wrapped_text)
                    if 'image/png' in output.data:
                        print("<Image output not shown>")
                elif output.output_type == 'error':
                    wrapped_text = "\n".join(wrap(f"Error: {output.ename}, {output.evalue}", width=line_width))
                    print(wrapped_text)
            print("-" * 40)  # Separator for readability

def print_code_and_output_cells(notebook_path, start_index, end_index, line_width=80):
    """
    Prints the content of code and output cells in a specified index range of a Jupyter notebook,
    wrapping text lines to a specified width for better readability.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to print from (inclusive).
    :param end_index: Ending index of the cell range to print to (exclusive).
    :param line_width: Maximum width of the text lines, default is 80 characters.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        
        if cell.cell_type == 'code':
            print(f"Code Cell {cell_index}:")
            for output in cell.outputs:
                if output.output_type == 'stream':
                    wrapped_text = "\n".join(wrap(output.text, width=line_width))
                    print(wrapped_text)
                elif output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        wrapped_text = "\n".join(wrap(output.data['text/plain'], width=line_width))
                        print(wrapped_text)
                    if 'image/png' in output.data:
                        print("<Image output not shown>")
                elif output.output_type == 'error':
                    wrapped_text = "\n".join(wrap(f"Error: {output.ename}, {output.evalue}", width=line_width))
                    print(wrapped_text)
            print("-" * 40)  # Separator for readability


def find_cells_by_indices(notebook_path, indices):
    """
    Finds cells by their indices in a Jupyter notebook and returns their details.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param indices: List of integers representing the indices of cells to find.
    :return: List of dictionaries, each containing the index, cell_id, cell_type, and content of the specified cells.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found_cells = []
    
    for index in indices:
        if index < len(nb.cells):
            cell = nb.cells[index]
            cell_info = {
                "index": index,
                "cell_id": cell.get("id", "N/A"),  # Handle the possibility of missing 'id' field
                "cell_type": cell.cell_type,
                "content": cell.source  # Include the content of the cell
            }
            found_cells.append(cell_info)
        else:
            print(f"Warning: No cell at index {index}. Skipping...")
    
    return found_cells

def find_cells_with_text(notebook_path, search_text, case_sensitive=False):
    """
    Searches for cells containing specified text in a Jupyter notebook and returns their contents.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param search_text: Text to search for within the notebook cells.
    :param case_sensitive: Boolean to indicate if the search should be case sensitive.
    :return: List of dictionaries, each containing the index, cell_id, cell_type, and content of cells where the text is found.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    found_cells = []
    
    # Normalize text based on case sensitivity
    if not case_sensitive:
        search_text = search_text.lower()
    
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' or cell.cell_type == 'markdown':
            # Retrieve cell content and normalize if case insensitive
            cell_content = cell.source
            if not case_sensitive:
                cell_content = cell_content.lower()
            
            # Check if search_text is in cell_content
            if search_text in cell_content:
                cell_info = {
                    "index": index,
                    "cell_id": cell.get("id", "N/A"),  # Some cells might not have an 'id' field
                    "cell_type": cell.cell_type,
                    "content": cell.source  # Return the original (non-normalized) content
                }
                found_cells.append(cell_info)
    
    return found_cells

def execute_cell(python_code, namespace):
    """Execute Python code safely and capture stdout."""
    try:
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            _safe_exec(python_code, namespace)
    except Exception as e:
        print(f"Error executing the code: {traceback.format_exc()}")

def extract_variables(notebook_path, cell_idx=-1):
    """
    Extracts variables from the notebook up to the specified cell index.
    Tracks variable renaming, so renamed variables are included.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    namespace = {}

    for index, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if cell_idx != -1 and index > cell_idx:
                break
            python_code = exporter.from_notebook_node(nbformat.v4.new_notebook(cells=[cell]))[0]
                        
            # Execute the cell
            execute_cell(python_code, namespace)            
    
    return namespace

def extract_initial_variables(notebook_path):
    """Extracts all variables initially loaded in the notebook, tracking renamed variables."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    namespaces = {}
    cell_namespace = {}

    for cell in nb.cells:
        if cell.cell_type == 'code':
            python_code = exporter.from_notebook_node(nbformat.v4.new_notebook(cells=[cell]))[0]
                        
            # Execute the cell
            execute_cell(python_code, cell_namespace)
                        
            # Only add new variables to the main namespace
            for key, value in cell_namespace.items():
                if key not in namespaces:
                    namespaces[key] = value

    return namespaces

def extract_initial_variables(notebook_path):
    """Extracts all variables right after they are loaded in the notebook for the first time."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    exporter = PythonExporter()
    namespaces = {}
    cell_namespace = {}

    for cell in nb.cells:
        if cell.cell_type == 'code':
            python_code = exporter.from_notebook_node(nbformat.v4.new_notebook(cells=[cell]))[0]

            execute_cell(python_code, cell_namespace)
            
            # Update namespace with new variables found in this cell, only if they
            # were not previously set
            for key, _ in cell_namespace.items():
                if key not in namespaces:
                    namespaces[key] = cell_namespace[key]
    
    return namespaces





def search_plots_in_extracted_vars(cell_vars):
    """
    Extracts plot data from the figures stored in the 'plt' variable of cell_vars.
    Returns three separate lists containing line plot data, histogram (patch) data,
    and correlation matrix data from collections.

    :param cell_vars: A dictionary containing variables from executed cells, expected
                      to include 'plt' if plots were generated.
    :return: Three lists: line_lists, patch_lists, and collection_lists, or None if 'plt' is not in cell_vars.
    """
    # Check if 'plt' exists in cell_vars
    if "plt" not in cell_vars:
        return None, None, None  # Return None if no plots were generated

    # Get active figures from the 'plt' variable
    figures = [manager.canvas.figure for manager in cell_vars["plt"]._pylab_helpers.Gcf.get_all_fig_managers()]

    if not figures:
        return [], []  # Return empty lists if no figures are found

    # Initialize lists to store different types of data
    line_lists = []
    # patch_lists = []
    collection_lists = []

    # Iterate through figures and extract data
    for fig_index, fig in enumerate(figures):
        figure_title = fig._suptitle.get_text() if fig._suptitle else "No title"

        for ax_index, ax in enumerate(fig.axes):
            axis_title = ax.get_title()
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()

            # Extract line plot data
            for line in ax.lines:
                try:
                    x_data = line.get_xdata().tolist()
                    y_data = line.get_ydata().tolist()
                except:
                    x_data = list(line.get_xdata())
                    y_data = list(line.get_ydata())
                line_lists.append({
                    "fig_index": fig_index,
                    "ax_index": ax_index,
                    "figure_title": figure_title,
                    "axis_title": axis_title,
                    "x_label": x_label,
                    "y_label": y_label,
                    "data_points": list(zip(x_data, y_data))
                })


            # TODO: This part does not work - should be written again
            # # Extract histogram (patch) data
            # for patch in ax.patches:
            #     import ipdb
            #     ipdb.set_trace()
            #     x_left = patch.get_x()
            #     x_right = patch.get_x() + patch.get_width()
            #     y_height = patch.get_height()
            #     patch_lists.append({
            #         "fig_index": fig_index,
            #         "ax_index": ax_index,
            #         "figure_title": figure_title,
            #         "axis_title": axis_title,
            #         "x_label": x_label,
            #         "y_label": y_label,
            #         "bar_midpoint": (x_left + x_right) / 2,
            #         "bar_leftpoint": x_left, 
            #         "bar_rightpoint": x_right,
            #         "bar_height": y_height
            #     })

            # Extract correlation matrix (collection) data from QuadMesh in collections
            for collection in ax.collections:
                if isinstance(collection, matplotlib.collections.QuadMesh):
                    # Extract matrix data from QuadMesh array
                    matrix_data = collection.get_array().data

                    x_labels = [label.get_text() for label in ax.get_xticklabels()]
                    y_labels = [label.get_text() for label in ax.get_yticklabels()]
                    collection_lists.append({
                        "fig_index": fig_index,
                        "ax_index": ax_index,
                        "figure_title": figure_title,
                        "axis_title": axis_title,
                        "x_label": x_label,
                        "y_label": y_label,
                        "matrix_data": matrix_data,
                        "x_labels": x_labels,
                        "y_labels": y_labels
                    })

    return line_lists, collection_lists

def extract_cell_content_and_outputs(notebook_path, start_index, end_index, line_width=80):
    """
    Extracts the content of text (markdown) and output (from code) cells in a specified index range of a Jupyter notebook,
    wrapping text lines to a specified width for better readability.
    
    :param notebook_path: Path to the Jupyter notebook file.
    :param start_index: Starting index of the cell range to extract from (inclusive).
    :param end_index: Ending index of the cell range to extract to (exclusive).
    :param line_width: Maximum width of the text lines, default is 80 characters.
    :return: List of dictionaries, each containing the index, cell_type, and extracted content or outputs.
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    extracted_cells = []
    
    for cell_index in range(start_index, min(end_index, len(nb.cells))):
        cell = nb.cells[cell_index]
        cell_data = {
            "index": cell_index,
            "cell_type": cell.cell_type,
            "content": None,
            "outputs": None
        }
        
        if cell.cell_type == 'markdown':
            # Wrap and extract markdown cell content
            wrapped_text = "\n".join(wrap(cell.source, width=line_width))
            cell_data["content"] = wrapped_text  # Store the wrapped markdown content
        
        elif cell.cell_type == 'code':
            # Extract code cell content and outputs
            cell_data["content"] = cell.source  # Store the code source
            code_outputs = []
            for output in cell.outputs:
                if output.output_type == 'stream':
                    wrapped_output = "\n".join(wrap(output.text, width=line_width))
                    code_outputs.append(wrapped_output)
                elif output.output_type in ['execute_result', 'display_data']:
                    if 'text/plain' in output.data:
                        wrapped_output = "\n".join(wrap(output.data['text/plain'], width=line_width))
                        code_outputs.append(wrapped_output)
                    if 'image/png' in output.data:
                        code_outputs.append("<Image output not shown>")
                elif output.output_type == 'error':
                    wrapped_output = "\n".join(wrap(f"Error: {output.ename}, {output.evalue}", width=line_width))
                    code_outputs.append(wrapped_output)
            cell_data["outputs"] = code_outputs  # Store the outputs of the code cell
            
        extracted_cells.append(cell_data)
    
    return extracted_cells


def search_text_in_extracted_content(extracted_cells, search_string, case_sensitive=False):
    """
    Searches for a string within the content and outputs of extracted cells.
    
    :param extracted_cells: List of dictionaries with cell content and outputs (from extract_cell_content_and_outputs).
    :param search_string: The string to search for.
    :param case_sensitive: Boolean flag indicating if the search should be case-sensitive. Default is False.
    :return: Tuple containing:
        - A boolean flag indicating if the search string was found.
        - List of dictionaries with cell details where the search string was found.
    """
    found_cells = []
    found_flag = False  # Initialize flag as False
    
    # Normalize search string if case sensitivity is off
    if not case_sensitive:
        search_string = search_string.lower()
    
    for cell in extracted_cells:
        content = cell.get('content', "")
        outputs = cell.get('outputs', []) or []

        # Normalize content and outputs if case sensitivity is off
        if not case_sensitive:
            content = content.lower()
            outputs = [output.lower() for output in outputs]
        
        # Check if the search string is in the content or any output
        if search_string in content or any(search_string in output for output in outputs):
            found_flag = True  # Set the flag to True if the search string is found
            found_cells.append(cell)
    
    return found_flag, found_cells
