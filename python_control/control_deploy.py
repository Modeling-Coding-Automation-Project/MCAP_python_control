"""
File: control_deploy.py

This module provides the ControlDeploy class,
which offers utility methods for data type validation and writing code to files.
It is designed to support deployment processes where ensuring correct data types
and exporting code are necessary steps.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
import inspect
import ast
import astor


class IntegerPowerReplacer(ast.NodeTransformer):
    """
    A custom AST NodeTransformer that replaces integer power operations
      with repeated multiplications.

    This class traverses the abstract syntax tree (AST) of Python code
      and transforms expressions of the form
    `a ** n` (where `n` is a positive integer constant) into equivalent
      repeated multiplication expressions,
    i.e., `a * a * ... * a` (n times). This can be useful for code generation
      or translation to languages that
    do not support the power operator.
    """

    def visit_BinOp(self, node):
        """
        Visits binary operation nodes in the AST.
        If the operation is a power operation with a positive integer exponent,
          it transforms it into repeated multiplication.
        Args:
            node (ast.BinOp): The binary operation node to visit.
        Returns:
            ast.AST: The transformed node, or the original node
              if no transformation is applied.
        """
        # First visit children so nested power operations get transformed
        self.generic_visit(node)

        # Helper to extract integer exponent value from node.right
        def _get_int_exponent(rhs):
            # Direct constant (Python 3.8+)
            if isinstance(rhs, ast.Constant) and isinstance(rhs.value, int):
                return rhs.value
            # Older style numeric literal
            if isinstance(rhs, ast.Num) and isinstance(rhs.n, int):
                return rhs.n
            # Unary minus literal like `-2` is parsed as UnaryOp(USub, Constant(2))
            if isinstance(rhs, ast.UnaryOp) and isinstance(rhs.op, ast.USub):
                operand = rhs.operand
                if isinstance(operand, ast.Constant) and isinstance(operand.value, int):
                    return -operand.value
                if isinstance(operand, ast.Num) and isinstance(operand.n, int):
                    return -operand.n
            return None

        exp_val = None
        if isinstance(node.op, ast.Pow):
            exp_val = _get_int_exponent(node.right)

        # Only transform when exponent is an integer constant
        if exp_val is None:
            return node

        n = exp_val
        # Handle special case n == 0 -> return 1
        if n == 0:
            return ast.copy_location(ast.Constant(value=1), node)

        # Positive exponent: repeated multiplications a * a * ...
        if n > 0:
            result = node.left
            for _ in range(n - 1):
                result = ast.BinOp(left=result, op=ast.Mult(), right=node.left)
            return result

        # Negative exponent: produce repeated divisions: 1 / a / a / ...
        abs_n = abs(n)
        result = ast.Constant(value=1)
        for _ in range(abs_n):
            result = ast.BinOp(left=result, op=ast.Div(), right=node.left)
        return result

    def transform_code(self, source_code):
        """
        Transforms the given Python source code by replacing
          integer power operations with an alternative implementation.

        Args:
            source_code (str): The Python source code to be transformed.

        Returns:
            str: The transformed Python source code
              with integer power operations replaced.

        Raises:
            SyntaxError: If the provided source code cannot be parsed.
        """
        tree = ast.parse(source_code)
        transformer = IntegerPowerReplacer()
        transformed_tree = transformer.visit(tree)

        # Ensure location information is present for codegen
        ast.fix_missing_locations(transformed_tree)

        transformed_code = astor.to_source(transformed_tree)

        if transformed_code.endswith("\n"):
            transformed_code = transformed_code[:-1]

        return transformed_code


class FunctionExtractor(ast.NodeVisitor):
    """
    A class to extract function definitions from a Python file and store them in a dictionary.
    This class reads the source code from a specified file, parses it into an abstract syntax tree (AST),
    and visits each function definition node to extract the function name and its source code.
    Attributes:
        file_path (str): The path to the Python file from which to extract function definitions.
        source_code (str): The source code of the Python file.
        functions (dict): A dictionary to store function names as keys and their source code as values.
    """

    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
        self.source_code = source_code

        self.functions = {}

    def visit_FunctionDef(self, node):
        """
        Visits function definition nodes in the AST and extracts the function name and source code.
        Args:
            node (ast.FunctionDef): The function definition node to visit.
        This method retrieves the function name and its source code segment from the original source code,
        and stores them in the `functions` dictionary.
        """
        function_name = node.name
        function_code = ast.get_source_segment(self.source_code, node)
        self.functions[function_name] = function_code
        self.generic_visit(node)

    def extract(self):
        """
        Parses the source code and extracts all function definitions.
        This method creates an AST from the source code, visits each function definition node,
        and populates the `functions` dictionary with function names and their corresponding source code.
        Returns:
            dict: A dictionary containing function names as keys and their source code as values.
        """
        tree = ast.parse(self.source_code)
        self.visit(tree)

        return self.functions


class InputSizeVisitor(ast.NodeVisitor):
    """
    A class to visit AST nodes and extract the value of the INPUT_SIZE variable.
    This class traverses the abstract syntax tree (AST) of Python code to find assignments to the INPUT_SIZE variable
    and stores its value in the `input_size` attribute. It is used to determine the input size for Kalman filter functions.
    Attributes:
        input_size (int or None): The value of the INPUT_SIZE variable if found, otherwise None.
    """

    def __init__(self):
        self.input_size = None

    def visit_Assign(self, node):
        """
        Visits assignment nodes in the AST and checks for assignments to the INPUT_SIZE variable.
        Args:
            node (ast.Assign): The assignment node to visit.
        If the assignment targets a variable named INPUT_SIZE, it checks the value assigned to it.
        If the value is a constant or a numeric literal, it stores that value in the `input_size` attribute.
        If the value is a constant, it retrieves the value directly; if it is a numeric literal (for Python < 3.8),
        it retrieves the numeric value using the `.n` attribute.
        """

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'INPUT_SIZE':
                if isinstance(node.value, ast.Constant):
                    self.input_size = node.value.value
                elif isinstance(node.value, ast.Num):  # For Python < 3.8
                    self.input_size = node.value.n


class ExpressionDeploy:
    @staticmethod
    def create_sympy_code(sym_object: sp.Matrix):
        """
        Creates a string representation of a symbolic function in Python, which can be used to generate C++ code.
        Args:
            sym_object (sympy.Matrix): A sympy Matrix object representing the symbolic function.
        Returns:
            tuple: A tuple containing the generated code as a string and the argument text for the function.
        This method generates a Python function definition that takes symbolic variables as arguments and returns
        a NumPy array representing the result of the symbolic computation.
        It constructs the function signature, including the argument types based on the dtype of the symbolic matrix.
        It also prepares the calculation code by converting the symbolic matrix to a list format.
        The generated code is suitable for deployment in a C++ environment, allowing for efficient execution of
        symbolic computations.
        """
        if not isinstance(sym_object, sp.MatrixBase):
            raise ValueError(
                "The input must be a sympy.Matrix object.")

        value_example = np.array([[1.0]])
        value_type = str(value_example.dtype.name)

        code_text = ""
        code_text += "def sympy_function("

        arguments_text = ""
        arguments_text_out = ""

        replacements_list, reduced_expression = sp.cse(sym_object)

        sym_symbols = sym_object.free_symbols
        for i, symbol in enumerate(sym_symbols):
            arguments_text += f"{symbol}"
            arguments_text_out += f"{symbol}"
            if i == len(sym_symbols) - 1:
                arguments_text += f": np.{value_type}"
                break
            else:
                arguments_text += f": np.{value_type}" + ", "
                arguments_text_out += ", "

        code_text += arguments_text + ")"

        code_text += f" -> Tuple[{sym_object.shape[0]}, {sym_object.shape[1]}]:\n\n"

        if len(replacements_list) > 0:
            for i, value in enumerate(replacements_list):
                code_text += f"    {value[0]} = {value[1]}\n\n"

            calculation_code = f"{reduced_expression[0].tolist()}"
        else:
            calculation_code = f"{sym_object.tolist()}"

        code_text += f"    return np.array({calculation_code})\n\n\n"

        return code_text, arguments_text_out

    @staticmethod
    def create_interface_code(sym_object, arguments_text, X, U=None):
        """
        Creates a Python function interface for the symbolic function, which can be used to generate C++ code.
        Args:
            sym_object (sympy.Matrix): A sympy Matrix object representing the symbolic function.
            arguments_text (str): A string containing the argument names for the function.
            X (np.ndarray): A NumPy array representing the state variables.
            U (np.ndarray, optional): A NumPy array representing the input variables. Defaults to None.
        Returns:
            str: A string containing the generated Python function interface code.
        This method generates a Python function definition that serves as an interface for the symbolic function.
        It constructs the function signature, including the argument types based on the dtype of the symbolic matrix.
        It also prepares the function body by assigning the input variables to the symbolic variables and
        initializing the parameters from the symbolic matrix.
        The generated code is suitable for deployment in a C++ environment, allowing for efficient execution of
        symbolic computations.
        """
        sym_symbols = sym_object.free_symbols

        code_text = ""
        code_text += "def function(X: X_Type"

        if U is not None:
            code_text += ", U: U_Type, Parameters: Parameter_Type = None)"
        else:
            code_text += ", Parameters: Parameter_Type = None)"

        code_text += f" -> Tuple[{sym_object.shape[0]}, {sym_object.shape[1]}]:\n\n"

        for i in range(X.shape[0]):
            if X[i] in sym_symbols:
                code_text += f"    {X[i]} = X[{i}, 0]\n"
                sym_symbols.remove(X[i])

        if U is not None:
            for i in range(U.shape[0]):
                if U[i] in sym_symbols:
                    code_text += f"    {U[i]} = U[{i}, 0]\n"
                    sym_symbols.remove(U[i])

        code_text += "\n"

        for symbol in sym_symbols:
            code_text += f"    {symbol} = "
            code_text += f"Parameters.{symbol}\n"

        code_text += "\n"

        code_text += "    return sympy_function("
        code_text += arguments_text
        code_text += ")\n"

        return code_text

    @staticmethod
    def write_function_code_from_sympy(sym_object, sym_object_name, X, U=None):
        """
        Writes the generated C++ code for a Kalman filter function based on a symbolic representation.
        Args:
            sym_object (sympy.Matrix): A sympy Matrix object representing the symbolic function.
            sym_object_name (str): The name of the symbolic function.
            X (np.ndarray): A NumPy array representing the state variables.
            U (np.ndarray, optional): A NumPy array representing the input variables. Defaults to None.
        This method generates a C++ header file containing the symbolic function code, including the necessary imports,
        class definitions, and constants for state and input sizes. It also creates the function code that implements
        the symbolic function using the provided symbolic matrix. The generated code is written to a file named
        `<sym_object_name>.py`, which can be used for deployment in a C++ environment.
        """

        header_code = ""
        header_code += "import numpy as np\n"
        header_code += "from math import *\n"
        header_code += "from typing import Tuple\n\n\n"

        header_code += "class X_Type:\n    pass\n\n\n"

        header_code += "class U_Type:\n    pass\n\n\n"

        header_code += "class Parameter_Type:\n    pass\n\n\n"

        header_code += "STATE_SIZE = " + str(X.shape[0]) + "\n"
        if U is not None:
            header_code += "INPUT_SIZE = " + str(U.shape[0])
        else:
            header_code += "INPUT_SIZE = 0"
        header_code += "\n\n\n"

        sympy_function_code, arguments_text = ExpressionDeploy.create_sympy_code(
            sym_object)

        interface_code = ExpressionDeploy.create_interface_code(
            sym_object, arguments_text, X, U)

        total_code = header_code + sympy_function_code + interface_code

        ControlDeploy.write_to_file(
            total_code, f"{sym_object_name}.py")

    @staticmethod
    def write_state_function_code_from_sympy(
            sym_object, X, U=None, file_name: str = None):
        """
        Writes the generated C++ code for a Kalman filter state function based on a symbolic representation.
        Args:
            sym_object (sympy.Matrix): A sympy Matrix object representing the symbolic function.
            X (np.ndarray): A NumPy array representing the state variables.
            U (np.ndarray, optional): A NumPy array representing the input variables. Defaults to None.
        This method generates a C++ header file containing the symbolic function code, including the necessary imports,
        class definitions, and constants for state and input sizes. It also creates the function code that implements
        the symbolic function using the provided symbolic matrix. The generated code is written to a file named
        `<sym_object_name>.py`, which can be used for deployment in a C++ environment.
        """

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = os.path.splitext(file_name)[0]

        function_code_file_name_without_ext = caller_file_name_without_ext + \
            "_" + sym_object_name

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object, function_code_file_name_without_ext, X, U)

        return function_code_file_name_without_ext

    @staticmethod
    def write_measurement_function_code_from_sympy(
            sym_object, X, file_name: str = None):
        """
        Writes the generated C++ code for a Kalman filter measurement function based on a symbolic representation.
        Args:
            sym_object (sympy.Matrix): A sympy Matrix object representing the symbolic function.
            X (np.ndarray): A NumPy array representing the state variables.
        This method generates a C++ header file containing the symbolic function code, including the necessary imports,
        class definitions, and constants for state and input sizes. It also creates the function code that implements
        the symbolic function using the provided symbolic matrix. The generated code is written to a file named
        `<sym_object_name>.py`, which can be used for deployment in a C++ environment.
        """

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = os.path.splitext(file_name)[0]

        function_code_file_name_without_ext = caller_file_name_without_ext + \
            "_" + sym_object_name

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object, function_code_file_name_without_ext, X, U=None)

        return function_code_file_name_without_ext

    @staticmethod
    def get_input_size_from_function_code(file_path):
        """
        Extracts the input size from a Python file containing function definitions.
        Args:
            file_path (str): The path to the Python file from which to extract the input size.
        Returns:
            int: The input size extracted from the file, or None if not found.
        This method reads the specified Python file, parses it into an abstract syntax tree (AST),
        and uses a custom AST visitor to find the assignment to the INPUT_SIZE variable.
        It returns the value of INPUT_SIZE if found, or None if not found.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            SyntaxError: If the file content cannot be parsed as valid Python code.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        tree = ast.parse(file_content)
        visitor = InputSizeVisitor()
        visitor.visit(tree)

        return visitor.input_size


class ControlDeploy:
    def __init__(self):
        pass

    @staticmethod
    def restrict_data_type(type_name):
        """
        Restricts the allowed data types to 'float32' or 'float64'.

        Args:
            type_name (str): The name of the data type to check.

        Raises:
            ValueError: If the provided data type is not 'float32' or 'float64'.
        """
        flag = False
        if type_name == 'float64':
            flag = True
        elif type_name == 'float32':
            flag = True
        else:
            flag = False

        if not flag:
            raise ValueError(
                "Data type not supported. Please use float32 or float64")

    @staticmethod
    def write_to_file(code_text, code_file_name_ext):
        """
        Writes the provided code text to a file with the specified name and extension.
        Args:
            code_text (str): The code text to write to the file.
            code_file_name_ext (str): The name of the file including its extension.
        Returns:
            str: The full path to the written file.
        Raises:
            ValueError: If the code text is empty or the file name is invalid.
        """
        with open(code_file_name_ext, "w", encoding="utf-8") as f:
            f.write(code_text)

        return code_file_name_ext

    @staticmethod
    def find_file(filename, search_path):
        """
        Searches for a file with the specified name in the given search path.
        Args:
            filename (str): The name of the file to search for.
            search_path (str): The path to search for the file.
        Returns:
            str or None: The full path to the file if found, otherwise None.
        This method traverses the directory structure starting from the specified search path,
        looking for a file with the specified name. If the file is found, it returns the full path to the file.
        If the file is not found, it returns None.
        Raises:
            FileNotFoundError: If the specified search path does not exist.
        """
        for root, _, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None
