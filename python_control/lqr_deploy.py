"""
File: lqr_deploy.py

This module provides utilities for generating C++ header files that implement LQR (Linear Quadratic Regulator) and LQI (Linear Quadratic Integrator) controllers from Python-based system matrices and weighting matrices. The generated C++ code is intended for use with a compatible C++ control library, enabling seamless deployment of control algorithms designed in Python to C++ environments.
"""
import os
import sys
sys.path.append(os.getcwd())

import copy
import numpy as np
import control
import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class LQR_Deploy:
    """
    A class for generating C++ code for Linear Quadratic Regulator (LQR) controllers.
    This class provides a static method to generate C++ header files that define the LQR controller
    based on the provided state-space matrices and weighting matrices.
    Attributes:
        None
    Methods:
        generate_LQR_cpp_code(Ac, Bc, Q, R, file_name=None):
            Generates C++ code for an LQR controller using the provided state-space matrices and weighting matrices.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_LQR_cpp_code(Ac, Bc, Q, R, file_name=None):
        """
        Generates C++ code for an LQR controller using the provided state-space matrices and weighting matrices.
        Args:
            Ac (np.ndarray): State matrix of the system.
            Bc (np.ndarray): Input matrix of the system.
            Q (np.ndarray): State weighting matrix for the LQR controller.
            R (np.ndarray): Input weighting matrix for the LQR controller.
            file_name (str, optional): The base name for the generated C++ header file. If None, the caller's file name is used.
        Returns:
            list: A list of file names for the generated C++ header files.
        Raises:
            ValueError: If the input matrices are not compatible or if the data type is unsupported.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(Ac.dtype.name)

        type_name = NumpyDeploy.check_dtype(Ac)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% code generation
        variable_name = "LQR"

        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create Ac, Bc matrices
        exec(f"{variable_name}_Ac = copy.deepcopy(Ac)")
        Ac_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Ac, caller_file_name_without_ext)")
        exec(f"{variable_name}_Bc = copy.deepcopy(Bc)")
        Bc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Bc, caller_file_name_without_ext)")

        deployed_file_names.append(Ac_file_name)
        deployed_file_names.append(Bc_file_name)

        Ac_file_name_no_extension = Ac_file_name.split(".")[0]
        Bc_file_name_no_extension = Bc_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{Ac_file_name}\"\n"
        code_text += f"#include \"{Bc_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"using Ac_Type = {Ac_file_name_no_extension}::type;\n\n"

        code_text += f"using Bc_Type = {Bc_file_name_no_extension}::type;\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = Ac_Type::COLS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = Bc_Type::ROWS;\n\n"

        code_text += f"using Q_Type = DiagMatrix_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using R_Type = DiagMatrix_Type<{type_name}, INPUT_SIZE>;\n\n"

        code_text += "using type = LQR_Type<" + \
            "Ac_Type, Bc_Type, Q_Type, R_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto Ac = {Ac_file_name_no_extension}::make();\n\n"

        code_text += f"  auto Bc = {Bc_file_name_no_extension}::make();\n\n"

        code_text += "  auto Q = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(Q[i, i]) + ")"
            if i == Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto R = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(R[i, i]) + ")"
            if i == R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  return make_LQR(Ac, Bc, Q, R);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names


class LQI_Deploy:
    """
    A class for generating C++ code for Linear Quadratic Integrator (LQI) controllers.
    This class provides a static method to generate C++ header files that define the LQI controller
    based on the provided state-space matrices and weighting matrices.
    Attributes:
        None
    Methods:
        generate_LQI_cpp_code(Ac, Bc, Cc, Q_ex, R_ex, file_name=None):
            Generates C++ code for an LQI controller using the provided state-space matrices and weighting matrices.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_LQI_cpp_code(Ac, Bc, Cc, Q_ex, R_ex, file_name=None):
        """
        Generates C++ code for an LQI controller using the provided state-space matrices and weighting matrices.
        Args:
            Ac (np.ndarray): State matrix of the system.
            Bc (np.ndarray): Input matrix of the system.
            Cc (np.ndarray): Output matrix of the system.
            Q_ex (np.ndarray): Extended state weighting matrix for the LQI controller.
            R_ex (np.ndarray): Input weighting matrix for the LQI controller.
            file_name (str, optional): The base name for the generated C++ header file. If None, the caller's file name is used.
        Returns:
            list: A list of file names for the generated C++ header files.
        Raises:
            ValueError: If the input matrices are not compatible or if the data type is unsupported.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(Ac.dtype.name)

        type_name = NumpyDeploy.check_dtype(Ac)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% code generation
        variable_name = "LQI"

        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create Ac, Bc matrices
        exec(f"{variable_name}_Ac = copy.deepcopy(Ac)")
        Ac_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Ac, caller_file_name_without_ext)")
        exec(f"{variable_name}_Bc = copy.deepcopy(Bc)")
        Bc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Bc, caller_file_name_without_ext)")
        exec(f"{variable_name}_Cc = copy.deepcopy(Cc)")
        Cc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Cc, caller_file_name_without_ext)")

        deployed_file_names.append(Ac_file_name)
        deployed_file_names.append(Bc_file_name)
        deployed_file_names.append(Cc_file_name)

        Ac_file_name_no_extension = Ac_file_name.split(".")[0]
        Bc_file_name_no_extension = Bc_file_name.split(".")[0]
        Cc_file_name_no_extension = Cc_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{Ac_file_name}\"\n"
        code_text += f"#include \"{Bc_file_name}\"\n"
        code_text += f"#include \"{Cc_file_name}\"\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"auto Ac = {Ac_file_name_no_extension}::make();\n\n"

        code_text += f"auto Bc = {Bc_file_name_no_extension}::make();\n\n"

        code_text += f"auto Cc = {Cc_file_name_no_extension}::make();\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = decltype(Ac)::COLS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = decltype(Bc)::ROWS;\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = decltype(Cc)::COLS;\n"
        code_text += "constexpr std::size_t Q_EX_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += "auto Q_ex = make_DiagMatrix<Q_EX_SIZE>(\n"
        for i in range(Q_ex.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(Q_ex[i, i]) + ")"
            if i == Q_ex.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "auto R_ex = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(R_ex.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(R_ex[i, i]) + ")"
            if i == R_ex.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "using type = LQI_Type<" + \
            "decltype(Ac), decltype(Bc), decltype(Cc), decltype(Q_ex), decltype(R_ex)>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += "    return make_LQI(Ac, Bc, Cc, Q_ex, R_ex);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
