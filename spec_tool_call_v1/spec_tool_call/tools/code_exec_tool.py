"""Code execution and generation tools."""
import sys
import io
import ast
import contextlib
import traceback
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in a restricted environment.
    
    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (not enforced in basic version)
        
    Returns:
        Dict with status, output, error, and result fields
    """
    # Create restricted global namespace
    restricted_globals = {
        '__builtins__': {
            # Allow safe built-ins
            'print': print,
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'list': list,
            'dict': dict,
            'set': set,
            'tuple': tuple,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'isinstance': isinstance,
            'type': type,
            'ord': ord,
            'chr': chr,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            # Allow imports for common libraries
            '__import__': __import__,
        },
        # Pre-import commonly used modules
        'math': __import__('math'),
        're': __import__('re'),
        'json': __import__('json'),
        'datetime': __import__('datetime'),
        'itertools': __import__('itertools'),
        'collections': __import__('collections'),
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        # Validate syntax first
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "status": "syntax_error",
                "error": f"Syntax error: {str(e)}",
                "output": "",
                "result": None
            }
        
        # Execute code with output capture
        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):
            
            local_vars = {}
            exec(code, restricted_globals, local_vars)
            
            # Try to get the result from the last expression or variable
            result = None
            if local_vars:
                # Get the last assigned variable or expression result
                result = local_vars.get('result', list(local_vars.values())[-1] if local_vars else None)
        
        stdout_text = stdout_capture.getvalue()
        stderr_text = stderr_capture.getvalue()
        
        return {
            "status": "success",
            "output": stdout_text,
            "error": stderr_text if stderr_text else None,
            "result": str(result) if result is not None else None
        }
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            "status": "runtime_error",
            "error": error_trace,
            "output": stdout_capture.getvalue(),
            "result": None
        }


def execute_calculation(expression: str) -> Dict[str, Any]:
    """
    Execute a mathematical calculation safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Dict with status and result
    """
    try:
        # Validate that it's a safe expression (no assignments, imports, etc.)
        tree = ast.parse(expression, mode='eval')
        
        # Check for unsafe operations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, 
                                ast.AugAssign, ast.Call)):
                if isinstance(node, ast.Call):
                    # Only allow specific safe functions
                    if not (isinstance(node.func, ast.Name) and 
                           node.func.id in ('abs', 'min', 'max', 'sum', 'round', 'pow')):
                        return {
                            "status": "error",
                            "error": "Only basic math operations are allowed in calculations",
                            "result": None
                        }
        
        # Create safe namespace with math operations
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'round': round,
            'pow': pow,
        }
        
        # Import math functions
        import math
        safe_dict.update({k: v for k, v in math.__dict__.items() if not k.startswith('_')})
        
        result = eval(expression, safe_dict, {})
        
        return {
            "status": "success",
            "result": result,
            "formatted": str(result)
        }
        
    except SyntaxError as e:
        return {
            "status": "syntax_error",
            "error": f"Invalid expression: {str(e)}",
            "result": None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "result": None
        }


def generate_python_code(task_description: str, context: str = None) -> Dict[str, Any]:
    """
    Generate Python code using GPT-5 based on task description.
    
    Args:
        task_description: What the code should do
        context: Optional context or constraints
        
    Returns:
        Dict with status, code, and explanation
    """
    try:
        # Build prompt
        system_prompt = (
            "You are an expert Python programmer. Generate clean, efficient, "
            "well-commented Python code based on the user's requirements.\n"
            "Return ONLY the Python code in a code block, with clear comments.\n"
            "Do not include explanations outside the code block."
        )
        
        user_prompt = f"Task: {task_description}"
        if context:
            user_prompt += f"\n\nAdditional context:\n{context}"
        
        user_prompt += (
            "\n\nGenerate Python code to accomplish this task. "
            "Include docstrings and comments. "
            "Make the code production-ready and handle edge cases."
        )
        
        llm = ChatOpenAI(model="gpt-5", temperature=0.2, max_tokens=2048)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        code_text = response.content
        
        # Extract code from markdown blocks if present
        if "```python" in code_text:
            # Extract code between ```python and ```
            import re
            pattern = r"```python\n(.*?)```"
            matches = re.findall(pattern, code_text, re.DOTALL)
            if matches:
                code_text = matches[0].strip()
        elif "```" in code_text:
            # Extract code between ``` and ```
            import re
            pattern = r"```\n(.*?)```"
            matches = re.findall(pattern, code_text, re.DOTALL)
            if matches:
                code_text = matches[0].strip()
        
        # Validate syntax
        try:
            ast.parse(code_text)
            syntax_valid = True
            syntax_error = None
        except SyntaxError as e:
            syntax_valid = False
            syntax_error = str(e)
        
        return {
            "status": "success",
            "code": code_text,
            "syntax_valid": syntax_valid,
            "syntax_error": syntax_error,
            "task": task_description,
            "model": "gpt-5"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "task": task_description
        }


if __name__ == "__main__":
    """Test the code execution tool"""
    
    print("=" * 70)
    print("CODE EXECUTION TOOL TEST")
    print("=" * 70)
    
    # Test 1: Simple calculation
    print("\n[TEST 1] Simple calculation:")
    result = execute_calculation("2 + 2 * 3")
    print(f"Expression: 2 + 2 * 3")
    print(f"Result: {result}")
    
    # Test 2: Math functions
    print("\n[TEST 2] Math functions:")
    result = execute_calculation("sqrt(16) + pow(2, 3)")
    print(f"Expression: sqrt(16) + pow(2, 3)")
    print(f"Result: {result}")
    
    # Test 3: Python code with loop
    print("\n[TEST 3] Python code with loop:")
    code = """
total = 0
for i in range(1, 11):
    total += i
print(f"Sum of 1 to 10: {total}")
result = total
"""
    result = execute_python_code(code)
    print(f"Code:\n{code}")
    print(f"Result: {result}")
    
    # Test 4: Code with imports
    print("\n[TEST 4] Code with datetime:")
    code = """
from datetime import datetime
now = datetime.now()
print(f"Current time: {now}")
result = str(now)
"""
    result = execute_python_code(code)
    print(f"Code:\n{code}")
    print(f"Result: {result}")
    
    # Test 5: Generate code
    print("\n[TEST 5] Generate Python code:")
    task = "Create a function to calculate the factorial of a number recursively"
    result = generate_python_code(task)
    print(f"Task: {task}")
    print(f"Generated code:\n{result.get('code')}")
    print(f"Syntax valid: {result.get('syntax_valid')}")

