# **Python Project Standards and Contribution Guide**

This document outlines the coding standards, best practices, and contribution guidelines we follow to ensure our codebase remains clean, readable, maintainable, and robust. Adhering to these standards helps us collaborate effectively and build a high-quality project.

## **1\. Python Coding Standards (PEP 8 Adherence)**

Our primary coding standard is **PEP 8**, the official style guide for Python code. Consistency is key\!

### **General Formatting & Layout**

* **Indentation:** Use 4 spaces per indentation level. **Never** use tabs.  
* **Line Length:** Limit all lines to a maximum of **79 characters**. For longer lines, use parentheses for implicit line continuation.  
  \# Good  
  long\_variable\_name \= (  
      first\_part \+ second\_part \+ third\_part  
  )

  \# Bad  
  long\_variable\_name \= first\_part \+ second\_part \+ third\_part \+ fourth\_part \+ fifth\_part \# This line is too long\!

* **Blank Lines:**  
  * Separate top-level function and class definitions with **two blank lines**.  
  * Separate method definitions inside a class with **one blank line**.  
  * Use blank lines sparingly within functions to indicate logical sections.  
* **Whitespace:**  
  * Surround binary operators (e.g., \=, \+, \==) with a single space.  
  * Avoid extraneous whitespace inside parentheses, brackets, or braces.

\# Good  
x \= 1 \+ 2  
my\_list \= \[1, 2, 3\]

\# Bad  
x=1+2  
my\_list \= \[ 1,2,3 \]

* **Trailing Commas:** Prefer adding a trailing comma for multiline lists, tuples, and dictionaries for easier diffs and reordering.

### **Naming Conventions**

* **Modules:** Short, lowercase names, with underscores if it improves readability (e.g., my\_module.py, data\_processor.py).  
* **Packages:** Short, lowercase names (e.g., my\_package).  
* **Classes:** **CamelCase** (e.g., MyClass, LtopClient).  
* **Functions & Variables:** **snake\_case** (lowercase with underscores) (e.g., my\_function, user\_name, calculate\_sum).  
* **Methods:** Follow function naming rules. For internal methods, use a single leading underscore (e.g., \_internal\_method).  
* **Constants:** **ALL\_CAPS\_WITH\_UNDERSCORES** (e.g., MAX\_RETRIES, DEFAULT\_TIMEOUT).  
* **Avoid:** Single-character variable names (except for loop counters like i, j), and the letters l, O, I (as they can be confused with 1 and 0).

### **Imports**

* Place all import statements at the **top of the file**, immediately after any module docstrings.  
* Group imports in the following order, separated by a blank line:  
  1. Standard library imports (e.g., os, sys)  
  2. Third-party library imports (e.g., google.generativeai, requests)  
  3. Local application/library-specific imports  
* Avoid wildcard imports (from module import \*).

### **Comments & Docstrings**

* **Comments:** Use sparingly. Explain *why* the code does something, not *what* it does if it's obvious from the code itself. Start block comments with \# followed by a single space.  
* **Docstrings:**  
  * Write docstrings for all public modules, functions, classes, and methods.  
  * Follow **PEP 257** conventions. Use triple double-quotes ("""Docstring content""").  
  * For one-liner docstrings: """Return the sum of two numbers."""  
  * For multi-line docstrings:  
    def my\_function(param1, param2):  
        """  
        Brief summary of the function.

        Detailed explanation of what the function does, its purpose,  
        and any specific logic or assumptions.

        Args:  
            param1 (type): Description of param1.  
            param2 (type): Description of param2.

        Returns:  
            type: Description of the return value.  
        """  
        pass

## **2\. Best Practices for Gemini CLI Project**

### **Virtual Environments**

Always use a **virtual environment** for your project dependencies. This prevents conflicts and ensures reproducibility.

python3 \-m venv venv  
source venv/bin/activate  \# On Linux/macOS  
\# .\\venv\\Scripts\\activate  \# On Windows

### **Dependency Management**

Use pip to manage dependencies and keep requirements.txt up-to-date.

pip install \-r requirements.txt  
pip freeze \> requirements.txt \# After adding new dependencies

### **Error Handling**

Handle exceptions gracefully using try-except blocks. Avoid bare except clauses; specify the exception type.

try:  
    \# Code that might raise an exception  
    result \= 10 / 0  
except ZeroDivisionError:  
    print("Error: Division by zero is not allowed.")  
except Exception as e:  
    print(f"An unexpected error occurred: {e}")

### **Logging**

Use Python's built-in logging module for application events, rather than print() statements for debugging. This allows for configurable log levels and output destinations.

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s \- %(levelname)s \- %(message)s')

def process\_data():  
    logging.info("Starting data processing.")  
    try:  
        \# ... processing logic ...  
        logging.debug("Intermediate step: data transformed.")  
    except Exception as e:  
        logging.error(f"Failed to process data: {e}")  
    logging.info("Data processing complete.")

### **Modularity**

Break down your code into smaller, reusable functions and classes. Each function or class should have a single, clear responsibility. This improves readability, testability, and maintainability.

### **Testing**

Write **unit tests** for your code using unittest or pytest. Aim for good test coverage to ensure reliability and prevent regressions.

\# Example using pytest  
pytest

## **3\. Gemini CLI Specific Guidelines**

* **API Key Management:** Do not hardcode your Gemini API key directly in the code. Use environment variables (e.g., GEMINI\_API\_KEY) or a secure configuration management system.  
* **Model Selection:** Be explicit about the Gemini model you are using (e.g., gemini-pro, gemini-1.5-flash).  
* **Tool Usage:** When interacting with the Gemini API and defining custom tools, ensure your tool specifications are clear, concise, and accurately reflect their functionality.  
* **Rate Limits & Exponential Backoff:** Implement exponential backoff for API calls to handle potential rate limits and transient errors gracefully.  
* **User Interaction:** If your CLI interacts with users, provide clear prompts, informative feedback, and handle user input robustly.

## **4\. Contribution Workflow**

1. **Fork the Repository:** Start by forking the main repository to your GitHub account.  
2. **Clone Your Fork:** Clone your forked repository to your local machine.  
3. **Create a New Branch:** For each new feature or bug fix, create a new branch from main (e.g., feature/add-new-cli-command, fix/bug-in-api-call).  
4. **Make Your Changes:** Implement your changes, adhering to the coding standards outlined above.  
5. **Write Tests:** Add or update unit tests to cover your changes.  
6. **Run Tests:** Ensure all existing and new tests pass.  
7. **Format Code:** Use a code formatter like Black or Ruff to automatically apply PEP 8\.  
   pip install black  
   black .

8. **Commit Your Changes:** Write clear, concise commit messages. Follow a conventional commit style if applicable.  
   git commit \-m "feat: Add new CLI command for image generation"

9. **Push to Your Fork:** Push your new branch to your forked repository.  
10. **Open a Pull Request (PR):** Open a pull request from your branch to the main branch of the original repository.  
    * Provide a descriptive title and detailed description of your changes.  
    * Reference any related issues (e.g., Fixes \#123).  
11. **Code Review:** Participate in the code review process, addressing feedback from maintainers.

Thank you for contributing to this Python project\! Your efforts help us make this project better for everyone.
