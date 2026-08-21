# ME360-Fall2026

# Python Setup and File Execution Guide

This guide provides instructions on how to set up Python and run Python files using two methods: locally on your computer and using Google Colab.

## Option 1: Using Google Colab

Google Colab is a cloud-based platform that allows you to write and execute Python code in your browser.

### Step 1: Access Google Colab

1. Go to https://colab.research.google.com/
2. Sign in with your Google account.

### Step 2: Create a New Notebook or Open an Existing One

- To create a new notebook, click on "New Notebook".
- To open an existing notebook from your Google Drive, click on "File" > "Open notebook" and select the file.

### Step 3: Write or Upload Your Code

- You can write Python code directly in the Colab notebook cells.
- To upload a local Python file:
  1. Click on the folder icon in the left sidebar.
  2. Click on the upload icon and select your Python file.
  3. Once uploaded, you can open and run the file in Colab.

### Step 4: Run the Code

- To run a single cell, click the play button next to the cell or press Shift+Enter.
- To run all cells, go to "Runtime" > "Run all".

### Note on Package Installation in Colab

Most common packages are pre-installed in Colab. If you need additional packages, you can install them using:

```python
!pip install package_name
```

Run this in a code cell before using the package.

## Option 2: Running Python Locally

### Step 1: Install Python

1. Visit the official Python website: https://www.python.org/downloads/
2. Download the latest version of Python for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the installation wizard. Make sure to check the box that says "Add Python to PATH" during installation.

### Step 2: Verify Installation

1. Open a command prompt (Windows) or terminal (macOS/Linux).
2. Type `python --version` and press Enter. You should see the Python version number.

### Step 3: Install Required Packages

1. In the command prompt or terminal, use pip (Python's package manager) to install necessary packages:
   ```
   pip install numpy matplotlib pandas
   ```

### Step 4: Run Python Files

1. Navigate to the directory containing your Python file using the `cd` command.
2. Run the file using:
   ```
   python your_file_name.py
   ```

## Additional Resources

- Python Documentation: https://docs.python.org/
- Google Colab Documentation: https://colab.research.google.com/notebooks/basic_features_overview.ipynb

For any issues or further assistance, please consult these resources or seek help from the course TA-s.
