import os
import json

def extract_code_from_notebook(notebook_path, output_path):
    """Extracts code cells from a Jupyter notebook and saves to a .py file."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_content = json.load(f)
        
        code_cells = []
        for cell in nb_content.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = cell.get('source', [])
                if isinstance(source, list):
                    code_cells.append("".join(source))
                else:
                    code_cells.append(source)
                code_cells.append("\n\n") # Add spacing between cells
        
        if code_cells:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("".join(code_cells))
            print(f"Successfully extracted: {os.path.basename(notebook_path)} -> {os.path.basename(output_path)}")
        else:
            print(f"No code cells found in: {os.path.basename(notebook_path)}")
            
    except Exception as e:
        print(f"Error processing {os.path.basename(notebook_path)}: {e}")

def main():
    notebooks_dir = 'notebooks'
    output_dir = 'extracted_scripts'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    notebook_files = [f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb')]
    
    if not notebook_files:
        print(f"No notebook files found in {notebooks_dir}")
        return
    
    for notebook_file in notebook_files:
        notebook_path = os.path.join(notebooks_dir, notebook_file)
        # Handle case where notebook might have dots in name other than extension
        base_name = os.path.splitext(notebook_file)[0]
        output_file = f"{base_name}.py"
        output_path = os.path.join(output_dir, output_file)
        
        extract_code_from_notebook(notebook_path, output_path)

if __name__ == "__main__":
    main()
