import os

def create_folder_structure():
    folders = [
        'data',
        'models',
        'logs',
        'examples'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}/")
    
    print("\nFolder structure created!")
    print("Please place your dataset in: data/Medicinal Leaf dataset/")
    print("Folder structure should be: data/Medicinal Leaf dataset/Class1/, data/Medicinal Leaf dataset/Class2/, etc.")

if __name__ == "__main__":
    create_folder_structure()