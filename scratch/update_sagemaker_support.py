import json

with open("04_advanced_ngram_graph_solution.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Cell 1 (index 2 in cells)
cell1_source = [
    "import os\n",
    "import sys\n",
    "\n",
    "# Auto-detect Cloud Environment (SageMaker, Colab, etc.)\n",
    "IN_COLAB = 'google.colab' in sys.modules\n",
    "IN_SAGEMAKER = os.path.exists('/opt/ml') or os.path.exists('/home/sagemaker-user') or os.path.exists('/opt/conda')\n",
    "IS_CLOUD = IN_COLAB or IN_SAGEMAKER or not os.path.exists(\"dataset_folder\")\n",
    "\n",
    "if IS_CLOUD:\n",
    "    print(f\"🚀 Running in Cloud Environment (Colab: {IN_COLAB}, SageMaker: {IN_SAGEMAKER})\")\n",
    "    !nvidia-smi\n",
    "    # Fix the huggingface_hub HfFolder incompatibility and install required packages\n",
    "    !pip install -q -U \"huggingface_hub<0.25.0\" datasets sentence-transformers rapidfuzz lightgbm scikit-learn\n",
    "    \n",
    "    # Download dataset from Unstop CDN if not already present\n",
    "    if not os.path.exists(\"student_resource/dataset\") and not os.path.exists(\"/content/student_resource/dataset\"):\n",
    "        if not os.path.exists(\"6ab10eb3b23ba_student_resource.zip\"):\n",
    "            print(\"Downloading 1.09 GB archive directly from Unstop CDN...\")\n",
    "            !wget -q --show-progress https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip\n",
    "        \n",
    "        dest_dir = \"/content/\" if IN_COLAB else \"./\"\n",
    "        print(f\"Extracting dataset to {dest_dir}...\")\n",
    "        !unzip -q 6ab10eb3b23ba_student_resource.zip -d {dest_dir}\n",
    "        print(\" Dataset extracted successfully!\")\n",
    "        \n",
    "    if IN_COLAB:\n",
    "        DATA_DIR = \"/content/student_resource/dataset\"\n",
    "        UTILS_DIR = \"/content/student_resource/utils\"\n",
    "    else:\n",
    "        DATA_DIR = os.path.abspath(\"student_resource/dataset\")\n",
    "        UTILS_DIR = os.path.abspath(\"student_resource/utils\")\n",
    "else:\n",
    "    print(\"💻 Running in local environment.\")\n",
    "    DATA_DIR = os.path.abspath(os.path.join(\"dataset_folder\", \"student_resource\", \"dataset\"))\n",
    "    UTILS_DIR = os.path.abspath(os.path.join(\"dataset_folder\", \"student_resource\", \"utils\"))\n",
    "\n",
    "TRAIN_DIR = os.path.join(DATA_DIR, \"train\")\n",
    "TEST_DIR = os.path.join(DATA_DIR, \"test\")\n",
    "\n",
    "print(\"Train Directory:\", TRAIN_DIR)\n",
    "print(\"Test Directory :\", TEST_DIR)\n"
]

nb["cells"][1]["source"] = cell1_source

with open("04_advanced_ngram_graph_solution.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("04_advanced_ngram_graph_solution.ipynb updated with SageMaker auto-detection and HfFolder fix!")
