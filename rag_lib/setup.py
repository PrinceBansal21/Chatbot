from setuptools import setup, find_packages

# Map the package 'rag_lib' to the current directory '.'
# and map subpackages found in '.' to 'rag_lib.subpackage'
packages = ['rag_lib']
for pkg in find_packages(where='.'):
    packages.append(f'rag_lib.{pkg}')

setup(
    name="rag_lib",
    version="0.1.0",
    description="Internal RAG Services Library (Chunking, Embeddings, Markdown)",
    author="BPS-SmartOps",
    packages=packages,
    package_dir={'rag_lib': '.'},
    install_requires=[
        "langchain>=1.2.6",
        "langchain-text-splitters>=1.1.2",
        "tiktoken",
        "langchain-aws",
        "numpy",
    ],
    extras_require={
        "all": [
            "docling>=2.74.0",
            "docling-ibm-models",
        ],
        "docling": ["docling>=2.74.0", "docling-ibm-models"],
    },
    python_requires=">=3.12",
)
