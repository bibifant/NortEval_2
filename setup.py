from setuptools import setup, find_packages

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='nortal-llm',
    version='0.1',
    authors="Yao Yao Zhu, Julia Schultze, Duong Nguyen, Laura Hartstein, Christina Gottschalk, Julia Dusan",
    description="Evaluationstool zur Berwertung von LLMs hinsichtlich der Qualität ihres Outputs in deutscher Sprache",
    url="https://gitlab.rz.htw-berlin.de/Christina.Gottschalk/nortal-llm",
    packages=find_packages(),
    install_requires=[
        "json",
        "rouge-score",
        "nltk",
        "spacy",
        "langdetect",
        "math",
        "torch",
        "transformers",
        "rouge_score",
        "sys"
    ],
    python_requires=">=3.11",
)
