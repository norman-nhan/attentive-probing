from setuptools import setup

NAME = "attentive-probing"
VERSION = "1.0.0"
DESCRIPTION = "Implementation Pytorch code for Attentive Probing-based Classifier to detect Atomic Operation in FineBio dataset."


def get_requirements():
    with open("./requirements.txt") as reqsf:
        reqs = reqsf.readlines()
    return reqs


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        install_requires=get_requirements(),
        packages=["src"]
    )
