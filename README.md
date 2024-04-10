# Situational_Awareness_Learning

To install dependecies, first install the conda environment:
`conda env create -f environment.yml`

We require the Carla Python API as well. If you have built carla, it is enough to simply install the generated wheel ( usually in `${CARLA_ROOT}\PythonAPI\dist\`):
`python -m pip install --user carla-0.9.13-cp37-cp37m-linux_x86_64.whl`