from setuptools import setup, find_packages

setup(
    name="DiffResBP_Simulator",
    version="0.1.0",
    include_package_data=True,
    package_dir={
        "": ".",
        "bpdfr_simulation_engine": "bpdfr_simulation_engine",
    },
    install_requires=["click", "simpy"],
    entry_points={
        "console_scripts": [
            "diff_res_bpsim = diff_res_bpsim:cli",
        ]
    },
)
