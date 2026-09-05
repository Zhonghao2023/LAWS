from setuptools import find_packages, setup
from pathlib import Path
from laws import __version__, __author__, __email__

setup(
      name='LAWS',
      version=__version__,
      description='LAWS: an open-source coupled water and crop model based on CWatM',
      long_description=Path("README.md").read_text(encoding="utf-8"),
      long_description_content_type="text/markdown",
      license='GPLv3',
      classifiers=[
            'License :: OSI Approved :: GNU General Public License v3',
            'Operating System :: OS Independent',
      ],
      url='',
      author=__author__,
      author_email=__email__,
      packages=find_packages(include=['laws', 'laws.*']),
      package_data={
            'laws': [
                  'metaNetcdf.xml',
            ],
            'laws.hydrological_modules': [
                  'routing_reservoirs/t5.dll',
                  'routing_reservoirs/t5cyg.so',
                  'routing_reservoirs/t5_linux.o',
                  'routing_reservoirs/t5_linux.so',
                  'routing_reservoirs/t5.cpp',
            ],
      },
      zip_safe=True,
      install_requires=[
            'numpy',
            'scipy',
            'netCDF4',
            'GDAL',
            'pandas',
      ],
      extras_require={
            'modflow': ['flopy>=3.3.2', 'xmipy'],
            'dev': ['pytest', 'pytest-html', 'GitPython'],
      },
      python_requires='>=3.8',
      entry_points={
            'console_scripts': [
                  'laws=laws.run_laws:run_from_command_line',
            ]
      }
)
